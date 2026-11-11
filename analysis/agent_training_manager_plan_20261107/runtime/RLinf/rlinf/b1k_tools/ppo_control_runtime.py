# Copyright 2026 The RLinf Authors.
"""Opt-in actor/critic scheduling for complete on-policy rollout batches."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from rlinf.algorithms.registry import calculate_adv_and_returns
from rlinf.b1k_tools.ppo_update_controller import UpdateController, atomic_json, schedule
from rlinf.utils.nested_dict_process import process_nested_dict_for_train, put_tensor_device, split_dict_to_chunk


def enabled(worker):
    return bool(worker.cfg.algorithm.get("ppo_control", {}).get("enable", False))


def parameter_groups(worker):
    actor, critic = [], []
    for name, parameter in worker.model.named_parameters():
        if parameter.requires_grad:
            (critic if "value_head" in name else actor).append(parameter)
    if not actor or not critic:
        raise ValueError("PPO control requires separate trainable actor and value-head parameters")
    return actor, critic


def digest(parameters, optimizer=None):
    result = hashlib.sha256()
    for parameter in parameters:
        values = [parameter] if optimizer is None else list(optimizer.state.get(parameter, {}).values())
        for value in values:
            if torch.is_tensor(value):
                data = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
                result.update(data.numpy().tobytes())
            else:
                result.update(repr(value).encode())
    return result.hexdigest()


def monte_carlo_returns(worker, batch):
    return calculate_adv_and_returns(
        task_type="embodied", adv_type="gae", rewards=batch["rewards"], dones=batch["dones"],
        values=batch["prev_values"], prev_logprobs=batch["prev_logprobs"],
        num_action_chunks=worker.cfg.actor.model.num_action_chunks,
        gamma=worker.cfg.algorithm.gamma, gae_lambda=1.0,
        group_size=1, reward_type="chunk_level", loss_mask=batch.get("loss_mask"),
        loss_mask_sum=batch.get("loss_mask_sum"),
    )["returns"]


def rollout_summary(worker):
    batch = worker.rollout_batch
    target = monte_carlo_returns(worker, batch).double()
    values = batch["prev_values"][:-1].double()
    mask = batch["loss_mask"].bool().expand_as(target)
    target, values = target[mask], values[mask]
    error = target - values
    sums = torch.tensor([target.numel(), target.sum().item(), target.square().sum().item(),
                         error.sum().item(), error.square().sum().item()],
                        dtype=torch.float64, device=worker.device)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(sums)
    count, total, squares, error_sum, error_squares = sums.cpu().tolist()
    variance = max(0.0, squares / max(count, 1) - (total / max(count, 1)) ** 2)
    error_variance = max(0.0, error_squares / max(count, 1) - (error_sum / max(count, 1)) ** 2)
    summary = {"valid_chunks": int(count), "return_mean": total / max(count, 1),
               "return_variance": variance, "value_mse": error_squares / max(count, 1),
               "value_informative": variance > 1e-8,
               "value_explained_variance": 1 - error_variance / variance if variance > 1e-8 else 0.0}
    return summary


class UpdateSession:
    def __init__(self, worker):
        self.worker = worker
        self.config = worker.cfg.algorithm.ppo_control
        if worker.cfg.algorithm.adv_type != "gae" or worker.cfg.algorithm.logprob_type != "chunk_level":
            raise ValueError("PPO control requires chunk-level GAE PPO")
        if worker.cfg.algorithm.get("agent_manager", {}).get("enable", False):
            raise ValueError("The sampling/LR manager must be disabled for the PPO-control experiment")
        if not hasattr(worker, "_update_controller"):
            worker._update_controller = UpdateController(
                self.config, getattr(worker, "_ppo_control_resume_state", None))
        self.controller = worker._update_controller
        self.iteration = worker.version + 1
        self.summary = rollout_summary(worker)
        self.directory = Path(self.config.out_dir)
        if worker._rank == 0:
            expected = worker.cfg.env.train.total_num_envs * worker.cfg.env.train.rollout_epoch
            records = worker._am_read_new(worker.cfg.env.train.task_stats_dir, expected, 180)
            counts = {}
            for record in records:
                row = counts.setdefault(str(record["task"]), [0, 0])
                row[0] += 1
                row[1] += int(record["s"])
            if len(records) != expected:
                raise ValueError(f"Expected {expected} training episodes, got {len(records)}")
            self.summary["training_task_counts"] = counts
            decision = self.controller.choose(self.iteration, self.summary, self.ask)
            message = [decision, self.summary]
        else:
            message = [None, None]
        if torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(message, src=0, device=worker.device)
        self.decision, self.summary = message
        self.controller.mode = self.decision["mode"]
        self.controller.last_decision = self.decision
        self.critic_passes, self.joint_passes = schedule(self.decision["mode"])
        self.epochs = self.critic_passes + self.joint_passes
        self.original = worker.rollout_batch
        self.actor_steps = 0
        self.value_steps = 0
        self.final_kl = 0.0
        self.stopped = False
        self.actor_parameters, self.critic_parameters = parameter_groups(worker)
        self.actor_digest = None
        self.optimizer_digest = None
        self.old_logprobs = None
        if self.critic_passes and self.config.get("verify_actor_identity", True):
            self.actor_digest = digest(self.actor_parameters)
            self.optimizer_digest = digest(self.actor_parameters, worker.optimizer)
            self.old_logprobs = self.original["prev_logprobs"].clone()

    def ask(self, prompt, schema):
        from rlinf.b1k_tools.agent_manager import ask_agent, ask_relay
        output = self.directory / f"decision_{self.iteration:03d}"
        if self.config.policy == "relay":
            return ask_relay(prompt, output, self.directory.parent.name, timeout=180, poll_s=2, schema=schema)
        return ask_agent(prompt, output, timeout=180, schema=schema)

    def step(self, critic_only):
        worker = self.worker
        if critic_only:
            for parameter in self.actor_parameters:
                parameter.grad = None
            worker.grad_scaler.unscale_(worker.optimizer)
            norm = worker._strategy.clip_grad_norm_(model=worker.model)
            if not torch.isfinite(torch.as_tensor(norm)):
                raise FloatingPointError("Non-finite critic gradient")
            worker.grad_scaler.step(optimizer=worker.optimizer)
            worker.grad_scaler.update()
            self.value_steps += 1
            return norm, [group["lr"] for group in worker.optimizer.param_groups]
        self.actor_steps += 1
        return worker.optimizer_step()

    def micro_batches(self, prefix_store):
        worker = self.worker
        rows = worker.rollout_batch["prev_logprobs"].shape[0]
        per_rank = worker.cfg.actor.global_batch_size // worker._world_size
        for global_index, global_batch in enumerate(split_dict_to_chunk(worker.rollout_batch, rows // per_rank)):
            count = global_batch["prev_logprobs"].shape[0] // worker.cfg.actor.micro_batch_size
            for micro_index, batch in enumerate(split_dict_to_chunk(global_batch, count)):
                if prefix_store is not None:
                    prefix_store.slot = (global_index, micro_index)
                yield put_tensor_device(batch, worker.device)

    @torch.no_grad()
    def after_epoch(self, critic_only, shuffle_id, prefix_store):
        worker = self.worker
        if critic_only:
            if self.actor_digest is not None:
                if digest(self.actor_parameters) != self.actor_digest:
                    raise RuntimeError("Critic-only pass changed actor parameters")
                if digest(self.actor_parameters, worker.optimizer) != self.optimizer_digest:
                    raise RuntimeError("Critic-only pass changed actor optimizer state")
            if not self.original["dones"][-1].any(dim=-1).all():
                raise ValueError("Critic refresh requires complete episodes with masked terminal bootstrap")
            predictions = []
            for batch in self.micro_batches(prefix_store):
                with worker.amp_context:
                    output = worker.model(forward_inputs=batch["forward_inputs"], compute_values=True)
                predictions.append(output["values"].detach().cpu())
            flat_values = torch.cat(predictions)[torch.argsort(shuffle_id).cpu()]
            previous = self.original["prev_values"]
            previous[:-1] = flat_values.reshape_as(previous[:-1]).to(previous.device, previous.dtype)
            worker.rollout_batch = self.original
            worker.compute_advantages_and_returns()
            if self.old_logprobs is not None and not torch.equal(self.original["prev_logprobs"], self.old_logprobs):
                raise RuntimeError("Value refresh changed behavior-policy log probabilities")
            worker.rollout_batch = process_nested_dict_for_train(self.original, shuffle_id)
            return False
        totals = torch.zeros(2, dtype=torch.float64, device=worker.device)
        for batch in self.micro_batches(prefix_store):
            with worker.amp_context:
                output = worker.model(forward_inputs=batch["forward_inputs"], compute_values=False)
            ratio = (output["logprobs"].float() - batch["prev_logprobs"].float()).flatten(1).sum(-1)
            mask = batch["loss_mask"].reshape(ratio.shape[0], -1).any(-1)
            estimates = torch.expm1(ratio[mask]) - ratio[mask]
            if not torch.isfinite(estimates).all():
                raise FloatingPointError("Non-finite full-batch policy divergence")
            totals[0] += estimates.double().sum()
            totals[1] += mask.sum()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(totals)
        self.final_kl = (totals[0] / totals[1].clamp_min(1)).item()
        self.stopped = self.final_kl > float(self.config.kl_stop)
        return self.stopped

    def finish(self, metrics):
        metrics.update({"pc/final_kl": self.final_kl, "pc/early_stop": float(self.stopped),
                        "pc/actor_steps": float(self.actor_steps), "pc/critic_only_steps": float(self.value_steps),
                        "pc/mode": float(("standard", "short_update", "critic_first").index(self.decision["mode"]))})
        self.controller.record(self.iteration, self.summary, metrics)
        if self.worker._rank == 0:
            atomic_json(self.directory / "state.json", self.controller.state_dict())
            atomic_json(self.directory / f"iteration_{self.iteration:03d}.json",
                        self.controller.history[-1])
        return metrics
