# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.expert import build_expert_model_config
from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.algorithms.utils import compute_entropy_loss
from rlinf.config import SupportedModel
from rlinf.data.schema.embodied_types import Trajectory
from rlinf.data.storage.lerobot import resolve_lerobot_repo_id
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.hybrid_engines.weight_syncer import WeightSyncer
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.distributed import (
    all_reduce_dict,
)
from rlinf.utils.metric_utils import (
    CRITIC_EXPLAINED_VARIANCE_KEY,
    append_to_dict,
    compute_critic_explained_variance_from_stats,
    compute_group_success_metrics,
    compute_loss_mask,
    compute_rollout_metrics,
    compute_split_num,
    pop_critic_explained_variance_stats,
)
from rlinf.utils.nested_dict_process import (
    flatten_nested_tensor_time_batch,
    process_nested_dict_for_adv,
    process_nested_dict_for_train,
    put_tensor_device,
    split_dict_to_chunk,
    trim_nested_tensor_time_dim,
)
from rlinf.utils.placement import (
    HybridComponentPlacement,
)
from rlinf.utils.utils import (
    clear_memory,
    get_loss_agg_func,
    masked_mean,
)


class EmbodiedFSDPActor(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)
        self.cfg = cfg
        self._env_group_name = cfg.env.group_name
        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        # stage_num: default to 2, use for pipeline rollout process
        self.stage_num = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.actor.get("enable_offload", False)
        self._opd_teacher_model = None
        self.entropy_op_type = self.cfg.algorithm.get("entropy_op_type", "torch")
        self.loss_agg_func = get_loss_agg_func(
            self.cfg.algorithm.get("loss_agg_func", "token-mean")
        )
        self._group_success_metrics: dict[str, float] = {}

        self.enable_sft_co_train = cfg.actor.get("enable_sft_co_train", False)
        self.version = 0
        if self.enable_sft_co_train:
            self._build_sft_data_loader()

        # create weight syncer
        weight_syncer_cfg = OmegaConf.select(cfg, "weight_syncer")
        self.weight_syncer = WeightSyncer.create(weight_syncer_cfg)

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        ), "global_batch_size is not divisible by micro_batch_size * world_size"

        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )
        self.update_epoch = self.cfg.algorithm.get("update_epoch", 1)

        self._sync_weight_comm_options = self.weight_syncer.comm_options

        self._is_weight_sender = self._rank == 0
        self._actor_world_size = self._world_size
        self._rollout_all_ranks = list(
            range(self._component_placement.get_world_size("rollout"))
        )

    def init_worker(self) -> None:
        """
        Initialize the actor worker. build the model and use corresponding training backend,
        if needed, offload model parameters and optimizer states to CPU.
        """
        self.setup_model_and_optimizer()

        if self.enable_offload:
            self.offload_param_and_grad()
            self.offload_optimizer()

    def model_provider_func(self) -> nn.Module:
        model = get_model(self.cfg.actor.model)
        if model is None:
            model = super().model_provider_func()

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            model.load_state_dict(model_dict)

        return model

    def get_rollout_state_dict(self) -> dict:
        return self.get_model_state_dict(cpu_offload=False, full_state_dict=False)

    @Worker.timer("actor/sync_model_to_rollout")
    async def sync_model_to_rollout(self) -> None:
        if self.enable_offload:
            if not self.is_optimizer_offloaded:
                self.offload_optimizer()

            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device, False)

        state_dict = self.get_rollout_state_dict()

        async def send_func(data):
            if not self._is_weight_sender:
                return
            await self.broadcast(
                data,
                groups=[
                    (self._group_name, 0),
                    (self._rollout_group_name, self._rollout_all_ranks),
                ],
                src=(self._group_name, 0),
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()

        async def recv_func():
            return await self.recv(
                src_group_name=self._rollout_group_name,
                src_rank=0,
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()

        if not self.weight_syncer.sender_initialized():
            await self.weight_syncer.init_sender(
                state_dict=state_dict,
                send=send_func,
                recv=recv_func,
                param_names_need_sync=self.param_names_need_sync,
                is_sender=self._is_weight_sender,
            )

        version = (
            self.get_rollout_sync_version()
            if hasattr(self, "get_rollout_sync_version")
            else self.version
        )
        await self.weight_syncer.sync(state_dict, send_func, version=version)

        if self.enable_offload:
            assert not self.is_weight_offloaded, (
                "weight should be offloaded in sync_model_to_rollout"
            )
            self.offload_param_and_grad(True)

    @Worker.timer("actor/recv_traj")
    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        """
        Receive rollout trajectories from rollout workers.

        Args:
            input_channel: The input channel to read from.
        """
        clear_memory(sync=False)

        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        recv_list = []
        for _ in range(split_num):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            recv_list.append(trajectory)

        self.rollout_batch = Trajectory.to_batch(recv_list)

        self.rollout_batch = self._process_received_rollout_batch(self.rollout_batch)

    def _process_received_rollout_batch(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
        target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
        """
        rollout_epoch = self.cfg.env.train.rollout_epoch
        rollout_batch = process_nested_dict_for_adv(rollout_batch, rollout_epoch)

        if (
            not self.cfg.env.train.auto_reset
            and not self.cfg.env.train.ignore_terminations
        ):
            dones = rollout_batch[
                "dones"
            ]  # [n_chunk_step, rollout_epoch x bsz, num_action_chunks]
            loss_mask, loss_mask_sum = compute_loss_mask(dones)

            if self.cfg.algorithm.reward_type == "chunk_level":
                loss_mask = loss_mask.any(dim=-1, keepdim=True)
                loss_mask_sum = loss_mask_sum[..., -1:]

            rollout_batch["loss_mask"] = loss_mask
            rollout_batch["loss_mask_sum"] = loss_mask_sum

        self._group_success_metrics = {}
        if self.cfg.algorithm.get("log_group_success_metrics", False):
            self._group_success_metrics = compute_group_success_metrics(
                rollout_batch["rewards"],
                group_size=self.cfg.algorithm.group_size,
                loss_mask=rollout_batch.get("loss_mask", None),
                rewards_lower_bound=self.cfg.algorithm.get("rewards_lower_bound", None),
                rewards_upper_bound=self.cfg.algorithm.get("rewards_upper_bound", None),
            )

        # filter data by rewards
        if self.cfg.algorithm.get("filter_rewards", False):
            rewards = rollout_batch[
                "rewards"
            ]  # [n_chunk_step, batch, num_action_chunks]
            if rollout_batch.get("loss_mask", None) is not None:
                rewards = rewards * rollout_batch["loss_mask"]
            n_chunk_step, batch_size, num_action_chunks = rewards.shape

            group_size = self.cfg.algorithm.group_size
            assert batch_size % group_size == 0, (
                f"batch {batch_size} not divisible by group_size {group_size}"
            )
            n_prompts = batch_size // group_size

            # calculate rewards by prompt
            rewards = rewards.transpose(
                0, 1
            )  # [batch, n_chunk_step, num_action_chunks]
            rewards = rewards.reshape(rewards.shape[0], -1)  # [batch, n_step]
            reward_matrix = rewards.reshape(
                n_prompts, group_size, rewards.shape[-1]
            )  # [n_prompts, group_size, n_step]
            reward_matrix = reward_matrix.sum(dim=-1)  # [n_prompts, group_size]
            mean_reward_in_group = reward_matrix.mean(dim=1)  # [n_prompts]

            # mask
            reward_filter_mask = (
                mean_reward_in_group >= self.cfg.algorithm.rewards_lower_bound
            ) & (
                mean_reward_in_group <= self.cfg.algorithm.rewards_upper_bound
            )  # [n_prompts]

            # extend mask dimension
            reward_filter_mask = reward_filter_mask.repeat_interleave(
                group_size
            )  # [batch]
            reward_filter_mask = (
                reward_filter_mask.unsqueeze(0).expand(n_chunk_step, -1).unsqueeze(-1)
            )  # [n_chunk_step, batch, 1]

            # update loss_mask
            if rollout_batch.get("loss_mask", None) is not None:
                rollout_batch["loss_mask"] = (
                    reward_filter_mask & rollout_batch["loss_mask"]
                )
            else:
                rollout_batch["loss_mask"] = reward_filter_mask

        return rollout_batch

    @Worker.timer("actor/compute_adv")
    def compute_advantages_and_returns(self) -> dict[str, torch.Tensor]:
        """
        Compute the advantages and returns.
        """
        if self.cfg.algorithm.adv_type == "opd":
            self.compute_opd_teacher_logprobs()

        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": self.rollout_batch["rewards"],
            "dones": self.rollout_batch["dones"],
            "values": self.rollout_batch.get("prev_values", None),
            "prev_logprobs": self.rollout_batch.get("prev_logprobs", None),
            "teacher_logprobs": self.rollout_batch.get("teacher_logprobs", None),
            "num_action_chunks": self.cfg.actor.model.num_action_chunks,
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "loss_mask": self.rollout_batch.get("loss_mask", None),
            "loss_mask_sum": self.rollout_batch.get("loss_mask_sum", None),
            "advantage_mode": self.cfg.algorithm.get("advantage_mode", None),
        }

        advantages_and_returns = calculate_adv_and_returns(**kwargs)

        self.rollout_batch.update(advantages_and_returns)
        if kwargs["loss_mask"] is not None:
            self.rollout_batch.update({"loss_mask": kwargs["loss_mask"]})
        if kwargs["loss_mask_sum"] is not None:
            self.rollout_batch.update({"loss_mask_sum": kwargs["loss_mask_sum"]})

        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        rollout_metrics.update(self._group_success_metrics)
        rollout_metrics.update(self._trajectory_length_metrics())
        return rollout_metrics

    def _trajectory_length_metrics(self) -> dict[str, float]:
        """Logging only (local addition 2026-11-03): valid length of successful vs failed trajectories and
        the share of total loss weight failed trajectories get under per-step (token-mean) vs
        per-trajectory (RLinf masked_mean_ratio) aggregation. Needs per-trajectory loss masks."""
        loss_mask_sum = self.rollout_batch.get("loss_mask_sum", None)
        if loss_mask_sum is None:
            return {}
        lengths = loss_mask_sum[0, :, 0].float()  # [B] valid env steps per trajectory
        returns = self.rollout_batch["rewards"].float().sum(dim=(0, 2))  # [B]
        success = (returns > 0).float().to(lengths.device)
        returns = returns.to(lengths.device)
        # Second definition (added after ppo_balanced started): a trajectory that ended before the
        # truncation limit terminated, which in LIBERO/CALVIN only happens on success.
        terminated = (lengths < float(self.cfg.env.train.max_episode_steps)).float()
        stats = torch.stack(
            [
                success.sum(),
                (1 - success).sum(),
                (lengths * success).sum(),
                (lengths * (1 - success)).sum(),
                terminated.sum(),
                (lengths * terminated).sum(),
                (lengths * (1 - terminated)).sum(),
            ]
        ).to(self.device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        n_succ, n_fail, len_succ, len_fail, n_term, len_term, len_trunc = [
            float(x) for x in stats.cpu()
        ]
        n = max(n_succ + n_fail, 1.0)
        n_trunc = n - n_term
        term_metrics = {
            "traj_term/terminated_frac": n_term / n,
            "traj_term/len_terminated_mean": len_term / max(n_term, 1.0),
            "traj_term/len_truncated_mean": len_trunc / max(n_trunc, 1.0),
            "traj_term/truncated_weight_share_tokenmean": len_trunc / max(len_term + len_trunc, 1.0),
            "traj_term/truncated_weight_share_balanced": n_trunc / n,
        }
        return term_metrics | {
            "traj/num": n,
            "traj/success_frac": n_succ / n,
            "traj/len_success_mean": len_succ / max(n_succ, 1.0),
            "traj/len_fail_mean": len_fail / max(n_fail, 1.0),
            "traj/fail_weight_share_tokenmean": len_fail / max(len_succ + len_fail, 1.0),
            "traj/fail_weight_share_balanced": n_fail / n,
        }

    @Worker.timer("actor/compute_opd_teacher_logprobs")
    def compute_opd_teacher_logprobs(self) -> None:
        assert self.rollout_batch.get("teacher_logprobs", None) is None, (
            "OPD teacher_logprobs must be computed after rollout on actor workers."
        )
        assert self.cfg.rollout.get("expert_model", None) is not None, (
            "OPD requires rollout.expert_model as teacher model config."
        )
        assert "forward_inputs" in self.rollout_batch, (
            "OPD teacher logprob computation requires rollout forward_inputs."
        )
        assert "prev_logprobs" in self.rollout_batch, (
            "OPD teacher logprob computation requires student prev_logprobs."
        )
        assert SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENVLA,
            SupportedModel.OPENVLA_OFT,
        ], "OPD teacher logprob computation currently supports OpenVLA models."

        prev_logprobs = self.rollout_batch["prev_logprobs"]
        time_dim, batch_dim = prev_logprobs.shape[:2]
        flat_batch_size = time_dim * batch_dim

        assert self.enable_offload and self.is_weight_offloaded, (
            "OPD teacher logprob computation expects actor weights to be "
            "offloaded before moving the teacher model to GPU."
        )
        teacher_model = self._get_opd_teacher_model()
        teacher_model.to(self.device)

        flat_forward_inputs = flatten_nested_tensor_time_batch(
            self.rollout_batch["forward_inputs"], ("forward_inputs",)
        )
        num_chunks = (
            flat_batch_size + self.cfg.actor.micro_batch_size - 1
        ) // self.cfg.actor.micro_batch_size
        teacher_logprobs = []
        kwargs = {
            "temperature": self.cfg.rollout.sampling_params.temperature_train,
            "top_k": self.cfg.rollout.sampling_params.top_k,
        }
        with torch.no_grad():
            for micro_batch in split_dict_to_chunk(flat_forward_inputs, num_chunks):
                micro_batch = put_tensor_device(micro_batch, self.device)
                with self.amp_context:
                    teacher_output = teacher_model(
                        forward_inputs=micro_batch,
                        compute_logprobs=True,
                        compute_entropy=False,
                        compute_values=False,
                        use_cache=False,
                        **kwargs,
                    )
                teacher_logprobs.append(teacher_output["logprobs"].detach().cpu())

        teacher_logprobs = torch.cat(teacher_logprobs, dim=0)
        expected_shape = (flat_batch_size, *prev_logprobs.shape[2:])
        assert teacher_logprobs.shape == expected_shape, (
            f"teacher_logprobs shape {teacher_logprobs.shape} must match "
            f"flattened student logprobs shape {expected_shape}."
        )
        self.rollout_batch["teacher_logprobs"] = teacher_logprobs.reshape(
            time_dim, batch_dim, *teacher_logprobs.shape[1:]
        )

        teacher_model.to("cpu")
        clear_memory()

    @Worker.timer("actor/recompute_logprobs")
    def recompute_prev_logprobs(self, batch_size_per_rank: int) -> dict[str, float]:
        """
        Recompute ``prev_logprobs`` with the actor's own forward, so both ends of
        the PPO ratio come from the same path. Runs after the shuffle and reuses
        the update loop's split, so each sample is scored in the micro-batch it
        will be trained in.

        Args:
            batch_size_per_rank: Samples per optimizer step on this rank.

        Returns:
            Dict with ``actor/rollout_train_logprob_gap`` (mean absolute logprob difference).
        """
        assert "forward_inputs" in self.rollout_batch, (
            "Recomputing logprobs requires rollout forward_inputs."
        )

        rollout_logprobs = self.rollout_batch["prev_logprobs"]
        rollout_size = rollout_logprobs.shape[0]
        kwargs = {
            "temperature": self.cfg.rollout.sampling_params.temperature_train,
            "top_k": self.cfg.rollout.sampling_params.top_k,
        }

        recomputed_logprobs = []
        with torch.no_grad():
            for global_batch in split_dict_to_chunk(
                self.rollout_batch, rollout_size // batch_size_per_rank
            ):
                for micro_batch in split_dict_to_chunk(
                    global_batch,
                    batch_size_per_rank // self.cfg.actor.micro_batch_size,
                ):
                    micro_batch = put_tensor_device(micro_batch, self.device)
                    with self.amp_context:
                        output = self.model(
                            forward_inputs=micro_batch["forward_inputs"],
                            compute_logprobs=True,
                            compute_entropy=False,
                            compute_values=False,
                            use_cache=False,
                            **kwargs,
                        )
                    recomputed_logprobs.append(output["logprobs"].detach().cpu())

        recomputed_logprobs = torch.cat(recomputed_logprobs, dim=0).to(
            rollout_logprobs.dtype
        )
        assert recomputed_logprobs.shape == rollout_logprobs.shape, (
            f"recomputed logprobs shape {recomputed_logprobs.shape} must match "
            f"rollout logprobs shape {rollout_logprobs.shape}."
        )
        clear_memory()

        # Logprobs are per action token, the mask and the advantages per action.
        action_dim = self.cfg.actor.model.get("action_dim", 7)
        log_ratio = (recomputed_logprobs - rollout_logprobs).reshape(
            *rollout_logprobs.shape[:-1], -1, action_dim
        )
        loss_mask = self.rollout_batch.get("loss_mask", None)
        mask = (
            None
            if loss_mask is None
            else loss_mask.bool().unsqueeze(-1).expand_as(log_ratio)
        )
        metrics = {
            "actor/rollout_train_logprob_gap": masked_mean(
                log_ratio.abs(), mask=mask
            ).item()
        }

        self.rollout_batch["prev_logprobs"] = recomputed_logprobs
        return metrics

    def _get_opd_teacher_model(self):
        if self._opd_teacher_model is None:
            teacher_model_config = build_expert_model_config(
                self.cfg, self.cfg.actor.model
            )
            teacher_model = get_model(teacher_model_config)
            if self.cfg.runner.get("expert_ckpt_path", None):
                teacher_model_dict = torch.load(
                    self.cfg.runner.expert_ckpt_path, map_location="cpu"
                )
                teacher_model.load_state_dict(teacher_model_dict)
            teacher_model.eval()
            teacher_model.requires_grad_(False)
            teacher_model.to("cpu")
            self._opd_teacher_model = teacher_model
        return self._opd_teacher_model

    def _build_sft_data_loader(self):
        if SupportedModel(self.cfg.actor.model.model_type) == SupportedModel.OPENPI:
            repo_id = resolve_lerobot_repo_id(self.cfg.actor.get("sft_data_path"))
            if repo_id is None:
                raise ValueError(
                    "actor.sft_data_path must be set to a local dataset path or "
                    "LeRobot repo id when enable_sft_co_train=True."
                )

            import openpi.training.data_loader as _data

            from rlinf.models.embodiment.openpi.dataconfig import (
                get_openpi_config,
            )

            training_config_name = OmegaConf.select(
                self.cfg.actor.model, "openpi.config_name", default=None
            )
            if not training_config_name:
                raise ValueError(
                    "enable_sft_co_train=True requires actor.model.openpi.config_name."
                )
            data_loader_config = get_openpi_config(
                training_config_name,
                model_path=self.cfg.actor.model.model_path,
                repo_id=repo_id,
                data_kwargs=getattr(self.cfg.actor.model, "openpi_data", None),
            )
            self.data_loader = _data.create_data_loader(
                data_loader_config, framework="pytorch", shuffle=True
            )
            self.sft_iterator = iter(self.data_loader)
            self.train_epoch = 0
            self.sft_loss_weight = self.cfg.actor.get("sft_loss_weight", 0.1)
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def _train_sft_epoch(
        self, metrics_data: dict[str, torch.Tensor], loss: torch.Tensor
    ) -> torch.Tensor:
        """
        Train one epoch of SFT.
        """
        metrics_data["ppo_loss"] = loss.clone().detach().item()

        # Get next data batch
        try:
            observation, actions = next(self.sft_iterator)
        except StopIteration:
            self.train_epoch += 1
            self.data_loader.set_epoch(self.train_epoch)
            self.sft_iterator = iter(self.data_loader)
            observation, actions = next(self.sft_iterator)

        sft_loss = self.model(
            data=(observation, actions),
            forward_type=ForwardType.SFT,
        )
        metrics_data["sft_loss"] = sft_loss.detach().item()
        total_loss = loss + self.sft_loss_weight * sft_loss
        loss = total_loss

        metrics_data["loss_ratio"] = (
            np.abs(metrics_data["sft_loss"]) / np.abs(metrics_data["ppo_loss"])
            if np.abs(metrics_data["ppo_loss"]) > 0
            else float("inf")
        )
        if metrics_data["loss_ratio"] > 1e5:
            self.logger.warning(
                "SFT/PPO loss imbalance detected: "
                f"ratio={metrics_data['loss_ratio']:.3e}, "
                f"sft_loss={metrics_data['sft_loss']:.6f}, "
                f"ppo_loss={metrics_data['ppo_loss']:.6f}, "
                f"sft_loss_weight={self.sft_loss_weight:.6f}"
            )
        return loss

    @Worker.timer("run_training")
    def run_training(self) -> None:
        """
        Run the training process using the received rollout batch.
        """
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

        pc_session = None
        if self.cfg.algorithm.get("ppo_control", {}).get("enable", False):
            from rlinf.b1k_tools.ppo_control_runtime import UpdateSession
            pc_session = UpdateSession(self)

        # agent manager (local addition 2026-11-05): per-iteration training manager (rlinf/b1k_tools/agent_manager.py)
        am_metrics = None
        if self.cfg.algorithm.get("agent_manager", {}).get("enable", False):
            am_metrics = self._agent_manage()

        # accept_rollback (local addition 2026-11-05): verified acceptance of the last update (see _ar_check)
        if self.cfg.algorithm.get("accept_rollback", {}).get("enable", False):
            ar_metrics = self._ar_check()
            if not ar_metrics["ar/accepted"]:
                self.optimizer.zero_grad()
                clear_memory()
                return ar_metrics
        else:
            ar_metrics = None

        if self.cfg.algorithm.loss_type == "opd":
            target_steps = int(self.rollout_batch["advantages"].shape[0])
            for key in [
                "prev_logprobs",
                "forward_inputs",
                "loss_mask",
                "loss_mask_sum",
            ]:
                assert key in self.rollout_batch, f"OPD training requires {key}."
                self.rollout_batch[key] = trim_nested_tensor_time_dim(
                    self.rollout_batch[key], target_steps, (key,)
                )

        self.model.train()
        rollout_size = (
            self.rollout_batch["prev_logprobs"].shape[0]
            * self.rollout_batch["prev_logprobs"].shape[1]
        )
        g = torch.Generator()
        g.manual_seed(self.cfg.actor.seed + self._rank)
        shuffle_id = torch.randperm(rollout_size, generator=g)

        with torch.no_grad():
            self.rollout_batch = process_nested_dict_for_train(
                self.rollout_batch, shuffle_id
            )

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        rollout_size = self.rollout_batch["prev_logprobs"].size(0)
        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        assert rollout_size % batch_size_per_rank == 0, (
            f"{rollout_size} is not divisible by {batch_size_per_rank}"
        )
        metrics = {}
        if self.cfg.algorithm.get("recompute_logprobs", False):
            append_to_dict(metrics, self.recompute_prev_logprobs(batch_size_per_rank))
        prefix_store = self._actor_prefix_store_for_training()
        if self.cfg.algorithm.loss_type == "espo":
            append_to_dict(
                metrics, self._espo_prepare(prefix_store, rollout_size, batch_size_per_rank)
            )
        update_epoch = pc_session.epochs if pc_session else self.cfg.algorithm.get("update_epoch", 1)
        self._espo_stopped = False
        for _epoch in range(update_epoch):
            self._ppo_control_critic_only = bool(pc_session and _epoch < pc_session.critic_passes)
            if self.cfg.algorithm.loss_type == "espo" and _epoch > 0 and not self._espo_stopped:
                # trust region: once the features are within stop_frac of the initial target shift, stop moving the
                # policy for the rest of this iteration (critic keeps training)
                d = metrics.get("espo/phi_target_dist", [])[-self._espo_mb_per_epoch:]
                shift0 = getattr(self, "_espo_target_shift", None)
                stop_frac = float(self.cfg.algorithm.get("espo", {}).get("stop_frac", 0.5))
                if d and shift0 and float(np.mean(d)) < stop_frac * shift0:
                    self._espo_stopped = True
                    append_to_dict(metrics, {"espo/stopped_at_epoch": float(_epoch)})
            rollout_dataloader_iter = split_dict_to_chunk(
                self.rollout_batch,
                rollout_size // batch_size_per_rank,
            )
            for global_idx, train_global_batch in enumerate(rollout_dataloader_iter):
                # split batch into micro_batches
                train_global_batch_size = train_global_batch["prev_logprobs"].shape[0]
                assert (
                    train_global_batch_size
                    == self.cfg.actor.global_batch_size
                    // torch.distributed.get_world_size()
                )
                assert train_global_batch_size % self.cfg.actor.micro_batch_size == 0, (
                    f"{train_global_batch_size=}, {self.cfg.actor.micro_batch_size}"
                )

                train_micro_batch = split_dict_to_chunk(
                    train_global_batch,
                    train_global_batch_size // self.cfg.actor.micro_batch_size,
                )

                self.optimizer.zero_grad()
                for idx, batch in enumerate(train_micro_batch):
                    if prefix_store is not None:
                        # Same (global, micro) slot holds the same samples in every update epoch.
                        prefix_store.slot = (global_idx, idx)
                    self.train_micro_batch(
                        micro_batch=batch,
                        metrics=metrics,
                        is_last=(idx + 1) == self.gradient_accumulation,
                    )
                    # avoid gpu memory leak
                    train_micro_batch[idx] = None
                    del batch

                self.torch_platform.empty_cache()

                grad_norm, lr_list = (pc_session.step(self._ppo_control_critic_only)
                                      if pc_session else self.optimizer_step())
                data = {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                }
                if len(lr_list) > 1:
                    data["critic/lr"] = lr_list[1]
                append_to_dict(metrics, data)
            if pc_session and pc_session.after_epoch(self._ppo_control_critic_only, shuffle_id, prefix_store):
                break
        self._ppo_control_critic_only = False
        prefix_cache_stats = (
            prefix_store.end_training_call() if prefix_store is not None else None
        )
        # put LR scheduler step here
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()
        explained_variance_stats = pop_critic_explained_variance_stats(metrics)
        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )
        if explained_variance_stats:
            reduced_stats = all_reduce_dict(
                explained_variance_stats, op=torch.distributed.ReduceOp.SUM
            )
            mean_metric_dict[CRITIC_EXPLAINED_VARIANCE_KEY] = (
                compute_critic_explained_variance_from_stats(reduced_stats).item()
            )
        if prefix_cache_stats is not None:
            mean_metric_dict.update(
                {f"prefix_cache/{k}": float(v) for k, v in prefix_cache_stats.items()}
            )
        if ar_metrics is not None:
            mean_metric_dict.update(ar_metrics)
        if am_metrics is not None:
            mean_metric_dict.update(am_metrics)
        if pc_session is not None:
            mean_metric_dict = pc_session.finish(mean_metric_dict)

        return mean_metric_dict

    # ---------------- agent manager (local addition 2026-11-05) ----------------
    def _am_read_new(self, d, expected, wait_s):
        """New (task, success) records of the train envs since the last call (own offsets; AR keeps its own)."""
        import glob
        import json as _json
        import time

        if not hasattr(self, "_am_offsets"):
            self._am_offsets = {}
        t0 = time.time()
        while True:
            recs = []
            offsets = dict(self._am_offsets)
            for path in sorted(glob.glob(f"{d}/env_*.jsonl")):
                with open(path) as f:
                    f.seek(offsets.get(path, 0))
                    chunk = f.read()
                cut = chunk.rfind("\n") + 1
                for line in chunk[:cut].splitlines():
                    if line.strip():
                        recs.append(_json.loads(line))
                offsets[path] = offsets.get(path, 0) + len(chunk[:cut].encode())
            if len(recs) >= expected or time.time() - t0 > wait_s:
                self._am_offsets = offsets
                return recs
            time.sleep(2.0)

    def _agent_manage(self) -> dict:
        mcfg = self.cfg.algorithm.agent_manager
        if not hasattr(self, "_am"):
            from rlinf.b1k_tools.agent_manager import Manager

            self._am = Manager(str(mcfg.get("mode", "agent")), str(mcfg.get("out_dir")),
                               int(self.cfg.runner.max_epochs), seed=int(mcfg.get("seed", 0)),
                               stats_dir=self.cfg.env.train.get("task_stats_dir"),  # agent manager robocasa (local addition 2026-11-05)
                               episodes_per_iter=int(self.cfg.env.train.total_num_envs)
                               * int(self.cfg.env.train.get("rollout_epoch", 1)),
                               version=str(mcfg.get("version", "v2")),  # manager v3 (local addition 2026-11-06)
                               replay_from=mcfg.get("replay_from", None), replay_lr=bool(mcfg.get("replay_lr", True)),
                               replay_weights=bool(mcfg.get("replay_weights", False)))  # manager replay (local addition 2026-11-06)
        mult = torch.ones(1, device=self.device)
        if self._rank == 0:
            from rlinf.b1k_tools.agent_manager import write_weights

            d = self.cfg.env.train.get("task_stats_dir")
            expected = int(self.cfg.env.train.total_num_envs) * int(self.cfg.env.train.get("rollout_epoch", 1))
            recs = self._am_read_new(d, expected, float(mcfg.get("wait_s", 180)))
            counts = {}
            for r in recs:
                # agent manager calvin (local addition 2026-11-06): a chain record counts every attempted subtask
                for t, s in (r["sub"] if "sub" in r else [(r["task"], r["s"])]):
                    v = counts.setdefault(int(t), [0, 0])
                    v[0] += 1
                    v[1] += int(s)
            weights, m, _info = self._am.decide({t: tuple(v) for t, v in counts.items()})
            write_weights(self.cfg.env.train.agent_weights_file,
                          self._am.env_weights() if hasattr(self._am, "env_weights") else weights)
            mult = torch.tensor([float(m)], device=self.device)
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(mult, src=0)
        m = float(mult.item())
        if abs(m - 1.0) > 1e-9:
            self.optimizer.param_groups[0]["lr"] *= m
            if hasattr(self.lr_scheduler, "base_lrs"):
                self.lr_scheduler.base_lrs[0] *= m
        return {"am/lr_multiplier": m, "am/lr": float(self.optimizer.param_groups[0]["lr"])}

    # ---------------- accept_rollback (local addition 2026-11-05; RSI option 1) ----------------
    def _ar_read_new(self, d, expected, wait_s):
        """New (task, success) records written by the train envs since the last call (all env_*.jsonl files)."""
        import glob
        import json as _json
        import time

        st = self._ar_state
        t0 = time.time()
        while True:
            recs = []
            offsets = dict(st["offsets"])
            for path in sorted(glob.glob(f"{d}/env_*.jsonl")):
                with open(path) as f:
                    f.seek(offsets.get(path, 0))
                    chunk = f.read()
                cut = chunk.rfind("\n") + 1  # complete lines only
                for line in chunk[:cut].splitlines():
                    if line.strip():
                        recs.append(_json.loads(line))
                offsets[path] = offsets.get(path, 0) + len(chunk[:cut].encode())
            if len(recs) >= expected or time.time() - t0 > wait_s:
                st["offsets"] = offsets
                return recs
            time.sleep(2.0)

    @staticmethod
    def _ar_drop(ref, new, min_n):
        """Largest one-sided drop (new below ref) among tasks with >= min_n episodes in both, plus the overall drop."""
        import math

        def z(p1, n1, p2, n2):
            p = (p1 * n1 + p2 * n2) / max(n1 + n2, 1)
            se = math.sqrt(max(p * (1 - p) * (1 / max(n1, 1) + 1 / max(n2, 1)), 1e-12))
            return (p1 - p2) / se

        worst = (0.0, 0.0, None)  # (drop, z, task)
        per_task = []  # accept_rollback v2.1: every eligible task's (drop, z, task) for the Bonferroni test
        for t, (n1, s1) in ref.items():
            n2, s2 = new.get(t, (0, 0))
            if n1 >= min_n and n2 >= min_n:
                d = s1 / n1 - s2 / n2
                per_task.append((d, z(s1 / n1, n1, s2 / n2, n2), t))
                if d > worst[0]:
                    worst = (d, z(s1 / n1, n1, s2 / n2, n2), t)
        EmbodiedFSDPActor._ar_last_per_task = per_task
        N1, S1 = sum(v[0] for v in ref.values()), sum(v[1] for v in ref.values())
        N2, S2 = sum(v[0] for v in new.values()), sum(v[1] for v in new.values())
        mean = (S1 / max(N1, 1) - S2 / max(N2, 1), z(S1 / max(N1, 1), N1, S2 / max(N2, 1), N2) if N1 and N2 else 0.0)
        return worst, mean

    def _ar_check(self) -> dict:
        import copy
        import json as _json
        import time

        acfg = self.cfg.algorithm.accept_rollback
        d = self.cfg.env.train.get("task_stats_dir")
        expected = int(self.cfg.env.train.total_num_envs) * int(self.cfg.env.train.get("rollout_epoch", 1))
        if not hasattr(self, "_ar_state"):
            self._ar_state = {"offsets": {}, "ref": None, "snap": None, "accepts": 0, "rollbacks": 0}
        st = self._ar_state
        dec = torch.zeros(8, device=self.device)  # accepted, n, worst drop, worst z, mean drop, mean z, move, remeasure
        info = {}
        if self._rank == 0:
            recs = self._ar_read_new(d, expected, float(acfg.get("wait_s", 180)))
            new = {}
            for r in recs:
                v = new.setdefault(int(r["task"]), [0, 0])
                v[0] += 1
                v[1] += int(r["s"])
            accepted, worst, mean = True, (0.0, 0.0, None), (0.0, 0.0)
            # accept_rollback v2.3: the rollout right after a rollback comes from the restored anchor policy itself -> pool it
            # into the anchor's reference and accept it without a test (opt-in: remeasure=True)
            remeasure = bool(acfg.get("remeasure", False)) and bool(st.get("restored")) and st["ref"] is not None and bool(recs)
            if remeasure:
                pooled = {t: list(v) for t, v in st["ref"].items()}
                for t, v in new.items():
                    pv = pooled.setdefault(t, [0, 0])
                    pv[0] += v[0]
                    pv[1] += v[1]
                st["ref"] = {t: tuple(v) for t, v in pooled.items()}
            elif st["ref"] is not None and recs:
                worst, mean = self._ar_drop(st["ref"], new, int(acfg.get("min_n", 8)))
                zc = float(acfg.get("z", 1.64))
                # accept_rollback v2.1: Bonferroni-corrected per-task threshold over the K eligible tasks
                from statistics import NormalDist
                per_task = getattr(EmbodiedFSDPActor, "_ar_last_per_task", [])
                zc_task = NormalDist().inv_cdf(1.0 - float(acfg.get("alpha", 0.05)) / max(len(per_task), 1))
                dt = float(acfg.get("delta_task", 0.15))
                hits = [p for p in per_task if p[0] > dt and p[1] > zc_task]
                bad_task = bool(hits)
                if hits:
                    worst = max(hits, key=lambda p: p[1])
                bad_mean = mean[0] > float(acfg.get("delta_mean", 0.08)) and mean[1] > zc
                accepted = not (bad_task or bad_mean)
            # accept_rollback v2 (2026-11-05): reference='best' keeps the best accepted rollout as the anchor (no downward
            # drift); the snapshot below only moves with the anchor (flag 6 of dec)
            move = accepted and not remeasure
            if move and recs and str(acfg.get("reference", "last")) == "best" and st["ref"] is not None:
                rate = lambda d: sum(v[1] for v in d.values()) / max(sum(v[0] for v in d.values()), 1)
                move = rate(new) >= rate(st["ref"])
            if move and recs:
                st["ref"] = {t: tuple(v) for t, v in new.items()}
            dec = torch.tensor([float(accepted), len(recs), worst[0], worst[1], mean[0], mean[1], float(move),
                                float(remeasure)], device=self.device)
            info = {"tasks": {str(t): v for t, v in sorted(new.items())}, "worst_task": worst[2],
                    "ref_tasks": {str(t): list(v) for t, v in sorted((st["ref"] or {}).items())}}
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(dec, src=0)
        accepted = bool(dec[0].item() > 0.5)
        params = [p for p in self.model.parameters() if p.requires_grad]
        if accepted or st["snap"] is None:
            if dec[6].item() > 0.5 or st["snap"] is None:
                # this rollout's policy is the new fallback point (v2 'best': only when it is the new anchor)
                # accept_rollback v2.2 (memory): drop the old snapshot first, copy tensors straight to host memory
                st["snap"] = None
                import gc

                gc.collect()
                st["snap"] = ([p.detach().to("cpu", copy=True) for p in params],
                              self._ar_to_cpu(self.optimizer.state_dict()))
            st["accepts"] += 1
            accepted = True
        else:
            with torch.no_grad():
                for p, s in zip(params, st["snap"][0]):
                    p.data.copy_(s.to(p.device, non_blocking=True))
            gamma = float(acfg.get("lr_decay", 0.5))
            lr_now = self.optimizer.param_groups[0]["lr"]  # accept_rollback v2.1: decay the lr in force, not the snapshot's
            # accept_rollback v2.2 (memory): no host deepcopy; load_state_dict copies the moments to the GPU params'
            # device, and CPU tensors it keeps by reference ('step') are cloned so the snapshot stays intact
            self.optimizer.load_state_dict(st["snap"][1])
            for _st in self.optimizer.state.values():
                for _k, _v in list(_st.items()):
                    if torch.is_tensor(_v) and _v.device.type == "cpu":
                        _st[_k] = _v.clone()
            self.optimizer.param_groups[0]["lr"] = lr_now * gamma
            if hasattr(self.lr_scheduler, "base_lrs"):
                self.lr_scheduler.base_lrs[0] *= gamma
            st["rollbacks"] += 1
        st["restored"] = not accepted  # accept_rollback v2.3: the next rollout re-measures the restored anchor
        metrics = {"ar/accepted": float(accepted), "ar/n_records": float(dec[1].item()), "ar/remeasure": float(dec[7].item()),
                   "ar/worst_task_drop": float(dec[2].item()), "ar/worst_task_z": float(dec[3].item()),
                   "ar/mean_drop": float(dec[4].item()), "ar/mean_z": float(dec[5].item()),
                   "ar/rollbacks": float(st["rollbacks"]), "ar/lr": float(self.optimizer.param_groups[0]["lr"])}
        if self._rank == 0 and acfg.get("log_path", None):
            with open(acfg.log_path, "a") as f:
                f.write(_json.dumps({"time": time.strftime("%H:%M:%S"), **metrics, **info}) + "\n")
        return metrics

    @staticmethod
    def _ar_to_cpu(obj):
        if torch.is_tensor(obj):
            return obj.detach().to("cpu", copy=True)
        if isinstance(obj, dict):
            return {k: EmbodiedFSDPActor._ar_to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(EmbodiedFSDPActor._ar_to_cpu(v) for v in obj)
        return obj

    def _actor_prefix_store_for_training(self):
        """Opt-in host cache of the frozen-VLM prefix across update epochs (actor.prefix_cache)."""
        if not self.cfg.actor.get("prefix_cache", False):
            return None
        store = getattr(self, "_prefix_store", None)
        if store is None:
            from rlinf.models.embodiment.openpi.actor_prefix_cache import (
                ActorPrefixStore,
            )
            from rlinf.models.embodiment.openpi.tasks.rl import Pi0RL

            targets = [m for m in self.model.modules() if isinstance(m, Pi0RL)]
            assert len(targets) == 1, f"expected one Pi0RL module, found {len(targets)}"
            assert targets[0].rl_cfg.train_expert_only, (
                "actor.prefix_cache requires train_expert_only (frozen VLM prefix)"
            )
            assert not self.cfg.algorithm.get("recompute_logprobs", False)
            store = ActorPrefixStore(
                pin=self.cfg.actor.get("prefix_cache_pin", True),
                verify_every=self.cfg.actor.get("prefix_cache_verify_every", 0),
                max_gib=self.cfg.actor.get("prefix_cache_max_gib", 0.0),
            )
            targets[0]._actor_prefix_store = store
            self._prefix_store = store
        return store

    # ---------------- ESPO (local addition 2026-11-04; see rlinf/models/embodiment/openpi/espo.py) ----------------
    def _espo_policy_module(self):
        from rlinf.models.embodiment.openpi.tasks.rl import Pi0RL

        targets = [m for m in self.model.modules() if isinstance(m, Pi0RL)]
        assert len(targets) == 1, f"ESPO expects one Pi0RL module, found {len(targets)}"
        assert targets[0].rl_cfg.train_expert_only, "ESPO v1 assumes a frozen VLM prefix"
        return targets[0]

    def _espo_slots(self, rollout_size, batch_size_per_rank):
        """Same (global, micro) split the update loop uses, so prefix-cache slots line up."""
        out = []
        for g, gb in enumerate(split_dict_to_chunk(self.rollout_batch, rollout_size // batch_size_per_rank)):
            n = gb["prev_logprobs"].shape[0]
            for m, mb in enumerate(split_dict_to_chunk(gb, n // self.cfg.actor.micro_batch_size)):
                out.append(((g, m), mb))
        return out

    def _espo_prepare(self, prefix_store, rollout_size, batch_size_per_rank) -> dict:
        from rlinf.models.embodiment.openpi import espo

        ecfg = self.cfg.algorithm.get("espo", {})
        M = int(ecfg.get("model_samples", 2))
        eta = float(ecfg.get("eta", 3.0))
        seed = int(self.cfg.actor.seed) * 100003 + int(getattr(self, "version", 0)) * 101 + self._rank
        model = self._espo_policy_module()
        slots = self._espo_slots(rollout_size, batch_size_per_rank)
        zs, us = [], []
        with torch.no_grad(), self.amp_context:
            for slot, mb in slots:
                if prefix_store is not None:
                    prefix_store.slot = slot
                fi = put_tensor_device(mb["forward_inputs"], self.device)
                # through the FSDP root forward (valid parameter views), prefix stored in this slot
                zs.append(self.model(forward_inputs=fi, espo_z_only=True)["espo_z"])
                us.append(espo.executed_from_chains(model, fi["chains"]))
        Z, U = torch.cat(zs), torch.cat(us)
        returns = self.rollout_batch["returns"].reshape(-1).to(self.device, torch.float32)
        adv = self.rollout_batch["advantages"].reshape(-1).to(self.device, torch.float32)
        # v4 option (2026-11-04): 'positive' keeps only A > 0 (negative-advantage states get target = Phi0, i.e. they
        # anchor the sampler where it is); 'clip' bounds the repulsive push from negative advantages. Default 'signed'.
        adv_mode = str(ecfg.get("adv_mode", "signed"))
        if adv_mode == "positive":
            adv = adv.clamp(min=0.0)
        elif adv_mode == "clip":
            adv = adv.clamp(min=-float(ecfg.get("adv_neg_clip", 0.5)))
        elif adv_mode != "signed":
            raise ValueError(f"unknown algorithm.espo.adv_mode {adv_mode!r}")
        valid = self.rollout_batch["loss_mask"].reshape(-1).to(self.device).bool()
        q, qstats = espo.fit_return_predictor(
            Z[valid], U[valid], returns[valid], steps=int(ecfg.get("q_steps", 400)),
            lr=float(ecfg.get("q_lr", 1e-3)), seed=seed,
        )
        if str(ecfg.get("feature", "qfeat")) == "action":
            q = espo.ActionMomentFeatures()  # Q is still fitted for diagnostics only
        self._espo_q = q
        targets, noises, u0s, i0 = [], [], [], 0
        H, D = model.action_horizon, model.model_action_dim
        gen = torch.Generator(device=Z.device).manual_seed(seed)
        shift = []
        with torch.no_grad(), self.amp_context:
            for slot, mb in slots:
                if prefix_store is not None:
                    prefix_store.slot = slot
                fi = put_tensor_device(mb["forward_inputs"], self.device)
                B = fi["chains"].shape[0]
                noise = torch.randn(B, M, H, D, device=Z.device, dtype=torch.float32, generator=gen)
                z = Z[i0:i0 + B]
                # Phi0 at the current parameters with the same noise the update epochs will reuse (FSDP forward)
                out0 = self.model(forward_inputs=fi, espo_noise=noise, espo_feature_fn=q.features)
                phi0 = out0["espo_phi"].float()
                u0s.append(out0["espo_u"].float().cpu())
                b = adv[i0:i0 + B, None] * (q.features(z, U[i0:i0 + B]) - phi0)
                b = b * valid[i0:i0 + B, None].float()
                targets.append((phi0 + eta * b).cpu())
                noises.append(noise.cpu())
                shift.append((eta * b).norm(dim=-1)[valid[i0:i0 + B]].cpu())
                i0 += B
        if prefix_store is not None:
            prefix_store.slot = None
        self.rollout_batch["espo_target"] = torch.cat(targets)
        self.rollout_batch["espo_noise"] = torch.cat(noises)
        self.rollout_batch["espo_u0"] = torch.cat(u0s)  # pre-update executed actions of the model samples
        U0 = self.rollout_batch["espo_u0"]
        self._espo_u_scale = float(U0.var(dim=1).mean().clamp(min=1e-6)) if U0.shape[1] > 1 else 1.0
        sh = torch.cat(shift)
        # Loss scale: divide by the mean squared target shift so the ESPO loss starts at ~1 (comparable to the PPO
        # actor loss) and does not crowd the critic out of RLinf's global grad-norm clip.
        ss = torch.tensor([float((sh ** 2).sum()), float(sh.numel())], device=Z.device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(ss, op=torch.distributed.ReduceOp.SUM)
        self._espo_loss_scale = 1.0 / max(float(ss[0] / ss[1].clamp(min=1.0)), 1e-6)
        self._espo_target_shift = float(sh.mean()) if sh.numel() else None
        self._espo_mb_per_epoch = len(slots)
        return {**{f"espo/{k}": v for k, v in qstats.items()},
                "espo/target_shift_mean": float(sh.mean()) if sh.numel() else 0.0,
                "espo/valid_frac": float(valid.float().mean())}

    def _train_micro_batch_espo(self, micro_batch, metrics, *, is_last: bool) -> None:
        from rlinf.algorithms.losses import compute_ppo_critic_loss

        micro_batch = put_tensor_device(micro_batch, self.device)
        backward_ctx = self.before_micro_batch(self.model, is_last_micro_batch=is_last)
        ecfg = self.cfg.algorithm.get("espo", {})
        with self.amp_context:
            out = self.model(
                forward_inputs=micro_batch["forward_inputs"],
                espo_noise=micro_batch["espo_noise"],
                espo_feature_fn=self._espo_q.features,
            )
        phi = out["espo_phi"].float()
        target = micro_batch["espo_target"].to(phi.device, phi.dtype)
        mask = micro_batch["loss_mask"].reshape(-1).float()
        lms = micro_batch.get("loss_mask_sum", None)
        max_steps = self.cfg.env.train.max_episode_steps
        use_norm = self.cfg.algorithm.get("length_normalization", True) and lms is not None
        if use_norm:  # same per-trajectory weighting as RLinf's masked_mean_ratio
            w = mask * max_steps / lms.reshape(-1).float().clamp(min=1.0)
            espo_loss = (w * ((phi - target) ** 2).sum(-1)).mean()
        else:
            espo_loss = (mask * ((phi - target) ** 2).sum(-1)).sum() / mask.sum().clamp(min=1.0)
        critic_loss, critic_metrics = compute_ppo_critic_loss(
            values=out["values"].reshape(-1),
            returns=micro_batch["returns"].reshape(-1).float(),
            prev_values=micro_batch["prev_values"].reshape(-1).float(),
            value_clip=self.cfg.algorithm.get("value_clip", None),
            huber_delta=self.cfg.algorithm.get("huber_delta", None),
            loss_mask=micro_batch["loss_mask"].reshape(-1).bool(),
            max_episode_steps=max_steps if use_norm else None,
            loss_mask_sum=lms.reshape(-1).float() if lms is not None else None,
        )
        espo_loss = espo_loss * getattr(self, "_espo_loss_scale", 1.0)
        prox_coef = float(ecfg.get("prox_coef", 0.0))
        prox = torch.zeros((), device=phi.device)
        if prox_coef > 0:
            # action-space proximal term: keep the sampler's executed actions (same noise) near their pre-update
            # values, in units of the policy's own action spread (a trust region in execution space)
            du = (out["espo_u"].float() - micro_batch["espo_u0"].to(phi.device).float()) ** 2
            prox = (mask[:, None] * du.mean(-1)).sum() / (mask.sum().clamp(min=1.0) * du.shape[1])
            prox = prox / getattr(self, "_espo_u_scale", 1.0)
            espo_loss = espo_loss + prox_coef * prox
        if self.optimizer_steps < self.critic_warmup_steps or getattr(self, "_espo_stopped", False):
            espo_loss = espo_loss * 0.0  # critic warmup, or trust-region stop for this iteration
        loss = float(ecfg.get("coef", 1.0)) * espo_loss + critic_loss
        loss = loss / self.gradient_accumulation
        with backward_ctx:
            self.grad_scaler.scale(loss).backward()
        metrics_data = {k: (v.item() if torch.is_tensor(v) and v.numel() == 1 else v) for k, v in critic_metrics.items()}
        metrics_data.update({
            "espo/loss": espo_loss.detach().item(),
            "espo/phi_target_dist": ((phi - target) ** 2).sum(-1).detach().sqrt().mean().item(),
            "espo/action_shift": float(prox.detach()),
            "actor/total_loss": loss.detach().item(),
        })
        append_to_dict(metrics, metrics_data)

    def train_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        metrics: dict[str, list[float]],
        *,
        is_last: bool,
    ) -> None:
        if self.cfg.algorithm.loss_type == "espo":
            return self._train_micro_batch_espo(micro_batch, metrics, is_last=is_last)
        micro_batch = put_tensor_device(micro_batch, self.device)
        backward_ctx = self.before_micro_batch(self.model, is_last_micro_batch=is_last)
        advantages = micro_batch["advantages"]
        prev_logprobs = micro_batch["prev_logprobs"]
        returns = micro_batch.get("returns", None)
        prev_values = micro_batch.get("prev_values", None)
        loss_mask = micro_batch.get("loss_mask", None)
        loss_mask_sum = micro_batch.get("loss_mask_sum", None)
        forward_inputs = micro_batch.get("forward_inputs", None)

        kwargs = {}
        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENVLA,
            SupportedModel.OPENVLA_OFT,
            SupportedModel.PI0_FAST,
        ]:
            temperature = self.cfg.rollout.sampling_params.temperature_train
            if (
                SupportedModel(self.cfg.actor.model.model_type)
                == SupportedModel.PI0_FAST
                and not self.cfg.rollout.sampling_params.do_sample
            ):
                temperature = 1.0
            kwargs["temperature"] = temperature
            kwargs["top_k"] = self.cfg.rollout.sampling_params.top_k
        elif SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.GR00T,
            SupportedModel.GR00T_N1D6,
            SupportedModel.GR00T_N1D7,
            SupportedModel.ABOT_M0,
        ]:
            kwargs["prev_logprobs"] = prev_logprobs

        compute_values = self.cfg.algorithm.adv_type == "gae"
        compute_entropy = (
            self.cfg.algorithm.entropy_bonus > 0
            or self.cfg.algorithm.get("log_entropy_metrics", False)
        )
        with self.amp_context:
            output_dict = self.model(
                forward_inputs=forward_inputs,
                compute_logprobs=True,
                compute_entropy=compute_entropy,
                compute_values=compute_values,
                use_cache=False,
                **kwargs,
            )

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.GR00T,
            SupportedModel.GR00T_N1D6,
            SupportedModel.GR00T_N1D7,
            SupportedModel.ABOT_M0,
        ]:
            prev_logprobs = output_dict["prev_logprobs"]

        loss_kwargs = {
            "loss_type": self.cfg.algorithm.loss_type,
            "logprob_type": self.cfg.algorithm.logprob_type,
            "reward_type": self.cfg.algorithm.reward_type,
            "single_action_dim": self.cfg.actor.model.get("action_dim", 7),
            "logprobs": output_dict["logprobs"],
            "values": output_dict.get("values", None),
            "old_logprobs": prev_logprobs,
            "advantages": advantages,
            "returns": returns,
            "prev_values": prev_values,
            "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
            "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
            "value_clip": self.cfg.algorithm.get("value_clip", None),
            "huber_delta": self.cfg.algorithm.get("huber_delta", None),
            "loss_mask": loss_mask,
            "loss_mask_sum": loss_mask_sum,
            "max_episode_steps": self.cfg.env.train.max_episode_steps,
            "task_type": self.cfg.runner.task_type,
            "critic_warmup": (self.optimizer_steps < self.critic_warmup_steps
                              or getattr(self, "_ppo_control_critic_only", False)),
            "logprob_mask": output_dict.get("logprob_mask", None),
            "loss_agg_func": self.loss_agg_func,
            "log_logprob_diagnostics": self.cfg.algorithm.get(
                "log_logprob_diagnostics", False
            ),
        }

        # Ablation switch (2026-11-03). With auto_reset=False and ignore_terminations=False, RLinf passes
        # each trajectory's valid length (loss_mask_sum) and max_episode_steps, so the PPO actor and critic
        # losses use masked_mean_ratio: every step is divided by its trajectory's valid length, i.e. each
        # trajectory gets equal total weight ("trajectory-balanced"). length_normalization=False restores
        # plain per-step ("token-mean") aggregation over valid steps for both losses.
        if not self.cfg.algorithm.get("length_normalization", True):
            loss_kwargs["max_episode_steps"] = None

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.GR00T_N1D6,
            SupportedModel.GR00T_N1D7,
            SupportedModel.PI0_FAST,
        ]:
            loss_kwargs["clip_ratio_c"] = self.cfg.algorithm.get("clip_ratio_c", 3.0)
            if self.cfg.algorithm.get("clip_log_ratio_min") is not None:
                loss_kwargs["clip_log_ratio_min"] = (
                    self.cfg.algorithm.clip_log_ratio_min
                )
            if self.cfg.algorithm.get("clip_log_ratio_max") is not None:
                loss_kwargs["clip_log_ratio_max"] = (
                    self.cfg.algorithm.clip_log_ratio_max
                )

        loss, metrics_data = policy_loss(**loss_kwargs)
        entropy_loss = torch.tensor(0.0, device=Worker.torch_platform.current_device())
        if compute_entropy and not loss_kwargs["critic_warmup"]:
            entropy = output_dict.get("entropy", None)
            if entropy is not None and self.cfg.algorithm.entropy_bonus <= 0:
                entropy = entropy.detach()
            if entropy is not None:
                entropy_mask = loss_mask
                logprob_mask = output_dict.get("logprob_mask", None)
                if logprob_mask is not None:
                    entropy_mask = logprob_mask.to(dtype=torch.bool)
                    if loss_mask is not None:
                        env_mask = loss_mask.to(
                            device=entropy_mask.device, dtype=torch.bool
                        )
                        while env_mask.ndim > 1 and env_mask.shape[-1] == 1:
                            env_mask = env_mask.squeeze(-1)
                        while env_mask.ndim < entropy_mask.ndim:
                            env_mask = env_mask.unsqueeze(-1)
                        entropy_mask = entropy_mask & env_mask
                entropy_loss = compute_entropy_loss(
                    entropy,
                    entropy_type=self.cfg.algorithm.entropy_type,
                    loss_mask=entropy_mask,
                    action_dim=self.cfg.actor.model.get("action_dim", 7),
                    batch_size=output_dict["logprobs"].shape[0],
                )
                if self.cfg.algorithm.entropy_bonus > 0:
                    loss -= self.cfg.algorithm.entropy_bonus * entropy_loss
        metrics_data["actor/entropy_loss"] = entropy_loss.detach().item()
        if compute_entropy:
            metrics_data["actor/entropy"] = entropy_loss.detach().item()
            metrics_data["actor/entropy_bonus_term"] = (
                self.cfg.algorithm.entropy_bonus * entropy_loss.detach().item()
            )

        if self.enable_sft_co_train:
            loss = self._train_sft_epoch(metrics_data, loss)

        loss /= self.gradient_accumulation
        with backward_ctx:
            self.grad_scaler.scale(loss).backward()

        metrics_data["actor/total_loss"] = loss.detach().item()
        append_to_dict(metrics, metrics_data)

    def set_global_step(self, global_step: int) -> None:
        """
        Set the global step for the model, if needed.
        """
        self.version = global_step
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)

    def finish_global_batch(self, metrics: dict[str, list[float]]) -> None:
        self.torch_platform.empty_cache()
        grad_norm, lr_list = self.optimizer_step()
        self.optimizer.zero_grad()
        metric_data = {
            "actor/grad_norm": grad_norm,
            "actor/lr": lr_list[0],
        }
        if len(lr_list) > 1:
            metric_data["critic/lr"] = lr_list[1]
        append_to_dict(metrics, metric_data)

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        super().save_checkpoint(save_path, step)
        if hasattr(self, "_update_controller") and self._rank == 0:
            from pathlib import Path
            from rlinf.b1k_tools.ppo_update_controller import atomic_json
            atomic_json(Path(save_path) / "ppo_controller.json", self._update_controller.state_dict())

    def load_checkpoint(self, load_path: str) -> None:
        super().load_checkpoint(load_path)
        if self.cfg.algorithm.get("ppo_control", {}).get("enable", False):
            import json
            from pathlib import Path
            path = Path(load_path) / "ppo_controller.json"
            if not path.exists():
                raise FileNotFoundError(f"PPO-controller checkpoint is incomplete: {path}")
            self._ppo_control_resume_state = json.loads(path.read_text())
