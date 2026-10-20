"""One bounded on-policy update using upstream RLinf sampling, GAE and PPO losses.

The simulator runs the frozen original evaluator. This validates the single-process
learning loop, not Ray/FSDP orchestration or an improvement in task success rate.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from common import Simulator, env_observation, label_text, torch_model, write


def hash_parameters(named):
    digest = hashlib.sha256()
    for name, parameter in named:
        digest.update(name.encode())
        # Bound temporary host memory even for the tied embedding table.
        flat = parameter.detach().view(torch.uint8).reshape(-1)
        for begin in range(0, flat.numel(), 8 * 1024**2):
            digest.update(flat[begin : begin + 8 * 1024**2].cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    args = p.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    root = Path(plan["output"])
    if not json.loads((root / "gate.json").read_text())["passed"]:
        raise RuntimeError("Closed-loop gate did not pass; training forbidden")
    # Imported functions are the pinned upstream implementation, not copied PPO math.
    from rlinf.algorithms.advantages import compute_gae_advantages_and_returns
    from rlinf.algorithms.losses import compute_ppo_actor_critic_loss
    from rlinf.algorithms.utils import preprocess_loss_inputs

    random.seed(plan["seed"])
    np.random.seed(plan["seed"])
    model = torch_model(plan)
    model.eval()
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    frozen = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
    before_frozen = hash_parameters(frozen)
    before_trainable = hash_parameters(trainable)
    original = {n: p.detach().cpu().clone() for n, p in trainable}
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for n, p in trainable if not n.startswith("value_head.")],
                "lr": 1e-6,
            },
            {
                "params": [p for n, p in trainable if n.startswith("value_head.")],
                "lr": 1e-4,
            },
        ],
        weight_decay=0.01,
        eps=1e-8,
        betas=(0.9, 0.999),
    )
    sim = Simulator(
        plan,
        root / "ppo" / "rollout",
        plan["train_instance"],
        "train",
        plan["ppo_steps"],
    )
    transitions = []
    try:
        while sim.current["status"]["steps"] < plan["ppo_steps"]:
            with torch.no_grad():
                actions, result = model.predict_action_batch(
                    env_observation(sim.current["observation"]), mode="train"
                )
            prev = {
                k: {n: t.detach().cpu() for n, t in v.items()}
                if isinstance(v, dict)
                else v.detach().cpu()
                for k, v in result.items()
                if k != "model_actions"
            }
            current = sim.step(actions[0].cpu().numpy(), label_text(model, result))
            prev.update(
                reward=float(sum(current["rewards"])) * 10.0,
                terminal=current["terminal"],
                done=current["done"],
            )
            transitions.append(prev)
            write(
                root / "ppo" / "progress.json",
                {"stage": "rollout", "chunks": len(transitions), **current["status"]},
            )
            if current["done"]:
                break
        # A time-limit boundary is bootstrapped, not treated as an absorbing failure.
        if transitions[-1]["terminal"]:
            bootstrap = torch.zeros(1)
        else:
            with torch.no_grad():
                _, end_result = model.predict_action_batch(
                    env_observation(sim.current["observation"]), mode="train"
                )
            bootstrap = end_result["prev_values"].reshape(1).cpu()
        final_status = sim.current["status"]
    finally:
        sim.close()
    if len(transitions) < 2:
        raise RuntimeError("Too few transitions for a meaningful PPO plumbing check")
    values = torch.stack(
        [x["prev_values"].reshape(1) for x in transitions] + [bootstrap]
    )
    rewards = torch.tensor([[x["reward"]] for x in transitions], dtype=torch.float32)
    dones = torch.tensor(
        [[False]] + [[x["terminal"]] for x in transitions], dtype=torch.bool
    )
    advantages, returns = compute_gae_advantages_and_returns(
        rewards=rewards,
        values=values,
        dones=dones,
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantages=True,
    )
    torch.save(
        {"transitions": transitions, "advantages": advantages, "returns": returns},
        root / "ppo" / "rollout.pt",
    )
    metrics = []
    # One epoch over this on-policy batch; microbatch=1, accumulate four chunks per update.
    for start in range(0, len(transitions), 4):
        chunk_ids = list(range(start, min(start + 4, len(transitions))))
        optimizer.zero_grad(set_to_none=True)
        for index in chunk_ids:
            old = transitions[index]
            replay = {k: v.cuda() for k, v in old["forward_inputs"].items()}
            output = model.default_forward(replay)
            fields = preprocess_loss_inputs(
                logprobs=output["logprobs"].float(),
                old_logprobs=old["prev_logprobs"].cuda().float(),
                advantages=advantages[index].cuda(),
                values=output["values"].float(),
                prev_values=old["prev_values"].cuda().float(),
                returns=returns[index].cuda(),
                logprob_type="chunk_level",
                reward_type="chunk_level",
                single_action_dim=23,
            )
            loss, data = compute_ppo_actor_critic_loss(
                **fields,
                clip_ratio_low=0.2,
                clip_ratio_high=0.2,
                value_clip=0.2,
                huber_delta=10.0,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite PPO loss")
            (loss / len(chunk_ids)).backward()
            metrics.append(
                {
                    "chunk": index,
                    "loss": float(loss.detach()),
                    **{
                        k: float(v.detach().float().mean())
                        for k, v in data.items()
                        if torch.is_tensor(v)
                    },
                }
            )
        assert all(p.grad is None for _, p in frozen)
        grad_norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError("Non-finite PPO gradients")
        optimizer.step()
        write(
            root / "ppo" / "progress.json",
            {
                "stage": "update",
                "updated_chunks": start + len(chunk_ids),
                "gradient_norm": float(grad_norm),
            },
        )
    optimizer.zero_grad(set_to_none=True)
    after_frozen = hash_parameters(frozen)
    after_trainable = hash_parameters(trainable)
    assert before_frozen == after_frozen and before_trainable != after_trainable
    delta = {
        n: float((p.detach().cpu() - original[n]).abs().max()) for n, p in trainable
    }
    assert any(v > 0 for n, v in delta.items() if not n.startswith("value_head."))
    assert any(v > 0 for n, v in delta.items() if n.startswith("value_head."))
    checkpoint = root / "ppo" / "checkpoint"
    checkpoint.mkdir()
    torch.save({n: p.detach().cpu() for n, p in trainable}, checkpoint / "trainable.pt")
    torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    # Verify round-trip of the exact updated tensors; base weights are explicitly referenced.
    saved = torch.load(
        checkpoint / "trainable.pt", map_location="cpu", weights_only=True
    )
    assert all(torch.equal(saved[n], p.detach().cpu()) for n, p in trainable)
    write(
        checkpoint / "base_reference.json",
        {
            "base_checkpoint": plan["converted"],
            "overlay": "trainable.pt",
            "optimizer": "optimizer.pt",
            "upstream_commit": plan["upstream_commit"],
            "loading": "Build the base JointPi0RL model, then load trainable.pt with strict=False; missing keys must equal the frozen parameter keys.",
        },
    )
    report = {
        "passed": True,
        "iterations": 1,
        "epochs_over_batch": 1,
        "chunks": len(transitions),
        "optimizer_steps": (len(transitions) + 3) // 4,
        "controls": final_status["steps"],
        "reward_sum": float(rewards.sum()),
        "final_status": final_status,
        "frozen_sha_before": before_frozen,
        "frozen_sha_after": after_frozen,
        "trainable_sha_before": before_trainable,
        "trainable_sha_after": after_trainable,
        "max_parameter_changes": delta,
        "metrics": metrics,
        "checkpoint_round_trip": True,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "scope": "Single-process RLinf PPO smoke; no Ray/FSDP validation and no success-rate improvement claim.",
    }
    write(root / "ppo" / "result.json", report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "passed",
                    "chunks",
                    "optimizer_steps",
                    "controls",
                    "reward_sum",
                ]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
