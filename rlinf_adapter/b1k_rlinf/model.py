"""Joint-subtask policy with RLinf's unchanged Flow-SDE / PPO implementation."""

import numpy as np
import torch
from rlinf.models.embodiment.openpi.modules.model import (
    IMAGE_KEYS,
    Observation,
    preprocess_observation,
)
from rlinf.models.embodiment.openpi.tasks.rl import Pi0RL

from .action_interface import ExecutionActionInterface


class JointPi0RL(Pi0RL):
    """Freeze the language policy and improve actions conditioned on its tokens."""

    def install_joint_io(self, checkpoint_io, tokenizer, action_interface=None):
        self.checkpoint_io = checkpoint_io
        self.joint_tokenizer = tokenizer
        self.execution_interface = action_interface or ExecutionActionInterface()

    def embed_prefix(self, observation):
        embedded, valid, ar = super().embed_prefix(observation)
        lang_ar = observation.token_ar_mask
        if lang_ar is not None:
            if lang_ar.shape != observation.tokenized_prompt_mask.shape:
                raise ValueError("token_ar_mask shape differs from tokenized_prompt_mask")
            ar = ar.expand(valid.shape[0], -1).clone()
            ar[:, -lang_ar.shape[1] :] = lang_ar.bool()
        return embedded, valid, ar

    @torch.no_grad()
    def generate_subtask(self, observation):
        """Greedy batched decoding with per-environment EOS and a local KV cache."""
        observation = preprocess_observation(observation, train=False)
        mask = observation.tokenized_prompt_mask.bool()
        lengths = mask.sum(-1)
        limit = self.joint_tokenizer.max_subtask_len
        if not torch.all(lengths > 0) or not torch.all(lengths + limit <= mask.shape[1]):
            raise ValueError("Invalid or overflowing generation prefix")
        if not torch.equal(mask, torch.arange(mask.shape[1], device=self.device)[None] < lengths[:, None]):
            raise ValueError("Generation prefix must be contiguous before padding")
        if observation.token_ar_mask is not None and observation.token_ar_mask.any():
            raise ValueError("Generation prefix cannot contain autoregressive label tokens")
        hidden, valid, cache = self.build_prefix_cache(observation)
        batch = mask.shape[0]
        rows = torch.arange(batch, device=self.device)
        last = hidden.shape[1] - mask.shape[1] + lengths - 1
        logits = self.llm.embedder.decode(hidden[rows, last]).float()
        eos = self.joint_tokenizer.eos_id
        if not 0 <= eos < logits.shape[-1] or eos in (0, 2):
            raise ValueError("Expected EOS distinct from PAD and BOS")
        generated = torch.zeros(batch, limit, dtype=torch.long, device=self.device)
        out_lengths = torch.zeros(batch, dtype=torch.long, device=self.device)
        done = torch.zeros(batch, dtype=torch.bool, device=self.device)
        for index in range(limit):
            logits = logits.clone()
            logits[:, 0] = -torch.inf
            logits[:, 2] = -torch.inf
            if index == 0:
                logits[:, eos] = -torch.inf
            active = ~done
            token = torch.where(active, logits.argmax(-1), 0)
            generated[:, index] = token
            out_lengths += active.long()
            done |= active & (token == eos)
            if done.all() or index + 1 == limit:
                break
            positions = valid.sum(-1)[:, None]
            valid = torch.cat([valid, active[:, None]], -1)
            hidden_list, cache = self.llm(
                [self.llm.embed(token[:, None]), None],
                positions=positions,
                mask=(valid & active[:, None])[:, None, :],
                kv_cache=cache,
            )
            logits = self.llm.embedder.decode(hidden_list[0][:, 0]).float()
        if not done.all():
            rows = (~done).nonzero().flatten().tolist()
            raise RuntimeError(f"Subtask generation did not finish with EOS for rows {rows}")
        return generated, out_lengths

    def _token_observation(self, images, normalized, arrays):
        data = {
            "image": {k: torch.as_tensor(v, device=self.device) for k, v in images.items()},
            "image_mask": {
                k: torch.ones(len(normalized), dtype=torch.bool, device=self.device) for k in IMAGE_KEYS
            },
            "state": torch.as_tensor(
                np.pad(normalized, ((0, 0), (0, 9))), device=self.device, dtype=torch.float32
            ),
        }
        for key in ("tokenized_prompt", "tokenized_prompt_mask", "token_ar_mask"):
            data[key] = torch.as_tensor(np.stack([a[key] for a in arrays]), device=self.device)
        return Observation.from_dict(data)

    @torch.no_grad()
    def predict_action_batch(
        self, env_obs, mode="eval", compute_values=True, *, noise=None, rng=None, **kwargs
    ):
        if mode not in ("train", "eval"):
            raise ValueError(f"Expected train or eval mode, got {mode}")
        images, normalized, prompts = self.checkpoint_io.prepare(env_obs)
        prefixes = [self.joint_tokenizer.prefix(p, s) for p, s in zip(prompts, normalized)]
        prefix_obs = self._token_observation(images, normalized, prefixes)
        generated, lengths = self.generate_subtask(prefix_obs)
        labels = [generated[i, : int(lengths[i])].cpu().numpy() for i in range(len(prompts))]
        arrays = [
            self.joint_tokenizer.actions(p, s, label) for p, s, label in zip(prompts, normalized, labels)
        ]
        observation = self._token_observation(images, normalized, arrays)
        if mode == "eval":
            actions, result = self._predict_eval(observation, noise=noise, rng=rng)
        else:
            actions, result = self._predict_train(
                observation, noise=noise, rng=rng, compute_values=compute_values
            )
        replay = result["forward_inputs"]
        replay.update(
            {
                "tokenized_prompt": observation.tokenized_prompt.contiguous(),
                "tokenized_prompt_mask": observation.tokenized_prompt_mask.contiguous(),
                "token_ar_mask": observation.token_ar_mask.contiguous(),
                "obs_state": observation.state.contiguous(),
                "subtask_tokens": generated.contiguous(),
                "subtask_lengths": lengths.contiguous(),
            }
        )
        for name in IMAGE_KEYS:
            # Preserve the exact resized uint8 pixels; normalization is lossless on replay.
            replay[f"obs_image__{name}"] = torch.as_tensor(images[name], device=self.device).contiguous()
            replay[f"obs_image_mask__{name}"] = observation.image_masks[name].contiguous()
        # Keep the policy's raw physical action and the denoising-chain density
        # unchanged. Both train and eval execute the same deterministic mapping.
        result["policy_actions"] = actions
        execution_actions = self.execution_interface(actions)
        result["execution_actions"] = execution_actions
        return execution_actions, result

    def _observation_from_forward_inputs(self, forward_inputs):
        required = ("tokenized_prompt", "tokenized_prompt_mask", "token_ar_mask", "obs_state")
        missing = [k for k in required if k not in forward_inputs]
        if missing:
            raise ValueError(f"PPO replay is missing joint context: {missing}")
        data = {
            "image": {name: forward_inputs[f"obs_image__{name}"] for name in IMAGE_KEYS},
            "image_mask": {name: forward_inputs[f"obs_image_mask__{name}"] for name in IMAGE_KEYS},
            "state": forward_inputs["obs_state"],
            **{k: forward_inputs[k] for k in required if k != "obs_state"},
        }
        # No language generation or tokenization in actor recompute.
        return preprocess_observation(Observation.from_dict(data), train=False)

    def decode_actions(self, model_actions, state):
        return self.checkpoint_io.decode(model_actions, state, self.action_chunk)
