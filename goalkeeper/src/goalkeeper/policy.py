"""Inference-only composition; imports no simulator or oracle-label code."""
from __future__ import annotations
from collections import deque
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from .belief import BeliefMemory, PreservationSelector
from .calibration import ReliabilityCalibrator
from .observations import SensorSchema, RobotObservation, validate_observation
from .artifacts import load_checkpoint


class LearnedPredictor:
    def __init__(self, program, checkpoint, calibration, features, action_spec, device="cpu"):
        self.program, self.features, self.device = program, features, device
        self.model, self.artifact = load_checkpoint(checkpoint, device)
        calibration = json.loads(Path(calibration).read_text())
        if self.artifact["synthetic"]:
            raise ValueError("Synthetic smoke-test weights cannot be deployed as a competition policy")
        if calibration["checkpoint_sha256"] != hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest():
            raise ValueError("Calibration belongs to a different checkpoint")
        if self.artifact["feature_id"] != features.feature_id or self.artifact["action_spec"] != action_spec:
            raise ValueError("Feature encoder or action specification mismatch")
        self.calibrators = {k: ReliabilityCalibrator(**calibration[k]) for k in ("current", "future", "visibility")}
        queries = features.encode_queries([program.query(a) for a in program.atoms])
        self.queries = torch.as_tensor(queries, device=device)[None]
        self.reset()

    def reset(self): self.hidden = None

    @torch.inference_mode()
    def predict(self, observation, candidates):
        visual = torch.as_tensor(self.features.encode_images(observation.rgb), device=self.device)[None, None]
        state = torch.as_tensor(observation.proprio, device=self.device)[None, None]
        actions = torch.as_tensor(candidates, device=self.device)[None]
        stats = self.artifact["normalization"]
        state = (state - state.new_tensor(stats["proprio"]["mean"])) / state.new_tensor(stats["proprio"]["std"])
        actions = (actions - actions.new_tensor(stats["actions"]["mean"])) / actions.new_tensor(stats["actions"]["std"])
        out = self.model(visual, state, self.queries, actions, self.hidden)
        self.hidden = out["hidden"].detach()
        now = out["current_logits"].sigmoid()[0, -1].cpu().tolist()
        vis = out["visibility_logits"].sigmoid()[0, -1].cpu().tolist()
        future = out["future_logits"].sigmoid()[0].cpu().tolist()
        atoms = self.program.atoms
        current = {a: self.calibrators["current"].estimate(p) for a, p in zip(atoms, now)}
        visibility = {a: self.calibrators["visibility"].estimate(p).lower for a, p in zip(atoms, vis)}
        futures = [{a: self.calibrators["future"].estimate(p) for a, p in zip(atoms, values)} for values in future]
        return current, visibility, futures


def _resize_with_pad(image, size=224):
    # Use the exact helper supplied with the selected OpenPI checkout. Different
    # interpolation/padding changes a checkpoint's input distribution.
    from openpi_client.image_tools import resize_with_pad
    return resize_with_pad(image[None], size, size)[0]


class OpenPIChunkSampler:
    """Wrap raw openpi Policy.infer, NOT Comet's queue-based B1KPolicyWrapper.

    Calling an already-buffered act() K times would incorrectly consume successive
    actions instead of independently sampling K action chunks.
    """
    def __init__(self, policy, instruction, image_size=224, image_transform=None):
        self.policy, self.instruction, self.image_size = policy, instruction, image_size
        self.image_transform = image_transform or _resize_with_pad

    def reset(self):
        if hasattr(self.policy, "reset"): self.policy.reset()

    def _batch(self, observation: RobotObservation):
        images = [self.image_transform(x, self.image_size) for x in observation.rgb]
        # Match BEHAVIOR-VLAs/openpi's B1KInputs and B1KPolicyWrapper.
        return {"observation/image_0": images[0],
                "observation/image_1": images[1],
                "observation/image_2": images[2],
                "observation/state": observation.proprio, "prompt": self.instruction}

    def sample(self, observation: RobotObservation, count: int):
        batch = self._batch(observation)
        chunks = []
        for _ in range(count):
            response = self.policy.infer(batch)
            chunks.append(np.asarray(response["actions"], dtype=np.float32))
        return np.stack(chunks)


class JointSubtaskChunkSampler(OpenPIChunkSampler):
    """Generate one subtask, then sample all candidates conditioned on it."""

    def reset(self):
        super().reset()
        self.last_subtask = None
        self.last_subtask_source = None

    def sample(self, observation: RobotObservation, count: int):
        # A separate infer() call per candidate could refresh the subtask in
        # the middle of a GoalKeeper decision.
        result = self.policy.infer_candidates(self._batch(observation), count)
        self.last_subtask = result["subtask"]
        self.last_subtask_source = result["subtask_source"]
        return np.asarray(result["actions"], dtype=np.float32)


class GoalkeeperPolicy:
    def __init__(self, program, sampler, *, task_id: int, proprio_dim: int, action_dim: int,
                 horizon=16, candidates=4, predictor=None, mode="baseline", schema=None, selector=None):
        if mode not in ("baseline", "shadow", "rerank"): raise ValueError("Unknown mode")
        if mode != "baseline" and predictor is None: raise ValueError("A trained predictor is required")
        if min(horizon, candidates, proprio_dim, action_dim) < 1: raise ValueError("Invalid policy dimensions")
        if predictor is not None and predictor.model.config.horizon != horizon:
            raise ValueError("Must execute exactly the action prefix used to train the predictor")
        self.program, self.sampler, self.predictor, self.mode = program, sampler, predictor, mode
        self.task_id, self.proprio_dim, self.action_dim = task_id, proprio_dim, action_dim
        self.horizon, self.candidates = horizon, candidates
        self.schema = schema or SensorSchema()
        self.memory, self.selector = BeliefMemory(program), selector or PreservationSelector()
        self.reset()

    def reset(self):
        self.queue = deque(); self.last_decision = None
        self.last_subtasks = {}; self.last_subtask_sources = {}
        self.memory.reset(); self.sampler.reset()
        if self.predictor: self.predictor.reset()

    def act(self, payload: dict):
        observation = validate_observation(payload, self.schema, self.task_id, self.proprio_dim)
        if not self.queue:
            count = 1 if self.mode == "baseline" else self.candidates
            candidates = self.sampler.sample(observation, count)
            subtask = getattr(self.sampler, "last_subtask", None)
            source = getattr(self.sampler, "last_subtask_source", None)
            self.last_subtasks = {0: subtask} if subtask is not None else {}
            self.last_subtask_sources = {0: source} if source is not None else {}
            if candidates.ndim != 3 or candidates.shape[0] != count or candidates.shape[1] < self.horizon or candidates.shape[2] != self.action_dim:
                raise ValueError("Base policy action dimensions/horizon mismatch")
            if not np.isfinite(candidates).all(): raise ValueError("Base policy produced nonfinite actions")
            candidates = candidates[:, :self.horizon].astype(np.float32, copy=True)
            chosen = 0
            if self.predictor:
                estimates, visibility, futures = self.predictor.predict(observation, candidates)
                current = self.memory.update(estimates, visibility)
                self.last_decision = self.selector.choose(self.program, current, futures)
                if self.mode == "rerank": chosen = self.last_decision.index
            self.queue.extend(candidates[chosen])
        return self.queue.popleft().copy()
