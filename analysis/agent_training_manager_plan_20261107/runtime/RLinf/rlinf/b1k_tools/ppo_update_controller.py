# Copyright 2026 The RLinf Authors.
"""Bounded, checkpointable selection of PPO update passes."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

SCHEDULES = {"standard": (0, 4), "short_update": (0, 2), "critic_first": (1, 3)}
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["mode", "supporting_metrics", "reason"],
    "properties": {
        "mode": {"type": "string", "enum": list(SCHEDULES)},
        "supporting_metrics": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
        "reason": {"type": "string", "maxLength": 600},
    },
}
HEADER = "You manage bounded PPO update schedules for a vision-language-action policy."


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


class UpdateController:
    def __init__(self, config, state=None):
        self.config = config
        self.mode = "standard"
        self.history = []
        self.last_decision = {}
        if state:
            self.mode = state["mode"]
            self.history = state["history"]
            self.last_decision = state["last_decision"]
        if self.mode not in SCHEDULES:
            raise ValueError(f"Unsupported saved update mode {self.mode}")

    def state_dict(self):
        return {"version": 1, "mode": self.mode, "history": self.history,
                "last_decision": self.last_decision}

    def choose(self, iteration: int, current: dict, ask=None):
        expected = len(self.history) + 1
        if iteration != expected:
            raise ValueError(f"Controller expected iteration {expected}, got {iteration}")
        policy = self.config.get("policy", "fixed")
        decision = {"iteration": iteration, "mode": self.mode, "reason": "held for two iterations"}
        if iteration <= 2 or policy == "fixed":
            decision.update(mode="standard", reason="fixed baseline or initial warmup")
        elif policy == "forced":
            decision.update(mode=self.config["forced_mode"], reason="engineering verification")
        elif iteration % 2 == 1:
            recent = self.history[-3:]
            if policy == "rule":
                kl = sum(row["update"].get("pc/final_kl", 0.0) for row in recent) / len(recent)
                poor = [row["rollout"].get("value_informative", False)
                        and row["rollout"]["value_explained_variance"] < 0 for row in recent[-2:]]
                selected = "standard"
                if kl > float(self.config["rule_kl_threshold"]):
                    selected = "short_update"
                elif len(poor) == 2 and all(poor):
                    selected = "critic_first"
                decision.update(mode=selected, reason="frozen training-statistic rule")
            elif policy in ("agent", "relay"):
                prompt = HEADER + "\n" + (
                    "Select one schedule for the next two iterations. standard=4 joint passes; "
                    "short_update=2 joint passes; critic_first=1 value-only pass then 3 joint passes. "
                    "Reward, learning rates, task sampling, clipping, and the four-pass cap are fixed. "
                    "Use several observations, not a single noisy statistic. Low value explained variance "
                    "alone does not prove that more value training helps, especially with little return variance. "
                    "Your inputs contain training data only. Reply with the required JSON.\n"
                ) + json.dumps({"iteration": iteration, "current_rollout": current,
                                "previous_iterations": recent}, allow_nan=False)
                try:
                    answer = ask(prompt, SCHEMA)
                    if not isinstance(answer, dict) or answer.get("mode") not in SCHEDULES:
                        raise ValueError("Missing or unsupported controller mode")
                    if not isinstance(answer.get("supporting_metrics"), list) or not isinstance(answer.get("reason"), str):
                        raise ValueError("Controller reply has invalid fields")
                    decision.update(answer)
                except Exception as exc:
                    decision.update(mode="standard", reason="controller fallback", error=type(exc).__name__)
            else:
                raise ValueError(f"Unsupported controller policy {policy}")
        self.mode = decision["mode"]
        self.last_decision = decision
        return decision

    def record(self, iteration, rollout, update):
        if iteration != len(self.history) + 1:
            raise ValueError("Controller history is not consecutive")
        finite = {k: float(v) for k, v in update.items()
                  if isinstance(v, (int, float)) and math.isfinite(float(v))}
        self.history.append({"iteration": iteration, "decision": self.last_decision,
                             "rollout": rollout, "update": finite})


def schedule(mode: str):
    try:
        return SCHEDULES[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported PPO schedule {mode}") from exc
