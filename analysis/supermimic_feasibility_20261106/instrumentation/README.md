# Opt-in state and timing instrumentation

These hooks prepare a **future separately launched pilot**. They are not imported
by any current experiment, do not start OmniGibson, and default to disabled.

`snapshot_entities(robot, objects, arms=..., contact_reader=...)` reads only the
explicit task-object mapping. It records world pose, quaternion in xyzw order,
world linear/angular velocity, and available joint state in the native getter
ordering. Robot joint state is required; object joint state is optional. Missing
required methods or invalid state values raise errors. Optional reads carry
`available` and a bounded failure reason, never fabricated zero values.

The public `robot.is_grasping(arm=..., candidate_obj=...)` API is used only when
arms are explicitly supplied. Its enum name/value is preserved: `UNKNOWN` is not
converted to a true boolean. No private assisted-grasp fields are read. Contact
state is unavailable by default because the current object API has no common
verified getter. An explicit, separately verified `contact_reader` may return a
bounded map with exactly the selected object names; expanding or silently omitting
objects marks contact unavailable. No whole-scene traversal or state dump occurs.

`StepRecorder.record_chunk` wraps individual executed controls. A supplied actual
control counter must increase by one per step. The recorder stores requested and
executed lengths, executed action values, per-control rewards, terminal flags,
before/after structured state, and a persisted **next observation including after
termination**. The observation sink owns array serialization, such as writing an
NPZ with RGB and proprio. Its artifact reference enters the JSON-ready record;
the raw next observation is also returned to the caller.

`physics_dt` and `control_dt` must be supplied from verified runtime configuration.
The elapsed simulated time and physics substeps in the record are explicitly
**inferred from dt**, not claimed as measured hidden physics-engine counters.
Snapshot, individual simulator controls, observation reads and observation I/O
have separate monotonic spans. The overall span includes instrumentation cost;
individual simulator spans exclude before/after state serialization overhead.

## Adapter sketch for a new simulator launcher

This should be added to a new pilot's copy, not patched into a frozen run. The
existing B1K evaluator already exposes `_rollout_steps` and a queued single
`evaluator.step()`; this sketch uses only that explicit counter, not arbitrary
private state. Resolve and verify `selected_objects` from the task contract once.

```python
from instrumentation import ControlOutcome, StepRecorder, TimingRecorder, snapshot_entities

enabled = pilot_plan.get("record_model_transitions", False)
timing = TimingRecorder(enabled=enabled, sink=write_timing_jsonl)

with timing.measure("simulator_initialization"):
    evaluator = create_and_reset_evaluator()
with timing.measure("policy_load"):
    model = load_model()

recorder = StepRecorder(
    enabled=enabled,
    snapshot=lambda: snapshot_entities(
        evaluator.robot, selected_objects, arms=("left", "right"),
    ),
    read_control_count=lambda: evaluator._rollout_steps,
    physics_dt=verified_physics_dt,
    control_dt=verified_control_dt,
    observation_sink=save_next_rgb_proprio_npz,  # (observation, is_terminal) -> path
    record_sink=write_transition_jsonl,
)

with timing.measure("actor"):
    actions, policy_result = model.predict_action_batch(current_observation, mode="train")

# Preserve the existing execution queue and its selected subtask.
policy.actions = list(actions[0])
policy.current_subtask = generated_subtask

def one_control(_action):
    terminated, truncated = evaluator.step()
    return ControlOutcome(bool(terminated), bool(truncated), last_step_reward())

result = recorder.record_chunk(
    actions[0], step_control=one_control, read_observation=read_rgb_proprio_observation,
)
current_observation = result["next_observation"]

with timing.measure("policy_result_io"):
    persist_policy_result(policy_result)
```

Wrap other expensive phases with `timing.measure("name")`: cache construction,
world-model fitting, actor backward, optimizer step, and output writes. Nested
spans overlap and must not be summed as exclusive stage costs. Disabled timing
does not read its clock or call its sink; disabled state recording does not read
physics state, counter, or observation sink. The normal next-observation callback
still runs because it is part of the returned step result.

Exceptions from required state/counter/observation operations propagate. For
failure accounting wrap the caller in `timing.measure("chunk")`, whose span
records an error status. A missing state is not silently written as a valid
transition, and incomplete failed controls must not enter training data.

## Verification

Six dependency-light CPU tests cover required field errors, optional masks,
UNKNOWN grasp preservation, bounded contact scope, actual execution counts,
early termination and terminal observation persistence, disabled behavior, and
timing failure status. These use structural fake entities and no simulator.

```sh
analysis/rlinf_adaptation_20261020/venv/bin/python -m pytest \
  analysis/supermimic_feasibility_20261106/instrumentation/test_recorder.py -q
```

Observed result on 2026-11-06: `6 passed in 0.02s`. Actual API compatibility,
instrumentation overhead, and observation serialization need a future bounded
simulator pilot; they are not validated by these structural tests.
