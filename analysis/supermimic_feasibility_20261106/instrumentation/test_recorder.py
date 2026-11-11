from enum import IntEnum

import pytest

from .recorder import ControlOutcome, StepRecorder, TimingRecorder, snapshot_entities


class Grasp(IntEnum):
    FALSE = -1
    TRUE = 1
    UNKNOWN = 0


class Entity:
    n_joints = 0

    def get_position_orientation(self):
        return [1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0]

    def get_linear_velocity(self):
        return [0.1, 0.0, 0.0]

    def get_angular_velocity(self):
        return [0.0, 0.1, 0.0]


class Robot(Entity):
    n_joints = 2

    def get_joint_positions(self):
        return [0.2, 0.3]

    def get_joint_velocities(self):
        return [0.4, 0.5]

    def is_grasping(self, *, arm, candidate_obj):
        return Grasp.UNKNOWN


def test_required_dynamics_and_optional_availability():
    state = snapshot_entities(Robot(), {"target": Entity()}, arms=["left"])
    assert state["robot"]["joints"]["available"]
    assert not state["objects"]["target"]["joints"]["available"]
    assert not state["contacts"]["available"]
    assert state["grasp"]["left"]["value"]["target"]["name"] == "UNKNOWN"
    entity = Entity()
    entity.get_linear_velocity = None
    with pytest.raises(ValueError, match="Required dynamics method missing"):
        snapshot_entities(Robot(), {"target": entity})


def test_contact_reader_cannot_expand_scope_or_silently_omit_targets():
    for result in ({"unrelated": True}, {}):
        state = snapshot_entities(Robot(), {"target": Entity()}, contact_reader=lambda *_: result)
        assert not state["contacts"]["available"]
        assert "exactly" in state["contacts"]["reason"]


def test_terminal_next_observation_partial_chunk_and_real_control_count():
    count = [0]
    clock_value = [100.0]
    stored, records = [], []

    def clock():
        clock_value[0] += 0.01
        return clock_value[0]

    def step(action):
        count[0] += 1
        return ControlOutcome(terminated=count[0] == 2, reward=float(action[0]))

    def store(obs, terminal):
        stored.append((obs, terminal))
        return "terminal-observation.npz"

    recorder = StepRecorder(
        enabled=True, snapshot=lambda: snapshot_entities(Robot(), {"target": Entity()}),
        read_control_count=lambda: count[0], physics_dt=0.005, control_dt=0.02,
        observation_sink=store, record_sink=records.append, clock=clock,
    )
    result = recorder.record_chunk([[1.0], [2.0], [3.0]], step_control=step,
                                   read_observation=lambda: {"state": [count[0]]})
    record = result["record"]
    assert count[0] == 2 and record["requested_control_steps"] == 3
    assert record["executed_control_steps"] == 2
    assert record["execution_actions"] == [[1.0], [2.0]]
    assert record["inferred_physics_substeps"] == 8
    assert record["physics_substep_count_is_measured"] is False
    assert stored == [({"state": [2]}, True)]
    assert result["next_observation"] == {"state": [2]}
    assert len(records) == 1 and record["wall_seconds"] > 0
    assert {span["stage"] for span in record["timing_spans"]} == {
        "state_before", "sim_control", "observation_read", "state_after", "observation_io",
    }


def test_bad_counter_and_required_configuration_fail_visibly():
    with pytest.raises(ValueError, match="requires snapshot"):
        StepRecorder(enabled=True)
    recorder = StepRecorder(
        enabled=True, snapshot=lambda: {}, read_control_count=lambda: 0,
        physics_dt=0.005, control_dt=0.02, observation_sink=lambda *_: "obs.npz",
    )
    with pytest.raises(ValueError, match="exactly one"):
        recorder.record_chunk([[0]], step_control=lambda _: ControlOutcome(), read_observation=lambda: {})


def test_off_by_default_does_not_read_physics_state():
    def forbidden():
        raise AssertionError("disabled recorder must not read state")
    recorder = StepRecorder(snapshot=forbidden, read_control_count=forbidden)
    result = recorder.record_chunk([[0]], step_control=lambda _: ControlOutcome(truncated=True),
                                   read_observation=lambda: {"terminal": True})
    assert result["record"] is None and result["next_observation"]["terminal"]


def test_phase_timing_marks_failures_and_off_is_noop():
    timer = TimingRecorder(enabled=True)
    with pytest.raises(RuntimeError):
        with timer.measure("actor"):
            raise RuntimeError("failed inference")
    assert timer.spans[0]["stage"] == "actor" and timer.spans[0]["status"] == "error"
    off = TimingRecorder()
    with off.measure("initialization"):
        pass
    assert off.spans == []
