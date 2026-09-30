import asyncio
from types import SimpleNamespace
import msgpack
import numpy as np
import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from goalkeeper.observations import SensorSchema, validate_observation
from goalkeeper.protocol import packb, unpackb
from goalkeeper.goals import compile_bddl
from goalkeeper.policy import GoalkeeperPolicy, JointSubtaskChunkSampler, OpenPIChunkSampler
from goalkeeper.server import handler_factory, health_check
from goalkeeper.belief import Estimate


def observation():
    schema = SensorSchema()
    result = {c + "::rgb": np.zeros((8, 9, 4), np.uint8) for c in schema.cameras}
    result["robot_r1::proprio"] = np.zeros(23, np.float32)
    result["task_id"] = np.array([3], np.int64)
    return result


def test_sensor_boundary_rejects_oracle_and_wrong_task():
    obs = observation()
    clean = validate_observation(obs, SensorSchema(), 3, 23)
    assert clean.rgb[0].shape == (8, 9, 3)
    for key in ["object_states", "robot_world_pose", SensorSchema().head + "::instance_seg"]:
        with pytest.raises(ValueError): validate_observation(obs | {key: {}}, SensorSchema(), 3, 23)
    with pytest.raises(ValueError): validate_observation(obs, SensorSchema(), 4, 23)


def test_wire_format_matches_official_bytes_dictionary():
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    wire = msgpack.packb({"x": {b"__ndarray__": True, b"data": arr.tobytes(), b"dtype": "<f4", b"shape": [2, 3]}}, use_bin_type=True)
    assert np.array_equal(unpackb(wire)["x"], arr)
    assert np.array_equal(unpackb(packb({"x": arr}))['x'], arr)
    with pytest.raises(ValueError): packb(np.array([object()]))
    malformed = msgpack.packb({b"__ndarray__": True, b"data": b"", b"dtype": "<f4", b"shape": [1]})
    with pytest.raises(ValueError): unpackb(malformed)


class Sampler:
    def reset(self): self.calls = 0
    def sample(self, obs, count):
        self.calls += 1
        return np.full((count, 4, 23), self.calls, np.float32)


def make_policy(sampler=None):
    program = compile_bddl("(define (problem p) (:objects a - switch) (:goal (on a)))")
    return GoalkeeperPolicy(program, sampler or Sampler(), task_id=3, proprio_dim=23, action_dim=23, horizon=4)


def test_action_queue_and_episode_reset():
    policy = make_policy()
    for _ in range(4): assert (policy.act(observation()) == 1).all()
    assert (policy.act(observation()) == 2).all()
    policy.reset()
    assert (policy.act(observation()) == 1).all()
    with pytest.raises(ValueError): policy.act(observation() | {"object_states": {}})


def test_sample_calls_raw_infer_not_buffered_act():
    class Raw:
        def __init__(self): self.calls = 0
        def infer(self, batch):
            assert batch["observation/state"].shape == (23,)
            assert {f"observation/image_{i}" for i in range(3)} <= batch.keys()
            assert not any("wrist_image" in key or "egocentric_camera" in key for key in batch)
            self.calls += 1
            return {"actions": np.full((4, 23), self.calls)}
    raw = Raw()
    sampler = OpenPIChunkSampler(raw, "turn on the switch", image_transform=lambda x, size: x)
    candidates = sampler.sample(validate_observation(observation(), SensorSchema(), 3, 23), 4)
    assert candidates.shape == (4, 4, 23)
    assert np.array_equal(candidates[:, 0, 0], [1, 2, 3, 4])


def test_joint_sampler_uses_one_subtask_for_all_candidates_and_resets():
    class Joint:
        def __init__(self): self.calls = 0; self.resets = 0
        def infer(self, batch): raise AssertionError("joint candidates must use infer_candidates")
        def infer_candidates(self, batch, count):
            self.calls += 1
            assert batch["prompt"] == "turn on the switch"
            return {"actions": np.stack([np.full((4, 23), i, np.float32) for i in range(count)]),
                    "subtask": "press switch", "subtask_source": "generated"}
        def reset(self): self.resets += 1

    joint = Joint()
    sampler = JointSubtaskChunkSampler(joint, "turn on the switch", image_transform=lambda x, size: x)
    candidates = sampler.sample(validate_observation(observation(), SensorSchema(), 3, 23), 4)
    assert candidates.shape == (4, 4, 23)
    assert np.array_equal(candidates[:, 0, 0], [0, 1, 2, 3])
    assert joint.calls == 1
    assert sampler.last_subtask == "press switch"
    policy = make_policy(sampler)
    assert policy.act(observation()).shape == (23,)
    assert policy.last_subtasks == {0: "press switch"}
    assert policy.last_subtask_sources == {0: "generated"}
    policy.reset()
    assert policy.last_subtasks == {}
    assert sampler.last_subtask is None
    assert joint.resets == 2


def test_shadow_and_rerank_change_only_selected_candidate():
    program = compile_bddl("(define (problem p) (:objects a - switch) (:goal (on a)))")
    atom, = program.atoms
    class Choices(Sampler):
        def sample(self, obs, count):
            return np.stack([np.full((4, 23), i, np.float32) for i in range(count)])
    class Predictor:
        model = SimpleNamespace(config=SimpleNamespace(horizon=4))
        def reset(self): self.calls = 0
        def predict(self, observation, candidates):
            self.calls += 1
            no, yes = Estimate(0, 0, 0), Estimate(1, 1, 1)
            return {atom: no}, {atom: 1}, [{atom: no}, {atom: yes}]
    for mode, expected in [("shadow", 0), ("rerank", 1)]:
        predictor = Predictor()
        policy = GoalkeeperPolicy(program, Choices(), predictor=predictor, mode=mode, candidates=2,
                                  task_id=3, proprio_dim=23, action_dim=23, horizon=4)
        for _ in range(4): assert (policy.act(observation()) == expected).all()
        assert predictor.calls == 1
        policy.reset(); assert predictor.calls == 0


def test_reset_has_no_ack_and_connections_do_not_share_queues():
    async def scenario():
        async with serve(handler_factory(make_policy, server_token="expected-instance"),
                         "127.0.0.1", 0, process_request=health_check) as server:
            port = server.sockets[0].getsockname()[1]
            async with connect(f"ws://127.0.0.1:{port}", proxy=None) as a:
                metadata = unpackb(await a.recv())
                assert metadata["protocol"] == "b1k-v3.9.2"
                assert metadata["server_instance_token"] == "expected-instance"
                for _ in range(5):
                    await a.send(packb(observation())); await a.recv()
                await a.send(packb({"reset": True}))
                await a.send(packb(observation()))
                response = unpackb(await asyncio.wait_for(a.recv(), 3))
                assert (response["action"] == 1).all()
                async with connect(f"ws://127.0.0.1:{port}", proxy=None) as b:
                    await b.recv(); await b.send(packb(observation()))
                    assert (unpackb(await b.recv())["action"] == 1).all()
    asyncio.run(scenario())


def test_server_rejects_empty_instance_token():
    with pytest.raises(ValueError, match="server_token"):
        handler_factory(make_policy, server_token="")


def test_server_reports_joint_subtask_across_reset():
    class Policy:
        def __init__(self): self.last_subtasks = {}; self.last_subtask_sources = {}
        def act(self, obs):
            self.last_subtasks = {0: "press switch"}
            self.last_subtask_sources = {0: "generated"}
            return np.zeros(23, np.float32)
        def reset(self): self.last_subtasks = {}; self.last_subtask_sources = {}

    async def scenario():
        async with serve(handler_factory(Policy), "127.0.0.1", 0, process_request=health_check) as server:
            port = server.sockets[0].getsockname()[1]
            async with connect(f"ws://127.0.0.1:{port}", proxy=None) as socket:
                await socket.recv()
                await socket.send(packb(observation()))
                response = unpackb(await socket.recv())
                assert response["subtasks"] == ["press switch"]
                assert response["subtask_sources"] == ["generated"]
                await socket.send(packb({"reset": True}))
                await socket.send(packb(observation()))
                assert unpackb(await socket.recv())["subtasks"] == ["press switch"]
    asyncio.run(scenario())
