"""CPU-only submission/worker contracts; no AI-Hub, data transfers or GPUs."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import subtask_mainline as mainline
import train_subtask_mainline as worker


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / "dataset"
    metadata = put(root / "meta/info.json", {"fps": 30})
    index = put(tmp_path / "intervals.json", {
        "schema_version": 1, "kind": "b1k_valid_intervals", "source_kind": "recovered", "action_horizon": 32,
        "held_out_instances": [],
        "root_id": "expert", "repo_id": "b1k/expert", "dataset_root": str(root),
        "episodes": [{"episode_index": 4, "length": 100, "task_name": "task", "task_instance_id": 7}],
        "intervals": [{"episode_index": 4, "start": 10, "end": 100, "subtask": "pick up cup"}],
        "provenance": {"source_files": [{"path": "meta/info.json", "sha256": mainline.sha(metadata), "role": "metadata"},
                                           {"path": "/missing/external/labels.parquet", "sha256": "b" * 64, "role": "labels"}]}})
    content = mainline.read(index)
    content["fingerprint"] = mainline.mixture_module()._digest(content)
    put(index, content)
    mixture = {"schema_version": 1, "kind": "b1k_three_source_mixture", "seed": 42,
               "global_batch_size": 64, "action_horizon": 32, "phases": [{"start_step": 0, "end_step": 500,
                 "sources": [{"kind": "recovered", "weight": 1.0, "indexes": [{"path": str(index), "sha256": mainline.sha(index)}]}]}]}
    config = {"schema_version": 1, "run_id": "mainline_cpu_test", "runtime_python": "/ready/env/bin/python",
              "openpi_data_home": "/ready/cache", "tokenizer_sha256": "d" * 64,
              "output_root": str(tmp_path / "runtime-output"), "data_roots": {"expert": str(root)},
              "resources": {"queue": "user_h200_queue", "gpu_card_type": "h200", "algo_name": "user_runtime", "cpu": 3200, "memory": 256000},
              "checkpoint": {"step": 1375000, "root": "/ready/checkpoint/1375000", "manifest_sha256": "a" * 64,
                             "norm_relative_path": "assets/b1k/original/norm_stats.json", "norm_sha256": "c" * 64},
              "train_steps": 500, "save_interval": 100, "global_batch_size": 64,
              "learning_rate": {"peak": 1e-5, "floor": 1e-6, "warmup_steps": 50, "decay_steps": 2000}, "resume": False}
    return config, mixture, index


@pytest.fixture
def prepared(inputs, tmp_path):
    config, mixture, index = inputs
    output = tmp_path / "prepared"
    payload = mainline.prepare(put(tmp_path / "config.json", config), put(tmp_path / "mixture.json", mixture), output)
    return output, payload, mainline.read(payload / "nebula/mainline_job.json")


def test_prepare_preserves_index_and_remaps_only_path(inputs, prepared):
    _, payload, plan = prepared
    original = inputs[2]
    bundled = payload / "inputs/indexes" / (mainline.sha(original) + ".json")
    assert bundled.read_bytes() == original.read_bytes()
    mixture = mainline.read(payload / "inputs/mixture.json")
    assert mixture["phases"][0]["sources"][0]["indexes"][0] == {
        "path": "indexes/" + mainline.sha(original) + ".json", "sha256": mainline.sha(original)}
    assert mainline.verify_bundle(payload) == plan
    assert mainline.read(payload / "nebula/mainline_cluster.json")["worker"]["gpu"] == 800
    worker.verify_data(payload, plan["config"])


@pytest.mark.parametrize("field,value", [
    ("global_batch_size", 24), ("train_steps", 1234), ("save_interval", 501),
    ("runtime_python", "oss://bucket/env/bin/python"), ("fsdp_devices", 16), ("process_count", 1),
])
def test_rejects_invalid_explicit_budget_or_topology(inputs, field, value):
    config, mixture, _ = inputs
    config[field] = value
    if field == "global_batch_size":
        mixture["global_batch_size"] = value
    with pytest.raises(mainline.MainlineError):
        mainline.validate_config(config, mixture)


def test_missing_resource_and_checkpoint_evidence_rejected(inputs):
    config, mixture, _ = inputs
    del config["resources"]["queue"]
    with pytest.raises(mainline.MainlineError, match="queue"):
        mainline.validate_config(config, mixture)
    config["resources"]["queue"] = "provided"
    config["checkpoint"]["norm_sha256"] = ""
    with pytest.raises(mainline.MainlineError, match="SHA256"):
        mainline.validate_config(config, mixture)


def test_dry_run_has_no_credentials_intent_or_command(prepared):
    output, _, _ = prepared
    def forbidden(*args, **kwargs):
        pytest.fail("dry-run invoked a process")
    result = mainline.submit(output, dry_run=True, environ={}, runner=forbidden)
    assert result["cloud_submitted"] is False
    assert not (output / "submission_intent.json").exists()
    assert not (output / "submission_receipt.json").exists()


def fake_env():
    return {"AI_HUB_TOKEN": "secret-token", "AI_HUB_NAMESPACE": "namespace", "AIHUB_MODEL_NAME": "behavior",
            "OSS_ACCESS_ID": "secret-id", "OSS_ACCESS_KEY": "secret-key", "OSS_BUCKET": "bucket",
            "OSS_ENDPOINT": "endpoint", "AI_HUB_CLI": "/configured/aihub"}


def test_actual_submit_is_explicit_single_attempt_and_redacted(prepared):
    output, payload, _ = prepared
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["cwd"] == payload
        return SimpleNamespace(returncode=0, stdout="secret-token secret-id secret-key job-123")
    result = mainline.submit(output, environ=fake_env(), runner=run)
    assert result["state"] == "submitted"
    assert calls[0][:3] == ["/configured/aihub", "train", "mdl"]
    assert "--worker_count=2" in calls[0] and "--queue=user_h200_queue" in calls[0]
    assert "secret-" not in (output / "submission.log").read_text()
    with pytest.raises(FileExistsError):
        mainline.submit(output, environ=fake_env(), runner=run)
    assert len(calls) == 1


def test_unknown_submission_does_not_allow_blind_retry(prepared):
    output, _, _ = prepared
    def fail(*args, **kwargs):
        raise OSError("network interruption")
    with pytest.raises(mainline.MainlineError, match="outcome unknown"):
        mainline.submit(output, environ=fake_env(), runner=fail)
    assert mainline.read(output / "submission_receipt.json")["state"] == "unknown"
    with pytest.raises(FileExistsError):
        mainline.submit(output, environ=fake_env(), runner=fail)


def test_mutation_blocks_even_dry_run(prepared):
    output, payload, _ = prepared
    (payload / "inputs/roots.json").write_text("{}")
    with pytest.raises(mainline.MainlineError, match="bundle changed"):
        mainline.submit(output, dry_run=True)


@pytest.mark.parametrize("name,memory,count,valid", [
    ("NVIDIA H200", 143771, 8, True), ("NVIDIA H200 NVL", 143771, 8, True),
    ("NVIDIA L20Z", 143771, 8, False), ("NVIDIA H200", 80000, 8, False),
    ("NVIDIA H200", 143771, 7, False), ("NVIDIA H100", 81559, 8, False),
])
def test_actual_h200_hardware_gate(name, memory, count, valid):
    output = (f"{name}, {memory}\n") * count
    if valid:
        assert len(worker.verify_gpus(output)) == 8
    else:
        with pytest.raises(mainline.MainlineError):
            worker.verify_gpus(output)


def test_training_command_keeps_baseline_and_separates_resume(prepared, monkeypatch):
    _, payload, plan = prepared
    monkeypatch.setenv("OPENPI_TRAIN_EPOCHS", "2")
    monkeypatch.setenv("TRAIN_STEPS", "1600000")
    command = worker.training_command(payload, plan)
    assert command[command.index("--num-train-steps") + 1] == "500"
    assert command[command.index("--fsdp-devices") + 1] == "8"
    assert command[command.index("--model.subtask-action-conditioning") + 1] == "task_and_subtask"
    assert "--weight-loader.params-path" in command and "--resume" not in command
    assert "--data.subtask-mixture-path" in command and "--data.subtask-mixture-roots-path" in command
    env = worker.training_environment(payload, plan, 1, "10.0.0.1:12355", payload / "proof.json")
    assert "OPENPI_TRAIN_EPOCHS" not in env and "TRAIN_STEPS" not in env
    assert env["OPENPI_PROCESS_ID"] == "1" and env["OPENPI_EXPECTED_GLOBAL_DEVICES"] == "16"
    assert env["OPENPI_DATA_HOME"] == "/ready/cache"
    assert env["HF_HUB_OFFLINE"] == "1" and env["HF_DATASETS_OFFLINE"] == "1"
    plan["config"]["resume"] = True
    resumed = worker.training_command(payload, plan)
    assert "--resume" in resumed and "--weight-loader.params-path" not in resumed


def test_resume_requires_matching_contract_and_full_optimizer_checkpoint(prepared, tmp_path):
    _, payload, plan = prepared
    norm = put(tmp_path / "norm.json", {"test": "norm"})
    worker.prepare_output(plan, norm, payload)
    plan["config"]["resume"] = True
    plan["attempt_id"] = "next-attempt"
    with pytest.raises(mainline.MainlineError, match="optimizer checkpoint"):
        worker.prepare_output(plan, norm, payload)
    checkpoint = Path(plan["config"]["output_root"]) / "checkpoints/pi05_b1k" / plan["config"]["run_id"] / "99"
    put(checkpoint / "params/_METADATA", {})
    put(checkpoint / "_CHECKPOINT_METADATA", {})
    (checkpoint / "train_state").mkdir()
    module = mainline.mixture_module(payload)
    module.save_consumed_contract(checkpoint / "assets", module.MixturePlan.load(payload / "inputs/mixture.json"), 100)
    worker.prepare_output(plan, norm, payload)
    plan["optimizer_contract_sha256"] = "e" * 64
    with pytest.raises(mainline.MainlineError, match="identical"):
        worker.prepare_output(plan, norm, payload)


def test_resume_can_extend_only_future_phases_without_changing_lr(inputs, prepared, tmp_path):
    _, payload, plan = prepared
    norm = put(tmp_path / "norm.json", {"test": "norm"})
    worker.prepare_output(plan, norm, payload)
    checkpoint = Path(plan["config"]["output_root"]) / "checkpoints/pi05_b1k" / plan["config"]["run_id"] / "99"
    put(checkpoint / "params/_METADATA", {})
    put(checkpoint / "_CHECKPOINT_METADATA", {})
    (checkpoint / "train_state").mkdir()
    module = mainline.mixture_module(payload)
    module.save_consumed_contract(checkpoint / "assets", module.MixturePlan.load(payload / "inputs/mixture.json"), 100)
    config, mixture, _ = copy.deepcopy(inputs)
    config.update(resume=True, train_steps=1000)
    mixture["phases"][0]["end_step"] = 500
    later = copy.deepcopy(mixture["phases"][0])
    later.update(start_step=500, end_step=1000)
    later["sources"][0]["group_weights"] = [{"task_name": "task", "subtask": "pick up cup", "weight": 2.0}]
    mixture["phases"].append(later)
    extended = mainline.prepare(put(tmp_path / "next-config.json", config), put(tmp_path / "next-mixture.json", mixture), tmp_path / "next-bundle")
    next_plan = mainline.read(extended / "nebula/mainline_job.json")
    assert next_plan["optimizer_contract_sha256"] == plan["optimizer_contract_sha256"]
    assert next_plan["training_contract_sha256"] != plan["training_contract_sha256"]
    command = worker.training_command(extended, next_plan)
    assert command[command.index("--lr-schedule.decay-steps") + 1] == "2000"
    worker.prepare_output(next_plan, norm, extended)
    changed_index = mainline.read(inputs[2])
    changed_index["intervals"][0]["start"] = 11
    changed_index["fingerprint"] = module._digest({k: v for k, v in changed_index.items() if k != "fingerprint"})
    changed_path = put(tmp_path / "changed-index.json", changed_index)
    for phase in mixture["phases"]:
        phase["sources"][0]["indexes"] = [{"path": str(changed_path), "sha256": mainline.sha(changed_path)}]
    bad = mainline.prepare(put(tmp_path / "bad-config.json", config), put(tmp_path / "bad-mixture.json", mixture), tmp_path / "bad-bundle")
    with pytest.raises(ValueError, match="consumed|Consumed"):
        worker.prepare_output(mainline.read(bad / "nebula/mainline_job.json"), norm, bad)
    config["learning_rate"]["decay_steps"] = 3000
    drift = mainline.prepare(put(tmp_path / "drift-config.json", config), put(tmp_path / "drift-mixture.json", inputs[1] | {"phases": [dict(inputs[1]["phases"][0], end_step=1000)]}), tmp_path / "drift-bundle")
    with pytest.raises(mainline.MainlineError, match="optimizer/LR"):
        worker.prepare_output(mainline.read(drift / "nebula/mainline_job.json"), norm, drift)


def test_changed_dataset_metadata_rejected(inputs, prepared):
    _, payload, plan = prepared
    root = Path(plan["config"]["data_roots"]["expert"])
    put(root / "meta/info.json", {"fps": 15})
    with pytest.raises(mainline.MainlineError, match="metadata identity"):
        worker.verify_data(payload, plan["config"])


def test_tokenizer_requires_frozen_staged_cache(inputs, tmp_path):
    config = inputs[0]
    config["openpi_data_home"] = str(tmp_path / "cache")
    with pytest.raises(mainline.MainlineError, match="tokenizer cache"):
        worker.verify_tokenizer(config)
    tokenizer = Path(config["openpi_data_home"]) / "big_vision/paligemma_tokenizer.model"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_bytes(b"fixture-tokenizer")
    config["tokenizer_sha256"] = mainline.sha(tokenizer)
    assert worker.verify_tokenizer(config) == tokenizer
    tokenizer.write_bytes(b"changed")
    with pytest.raises(mainline.MainlineError, match="tokenizer cache"):
        worker.verify_tokenizer(config)


def test_checkpoint_requires_full_hashed_baseline(inputs, tmp_path):
    config, _, _ = inputs
    root = tmp_path / "checkpoint"
    norm = {"norm_stats": {feature: {key: [0.0] * 23 for key in ("mean", "std", "q01", "q99")} for feature in ("state", "actions")}}
    for name, value in {"params/_METADATA": {}, "params/array": [1], "_CHECKPOINT_METADATA": {},
                        "assets/subtask_conditioning.json": {"action_conditioning": "task_and_subtask"},
                        config["checkpoint"]["norm_relative_path"]: norm}.items():
        put(root / name, value)
    entries = mainline.inventory(root)
    manifest = root / "SHA256SUMS.inference"
    manifest.write_text("".join(value + "  " + name + "\n" for name, value in sorted(entries.items())))
    put(root / "_READY", {"step": 1375000, "manifest_sha256": mainline.sha(manifest)})
    config["checkpoint"].update(root=str(root), manifest_sha256=mainline.sha(manifest),
                                norm_sha256=mainline.sha(root / config["checkpoint"]["norm_relative_path"]))
    worker.verify_checkpoint(config)
    put(root / "params/array", [2])
    with pytest.raises(mainline.MainlineError, match="content changed"):
        worker.verify_checkpoint(config)
