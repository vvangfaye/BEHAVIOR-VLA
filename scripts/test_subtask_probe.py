"""Validation scope and artifact checks remain separate from full-panel scores."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"nebula"))
import eval_b1k_subtask_probe as probe
import subtask_eval
from pi05_eval_queue import ArtifactValidationError


def inputs():
    samples = [{"sample_id":f"sample-{i}", "task":"task", "task_instance_id":2,
                "episode_index":3, "raw_episode_id":4, "task_index":0,
                "rollout_id":i, "subtask":"pick up"} for i in range(2)]
    panel = {"fingerprint":"frozen", "samples":samples}
    plan = {"schema_version":1, "kind":probe.KIND, "panel_fingerprint":"frozen",
            "sample_ids":[s["sample_id"] for s in samples], "max_steps":128}
    return panel, plan, samples


def put_result(root, sample, eligible=True):
    path = root/"pi05"/sample["task"]/"json"/(str(sample["rollout_id"])+".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    result = {**sample, "instance_id":sample["task_instance_id"], "panel_fingerprint":"frozen",
              "steps":10 if eligible else 0, "success":False, "q_score":0.0, "metric_eligible":eligible,
              "result_state":"evaluated" if eligible else "setup_excluded",
              "setup_exclusion_reason":None if eligible else "predicate_already_true",
              "restoration":{"full_snapshot_format":probe.runner.FULL_SNAPSHOT_FORMAT},
              "conditioning":{"source":"provided"}, "validation_kind":probe.KIND,
              "official_benchmark":False}
    path.write_text(json.dumps(result))
    if eligible:
        video = path.parent.parent/"videos"/(path.stem+".mp4")
        video.parent.mkdir(exist_ok=True);video.write_bytes(b"recorded frames")
    return path


@pytest.mark.parametrize("change", ["parent", "duplicate", "foreign", "limit", "episode"])
def test_probe_cannot_change_frozen_scope(tmp_path, change):
    panel, plan, _ = inputs()
    if change == "parent": plan["panel_fingerprint"]="another"
    elif change == "duplicate": plan["sample_ids"]*=2
    elif change == "foreign": plan["sample_ids"][0]="foreign"
    elif change == "limit": plan["max_steps"]=2048
    else: panel["samples"][1]["episode_index"]=99
    path=tmp_path/"plan.json";path.write_text(json.dumps(plan))
    with pytest.raises(ValueError): probe.load_plan(path, panel)


def test_probe_requires_complete_actions_and_video_but_accepts_explicit_exclusions(tmp_path):
    panel, plan, samples = inputs()
    path=tmp_path/"plan.json";path.write_text(json.dumps(plan))
    assert probe.load_plan(path, panel)==(plan,samples)
    put_result(tmp_path, samples[0])
    with pytest.raises(ValueError, match="incomplete"): probe.validate_results(tmp_path,panel,plan,samples)
    put_result(tmp_path, samples[1], eligible=False)
    summary=probe.validate_results(tmp_path,panel,plan,samples)
    assert summary["complete"] and summary["eligible_rollouts"]==1 and summary["official_benchmark"] is False
    assert not (tmp_path/"summary.json").exists()


@pytest.mark.parametrize("change", ["video", "zero_actions", "identity", "format", "label", "limit", "duplicate", "nan"])
def test_probe_rejects_incomplete_or_mislabeled_gpu_outputs(tmp_path, change):
    panel, plan, samples = inputs()
    path=put_result(tmp_path,samples[0]);put_result(tmp_path,samples[1],eligible=False)
    result=json.loads(path.read_text())
    if change=="video": (path.parent.parent/"videos"/(path.stem+".mp4")).write_bytes(b"")
    elif change=="zero_actions": result["steps"]=0
    elif change=="identity": result["raw_episode_id"]=99
    elif change=="format": result["restoration"]={}
    elif change=="label": result["official_benchmark"]=True
    elif change=="limit": result["steps"]=129
    elif change=="nan": result["q_score"]=float("nan")
    else: (path.parent/"duplicate.json").write_text(json.dumps(result))
    path.write_text(json.dumps(result))
    with pytest.raises((ValueError, ArtifactValidationError)): probe.validate_results(tmp_path,panel,plan,samples)


def test_restoration_only_does_not_pass_model_validation(tmp_path):
    panel, plan, samples = inputs()
    for sample in samples: put_result(tmp_path,sample,eligible=False)
    with pytest.raises(ValueError,match="no eligible model rollout"):
        probe.validate_results(tmp_path,panel,plan,samples)


def test_nebula_probe_reuses_server_launcher_and_stages_only_selected_raw(tmp_path, monkeypatch):
    panel, plan, samples = inputs()
    panel_path=tmp_path/"panel.json";panel_path.write_text(json.dumps(panel))
    plan_path=tmp_path/"plan.json";plan_path.write_text(json.dumps(plan))
    checkpoint=tmp_path/"checkpoint";checkpoint.mkdir();(checkpoint/"_READY").write_text('{"step":1375000}')
    args=SimpleNamespace(validation_plan=str(plan_path), panel=str(panel_path), workers=1, sample_max_steps=128,
        run_root=str(tmp_path/"run"), persistent_output=str(tmp_path/"published"), raw_root=str(tmp_path/"raw"),
        raw_source="/mount/raw", checkpoint_source=str(checkpoint), checkpoint="/local/checkpoint",
        run_id="probe", repo_id="joint", pi05_env="/env/policy", eval_env="/env/eval", asset_path="/assets",base_port=8320)
    inventory={"episodes":[{"relative_path":"task-0000/episode_00000004.hdf5"},{"relative_path":"task-0001/episode_00000005.hdf5"}]}
    staged=[]
    def stage(selected, source, destination):
        staged.extend(selected["episodes"]);return tmp_path/"raw"
    def supervise(command, local, published, *, env):
        assert command[:2]==["/bin/bash",str(subtask_eval.REPO_ROOT/"scripts/eval_b1k.sh")]
        assert command[2:]==["task"] and env["SERVER_GPU"]==env["EVAL_GPU"]=="0"
        assert env["MAX_TOKEN_LEN"]=="512" and env["SUBTASK_SAMPLE_MAX_STEPS"]=="128"
        assert "PI05_EVAL_QUEUE_DB" not in env and "EXTERNAL_SUBTASK" not in env
        for sample in samples: put_result(local,sample)
    monkeypatch.setenv("PI05_EVAL_QUEUE_DB","unrelated")
    monkeypatch.setenv("EXTERNAL_SUBTASK","unrelated")
    monkeypatch.setattr(subtask_eval,"stage_raw",stage)
    monkeypatch.setattr(subtask_eval,"supervise_results",supervise)
    subtask_eval.run_validation(args,panel,inventory)
    assert len(staged)==1
    published=Path(args.persistent_output)
    assert json.loads((published/"_SUCCESS").read_text())["kind"]==probe.KIND
    assert json.loads((published/"validation_summary.json").read_text())["samples"]==2
    assert not (published/"summary.json").exists()
