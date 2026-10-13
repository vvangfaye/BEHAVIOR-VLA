"""Offline boundary tests; real MP4 decoding is additionally required by the GPU pilot."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from video_eval_worker import (
    PROGRESS_HOOK, REQUIRED_SOURCE_FILES, ProgressWatchdog, sha256,
    validate_payload, validate_result, verify_inputs, verify_remote_archive,
)


def payload(root):
    return dict(checkpoint_step=2000000, task="turning_on_radio", instance_id=301,
                run_id="video2000000", execution_name="video-0001", pilot=True,
                checkpoint=str(root / "checkpoint"), code=str(root / "code"),
                runtime=str(root / "runtime"), assignment_root=str(root / "assignment"),
                source_manifest_sha256="a" * 64, checkpoint_ready_sha256="b" * 64,
                artifact_puts=dict(archive="https://example.test/a?secret=x", receipt="https://example.test/r?secret=y"),
                artifact_gets=dict(archive="https://example.test/a?secret=z"))


def result():
    return dict(task="turning_on_radio", instance_id=301, rollout_id=0, steps=123,
                success=False, terminated=False, truncated=True,
                conditioning=dict(mode="whole_task", oracle=False, privileged=False),
                subtask_events=[dict(source="generated"), dict(source="cached")], selector_events=[],
                q_score=dict(final=0.5), time=dict(simulator_steps=123, simulator_time=4.1, normalized_time=1.5),
                normalized_agent_distance=dict(base=0.5, left=0.6, right=0.7))


class WorkerBoundaryTests(unittest.TestCase):
    def test_scope_and_private_urls(self):
        p = payload(Path("/tmp/example"))
        clean = validate_payload(p)
        self.assertNotIn("secret", json.dumps(clean))
        for changes in (dict(checkpoint_step=1688977), dict(instance_id=311), dict(instance_id=True),
                        dict(job_timeout_seconds=96 * 3600 + 1), dict(task="../escape")):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_payload(dict(p, **changes))

    def test_result_requires_same_terminal_episode_and_keeps_zero_scores(self):
        p = payload(Path("/tmp/example"))
        progress = dict(run_id=p["run_id"], steps=123, terminal=True)
        value = result()
        self.assertEqual(validate_result(value, p, progress), (123, 0.5))
        value["q_score"]["final"] = 0
        self.assertEqual(validate_result(value, p, progress), (123, 0))
        for changes in (dict(run_id="different"), dict(steps=122), dict(terminal=False)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_result(value, p, dict(progress, **changes))
        bad = copy.deepcopy(value)
        bad["conditioning"]["privileged"] = True
        with self.assertRaises(ValueError):
            validate_result(bad, p, progress)
        bad = copy.deepcopy(value)
        bad["q_score"]["final"] = 1
        with self.assertRaises(ValueError):
            validate_result(bad, p, progress)

    def test_watchdog_requires_actual_step_growth(self):
        w = ProgressWatchdog(0, init_timeout=1800, stall_timeout=900)
        self.assertIsNone(w.observe(1799, dict(state="hook_ready", steps=0)))
        self.assertEqual(w.observe(1800, dict(state="hook_ready", steps=0)), "no_native_steps_within_initialization_grace")
        w = ProgressWatchdog(0, init_timeout=1800, stall_timeout=900)
        self.assertIsNone(w.observe(1000, dict(steps=1)))
        self.assertIsNone(w.observe(1899, dict(steps=1, observed_at=1899)))
        self.assertEqual(w.observe(1900, dict(steps=1, observed_at=1900)), "native_simulator_progress_stalled")
        self.assertIsNone(w.observe(1901, dict(steps=2)))
        with self.assertRaises(ValueError):
            w.observe(1902, dict(steps=1))

    def test_full_uploaded_bytes_must_match_not_just_content_length(self):
        raw = b"known original archive bytes"
        digest = hashlib.sha256(raw).hexdigest()
        self.assertTrue(verify_remote_archive("unused", len(raw), digest, opener=lambda *_a, **_kw: io.BytesIO(raw)))
        with patch("video_eval_worker.time.sleep"):
            for damaged in (raw[:-1], b"x" * len(raw), raw + b"x"):
                with self.subTest(damaged=damaged), self.assertRaises(RuntimeError):
                    verify_remote_archive("unused", len(raw), digest, opener=lambda *_a, **_kw: io.BytesIO(damaged))

    def test_checkpoint_readback_and_source_coverage(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            p = payload(root)
            code, ckpt, runtime = (Path(p[k]) for k in ("code", "checkpoint", "runtime"))
            for folder in (code, ckpt, runtime):
                folder.mkdir()
            (runtime / "_RUNTIME_READY").write_text("ready")
            for relative in REQUIRED_SOURCE_FILES:
                path = code / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('__version__ = "3.9.3"\n')
            (ckpt / "weights").write_bytes(b"complete weight content")
            (ckpt / "_READY").write_text(json.dumps(dict(checkpoint_step=2000000)))
            source = dict(checkpoint_step=2000000,
                          files_sha256={x: sha256(code / x) for x in REQUIRED_SOURCE_FILES},
                          checkpoint_files=[dict(relative_path="weights", bytes=(ckpt / "weights").stat().st_size, sha256=sha256(ckpt / "weights"))])
            (code / "_PUBLIC_TEST_SOURCE_READY").write_text(json.dumps(source))
            p["source_manifest_sha256"] = sha256(code / "_PUBLIC_TEST_SOURCE_READY")
            p["checkpoint_ready_sha256"] = sha256(ckpt / "_READY")
            self.assertEqual(verify_inputs(p), source)
            (ckpt / "weights").write_bytes(b"incomplete")
            with self.assertRaises(ValueError):
                verify_inputs(p)

    def test_progress_hook_only_advances_after_completed_native_step(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            hook = root / "hook"
            hook.mkdir()
            (hook / "sitecustomize.py").write_text(PROGRESS_HOOK)
            package = root / "omnigibson/eval"
            package.mkdir(parents=True)
            (package / "evaluator.py").write_text('''class Evaluator:
    def __init__(self): self._rollout_steps = 0
    def step(self, action):
        if action == "error": raise RuntimeError("native failed")
        self._rollout_steps += 1
        return (action == "done", False)
''')
            runner = root / "eval_b1k_queue.py"
            runner.write_text('''import json, os
from pathlib import Path
from omnigibson.eval.evaluator import Evaluator
p=Path(os.environ["VIDEO_EVAL_PROGRESS"])
e=Evaluator()
assert json.loads(p.read_text())["state"] == "hook_ready"
assert e.step("go") == (False,False)
assert json.loads(p.read_text())["steps"] == 1
try: e.step("error")
except RuntimeError: pass
else: raise AssertionError("Native error must propagate")
assert json.loads(p.read_text())["steps"] == 1
assert e.step("done") == (True,False)
assert json.loads(p.read_text())["steps"] == 2
assert json.loads(p.read_text())["terminal"] is True
''')
            env = dict(os.environ, PYTHONPATH=str(hook) + os.pathsep + str(root),
                       VIDEO_EVAL_PROGRESS=str(root / "progress.json"), VIDEO_EVAL_RUN_ID="test")
            run = subprocess.run([sys.executable, "-B", str(runner)], env=env, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)


if __name__ == "__main__":
    unittest.main()
