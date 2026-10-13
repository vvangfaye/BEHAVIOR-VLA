import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from video_eval_finalize import compute_summary, inspect_report, run, sha256


def plan(root):
    return dict(checkpoint_step=2000000, run_id="two-million-video", object_prefix="behavior/test-video",
                afs_output_root=str(root / "afs"),
                immutable_inputs=dict(source_manifest_sha256="a" * 64, checkpoint_ready_sha256="b" * 64),
                episodes=[dict(index=t*10+i, task="task_"+str(t), instance_id=301+i, rollout=0)
                          for t in range(100) for i in range(10)])


def report(p, episode, *, create=False):
    index, task, instance = episode["index"], episode["task"], episode["instance_id"]
    name = p["run_id"] + "-e" + str(index).zfill(4)
    success = index % 10 == 0
    score = 1.0 if success else 0.25
    # Non-canonical whitespace proves packaging retains original JSON bytes.
    contents = {f"json/{task}_{instance}_0.json": json.dumps(dict(task=task, instance_id=instance, rollout_id=0,
                    success=success, steps=3, q_score=dict(final=score)), indent=3).encode() + b"\n\n",
                f"videos/{task}_{instance}_0.mp4": b"previously independently verified video fixture",
                "config/r1pro.yaml": b"model: r1pro\n", "wrapper/default_wrapper.py": b"# exact wrapper\n",
                "metadata/execution.json": b'{"write_video": true}'}
    files = [dict(path=k, bytes=len(v), sha256=hashlib.sha256(v).hexdigest()) for k, v in contents.items()]
    if create:
        bundle = Path(p["afs_output_root"]) / "episodes" / str(index).zfill(4) / "bundle"
        for key, value in contents.items():
            target = bundle / key
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(value)
    return dict(checkpoint_step=2000000, task=task, instance_id=instance, rollout_id=0, run_id=name,
                execution_name=name, source_manifest_sha256="a"*64, checkpoint_ready_sha256="b"*64,
                state="complete", verified=True, archive_readback_verified=True, full_video_decode_verified=True,
                same_attempt_json_and_video=True, canonical_results_kept_regardless_of_score=True,
                steps=3, model_success=success, q_score=score,
                video_validation=dict(decoded_frames=3, advertised_frames=3, full_sequential_decode=True,
                                      frames_equal_simulator_steps=True),
                archive=dict(sha256="c"*64, bytes=100), files=files)


class FinalizationTests(unittest.TestCase):
    def test_partial_denominators_and_missing_episodes_are_explicit(self):
        p = plan(Path("/tmp/example"))
        accepted = [dict(index=0, report=report(p, p["episodes"][0]))]
        result = compute_summary(p, accepted)
        self.assertEqual(result["completed"], 1)
        self.assertEqual(result["success_rate_completed"], 1)
        self.assertEqual(result["success_rate_missing_zero"], 0.001)
        self.assertEqual(result["q_missing_zero"], 0.001)
        self.assertFalse(result["all_results_verified"])
        task = next(t for t in result["per_task"] if t["task"] == "task_0")
        self.assertEqual(task["missing_instances"], list(range(302, 311)))
        with self.assertRaises(ValueError):
            compute_summary(p, accepted + accepted)

    def test_receipt_source_and_video_identity_are_strict(self):
        p = plan(Path("/tmp/example"))
        e = p["episodes"][0]
        original = report(p, e)
        for changes in (dict(source_manifest_sha256="f"*64), dict(instance_id=311),
                        dict(same_attempt_json_and_video=False), dict(archive_readback_verified=False)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                inspect_report(p, e, dict(original, **changes))

    def test_incomplete_collection_never_creates_package(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            p = plan(root)
            destination = root / "final"
            with self.assertRaises(ValueError):
                run(dict(plan=p, accepted=[], destination=str(destination), readme="instructions"))
            self.assertFalse(destination.exists())

    def test_full_1000_zip_preserves_original_bytes_and_correct_denominator(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            p = plan(root)
            accepted = [dict(index=e["index"], uid="uid-"+str(e["index"]), report=report(p, e, create=True)) for e in p["episodes"]]
            out = run(dict(plan=p, accepted=accepted, destination=str(root / "final"),
                           readme="Frozen evaluation command and reproducible serving instructions."))
            self.assertEqual(out["completed"], 1000)
            self.assertEqual(out["successes"], 100)
            self.assertEqual(out["success_rate"], 0.1)
            self.assertEqual(out["q_score"], 0.325)
            self.assertFalse(out["direct_videos_uploaded"])
            self.assertEqual(out["state"], "local_package_complete")
            with zipfile.ZipFile(out["zip_path"]) as zipped:
                self.assertEqual(len([x for x in zipped.namelist() if x.startswith("json/")]), 1000)
                raw = zipped.read("json/task_0_301_0.json")
                expected = Path(p["afs_output_root"]) / "episodes/0000/bundle/json/task_0_301_0.json"
                self.assertEqual(raw, expected.read_bytes())
                self.assertTrue(raw.endswith(b"\n\n"))
            self.assertEqual(sha256(out["zip_path"]), out["zip_sha256"])
            index = json.loads((root / "final/VIDEO_INDEX.json").read_bytes())
            self.assertEqual(len(index), 1000)
            self.assertTrue(all(x["uploaded"] is False for x in index))


if __name__ == "__main__":
    unittest.main()
