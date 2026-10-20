import importlib.util
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location("submit_screen", Path(__file__).with_name("submit_screen.py"))
screen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(screen)


class SubmitScreenTest(unittest.TestCase):
    def archive(self, folder, entries):
        path = Path(folder) / "test.tar.gz"
        with tarfile.open(path, "w:gz") as out:
            for name, data, kind in entries:
                info = tarfile.TarInfo(name)
                info.type = kind
                info.size = len(data)
                if kind == tarfile.SYMTYPE:
                    info.linkname = "/outside"
                out.addfile(info, io.BytesIO(data))
        return path

    def test_archive_rejects_escape_links_duplicates_and_path_collisions(self):
        for entries in [
            [("../escape", b"x", tarfile.REGTYPE)],
            [("/escape", b"x", tarfile.REGTYPE)],
            [("link", b"", tarfile.SYMTYPE)],
            [("a", b"x", tarfile.REGTYPE), ("a", b"y", tarfile.REGTYPE)],
            [("a", b"x", tarfile.REGTYPE), ("a/b", b"y", tarfile.REGTYPE)],
        ]:
            with self.subTest(entries=entries), tempfile.TemporaryDirectory() as folder:
                with self.assertRaises(ValueError):
                    screen.archive_files(self.archive(folder, entries))

    def test_archive_accepts_safe_pax_long_name(self):
        name = "project/" + "nested/" * 20 + "module.py"
        with tempfile.TemporaryDirectory() as folder:
            result = screen.archive_files(self.archive(folder, [(name, b"code", tarfile.REGTYPE)]))
            self.assertEqual(result, {name: b"code"})

    def test_default_preflight_checks_real_frozen_bundle_without_network(self):
        with mock.patch.object(screen, "submit", side_effect=AssertionError("No submit during preflight")), \
             mock.patch("builtins.print") as printed:
            self.assertEqual(screen.main([]), 0)
        result = json.loads(printed.call_args.args[0])
        self.assertEqual(result["state"], "local_preflight_passed")
        self.assertEqual(result["patch_bytes"], 22502)
        self.assertEqual(result["archive_entries"], 231)
        self.assertEqual(set(result["changed_archive_entries"]), screen.PATCH_NAMES)
        self.assertEqual(result["outer_term_seconds"] + result["outer_kill_after_seconds"], 3600)

    def test_intent_is_exclusive(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "intent.json"
            screen.save_state(path, {"attempted": False}, exclusive=True)
            with self.assertRaises(FileExistsError):
                screen.save_state(path, {"attempted": True}, exclusive=True)
            self.assertEqual(json.loads(path.read_text()), {"attempted": False})

    def test_exact_name_reconciliation_discards_remote_command_and_urls(self):
        secret = "https://example.invalid/?X-Amz-Signature=SECRET"
        response = {"name": screen.JOB, "uid": "a" * 32, "state": "QUEUEING", "command": secret}
        runner = mock.Mock(return_value=SimpleNamespace(returncode=0, stdout=json.dumps(response)))
        state = screen.reconcile({"job_name": screen.JOB}, runner)
        self.assertEqual(state["uid"], "a" * 32)
        self.assertNotIn(secret, json.dumps(state))
        self.assertNotIn("command", state)
        self.assertEqual(runner.call_args.args[0][4], screen.JOB)
        self.assertTrue(runner.call_args.kwargs["capture_output"])
        runner.return_value = SimpleNamespace(returncode=0, stdout=json.dumps({**response, "name": "wrong-job"}))
        self.assertEqual(screen.reconcile({}, runner)["state"], "submission_response_requires_exact_name_query")

    def test_timeout_exception_does_not_persist_signed_argv(self):
        runner = mock.Mock(side_effect=subprocess.TimeoutExpired(["SECRET signed command"], timeout=60))
        state = screen.reconcile({}, runner)
        self.assertNotIn("SECRET", json.dumps(state))
        self.assertEqual(state["query_error_type"], "TimeoutExpired")

    def test_bootstrap_is_valid_python_and_uses_outer_total_bound(self):
        receipt = {"parent_bundle_bytes": 54651349, "changed_archive_entries": {
            name: "a" * 64 for name in screen.PATCH_NAMES}}
        command = screen.bootstrap_command(receipt, "https://parent.invalid", "https://patch.invalid", "https://put.invalid")
        self.assertTrue(command.startswith("timeout --signal=TERM --kill-after=60 3540 "))
        import shlex
        args = shlex.split(command)
        compile(args[-1], "<bootstrap>", "exec")
        self.assertIn("set(patch) != set(c['changes'])", args[-1])
        self.assertIn("'max_seconds'] != 3600", args[-1])

    def test_bootstrap_downloads_and_applies_only_three_verified_members(self):
        def compressed(files):
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
                for name, data in files.items():
                    member = tarfile.TarInfo(name)
                    member.size = len(data)
                    archive.addfile(member, io.BytesIO(data))
            return buffer.getvalue()

        plan = json.dumps({"max_seconds": 3600, "sample_count": 36, "save_raw_actions": True,
                           "training_executed": False, "simulator_executed": False,
                           "checkpoints": [{"name": name} for name in screen.CHECKPOINTS]}).encode()
        patch = {"plan.json": plan, "cloud_probe_worker.py": b"new worker",
                 "project/BEHAVIOR-VLAs/scripts/eval_openloop.py": b"new evaluator"}
        parent = {**{name: b"old" for name in patch}, "project/untouched": b"same cache"}
        blobs = {"https://parent.invalid": compressed(parent), "https://patch.invalid": compressed(patch)}

        class Response(io.BytesIO):
            status = 200

            def __init__(self, data):
                super().__init__(data)
                self.headers = {"Content-Length": str(len(data))}

        opener = SimpleNamespace(open=lambda url, **kwargs: Response(blobs[url]))
        with tempfile.TemporaryDirectory() as folder:
            config = {"root": str(Path(folder) / "job"), "python": "/fake/python",
                      "parent_url": "https://parent.invalid", "patch_url": "https://patch.invalid",
                      "put_url": "https://put.invalid/?secret=DO_NOT_LOG", "checkpoints": screen.CHECKPOINTS,
                      "changes": {name: hashlib.sha256(data).hexdigest() for name, data in patch.items()}}
            for kind in ("parent", "patch"):
                data = blobs[config[kind + "_url"]]
                config[kind + "_bytes"] = len(data)
                config[kind + "_sha"] = hashlib.sha256(data).hexdigest()
            code = screen._BOOTSTRAP.replace("__CONFIG__", repr(config), 1)
            with mock.patch("urllib.request.build_opener", return_value=opener), \
                 mock.patch("subprocess.call", return_value=0) as worker, \
                 mock.patch.dict(os.environ):
                # This executes only the bootstrap against in-memory fake HTTP;
                # the worker itself is mocked and no SCO/model call occurs.
                original_umask = os.umask(0o077)
                try:
                    with self.assertRaises(SystemExit) as stopped:
                        exec(compile(code, "<fake-bootstrap>", "exec"), {})
                finally:
                    os.umask(original_umask)
                self.assertEqual(stopped.exception.code, 0)
                self.assertEqual(worker.call_count, 1)
            result = Path(config["root"]) / "bundle"
            self.assertEqual({p.relative_to(result).as_posix(): p.read_bytes()
                              for p in result.rglob("*") if p.is_file()}, {**parent, **patch})


if __name__ == "__main__":
    unittest.main()
