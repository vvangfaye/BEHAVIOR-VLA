"""Authorization boundaries for extending the actual formal training stage."""
import copy
import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location("formal_worker_tested", Path(__file__).with_name("train_four_source.py"))
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class AuthorizationTest(unittest.TestCase):
    def setUp(self):
        self.job = dict(mode="formal_finetune", formal_training_authorized=True,
                        process_count=2, stage_end_step=5000, max_steps=20000,
                        cache_training_original_files=True, cache_smoke_original_files=False)
        self.auth = dict(formal_training_allowed=True, workers=2, gpus_per_worker=8,
                         max_completed_updates=20000, first_formal_stage_completed_updates=5000)
        self.smoke = dict(completed_updates=2, checkpoint_restore=True, gpus=16)

    def test_old_staged_authorization_does_not_grant_20000(self):
        worker.validate(self.job, self.auth, self.smoke)
        self.job["stage_end_step"] = 20000
        with self.assertRaisesRegex(ValueError, "scientific validation"):
            worker.validate(self.job, self.auth, self.smoke)

    def test_explicit_direct_target_without_claiming_validation(self):
        self.job["stage_end_step"] = 20000
        self.auth.update(execution_policy="direct_to_authorized_target",
                         direct_target_completed_updates=20000, first_formal_stage_completed_updates=20000)
        worker.validate(self.job, self.auth, self.smoke)
        self.assertNotIn("validation_gate_passed", self.job)
        different = copy.deepcopy(self.job)
        different["stage_end_step"] = 10000
        with self.assertRaisesRegex(ValueError, "scientific validation"):
            worker.validate(different, self.auth, self.smoke)

    def test_direct_target_preserves_smoke_and_topology_checks(self):
        self.job["stage_end_step"] = 20000
        self.auth.update(execution_policy="direct_to_authorized_target",
                         direct_target_completed_updates=20000, first_formal_stage_completed_updates=20000)
        with self.assertRaisesRegex(ValueError, "smoke"):
            worker.validate(self.job, self.auth, dict(self.smoke, gpus=8))
        with self.assertRaisesRegex(ValueError, "sixteen GPUs"):
            worker.validate(self.job, dict(self.auth, workers=1), self.smoke)
        with self.assertRaisesRegex(ValueError, "budget"):
            worker.validate(dict(self.job, stage_end_step=20001), self.auth, self.smoke)
        with self.assertRaisesRegex(ValueError, "budget"):
            worker.validate(dict(self.job, max_steps=40000), self.auth, self.smoke)


if __name__ == "__main__":
    unittest.main()
