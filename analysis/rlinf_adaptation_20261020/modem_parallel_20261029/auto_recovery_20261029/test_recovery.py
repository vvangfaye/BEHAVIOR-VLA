from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recovery_policy import attempt_cap,gpu_seconds_upper,other_rl_jobs


class RecoveryTests(unittest.TestCase):
    def test_stopped_job_uses_terminal_timestamp_not_full_reservation(self):
        job={'state':'SUSPENDED','start_time':'2026-10-29T14:00:00Z','update_time':'2026-10-29T22:05:00+08:00'}
        self.assertEqual(gpu_seconds_upper(job,{'max_seconds':5400,'gpu_count':1}),300.)

    def test_active_job_reserves_the_whole_limit(self):
        job={'state':'RUNNING','start_time':'2026-10-29T14:00:00Z'}
        self.assertEqual(gpu_seconds_upper(job,{'max_seconds':5400},datetime(2026,10,29,14,5,tzinfo=timezone.utc)),5400.)

    def test_never_started_terminal_job_uses_no_gpu(self):
        self.assertEqual(gpu_seconds_upper({'state':'SUSPENDED'}, {'max_seconds':5400}),0.)

    def test_retry_cap_is_explicit_bounded_and_phase_scoped(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/'auto_recovery_policy.json').write_text(json.dumps({'eligible_phases':['sample2'],'max_attempts_per_shard':8}))
            plan={'max_attempts_per_phase':2}
            self.assertEqual(attempt_cap(plan,'sample2',p),8)
            self.assertEqual(attempt_cap(plan,'baseline',p),2)

    def test_other_task_baseline_occupies_a_shared_slot(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);root=p/'modem';root.mkdir()
            target=p/'multitask_validation_20261029/trash/jobs/baseline_r0_a1';target.mkdir(parents=True)
            (target/'submission.json').write_text(json.dumps({'job':'baseline','uid':'known'}))
            def describe(_):return {'uid':'known','state':'RUNNING','ownership':{'user_name':'260010165'}}
            self.assertEqual(len(other_rl_jobs(root,describe)),1)


if __name__=='__main__':unittest.main(verbosity=2)
