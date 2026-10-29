import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bootstrap import driver_failure_report


class FailureReportingTests(unittest.TestCase):
    def report(self, folder):
        return driver_failure_report({'worker_prefix':'p8_','output':str(folder)},
            {'task_phase':'sample2','rank':6,'attempt':1},
            {'returncode':1,'signal':None,'after':{'cgroup':{'oom_kill':'0'}}})

    def test_preserves_interrupt_and_completed_progress(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'worker_status/p8_sample2_r6_a1.json';path.parent.mkdir()
            path.write_text(json.dumps({'state':'failed','error_type':'ExecutionStopped',
                                        'collected_by_worker':1,'error':'Interrupted'}))
            value=self.report(folder)
            self.assertEqual(value['error_type'],'ExecutionStopped')
            self.assertEqual(value['collected_by_worker'],1)
            self.assertEqual(value['returncode'],1)

    def test_does_not_turn_unknown_crash_into_retryable_interrupt(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(self.report(folder)['error_type'],'DriverProcessExit')

    def test_preserves_nonretryable_programming_error(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'worker_status/p8_sample2_r6_a1.json';path.parent.mkdir()
            path.write_text(json.dumps({'state':'failed','error_type':'AssertionError'}))
            self.assertEqual(self.report(folder)['error_type'],'AssertionError')


if __name__=='__main__':unittest.main(verbosity=2)
