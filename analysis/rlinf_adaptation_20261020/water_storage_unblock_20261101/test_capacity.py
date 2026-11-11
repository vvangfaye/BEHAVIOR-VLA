import json,time,unittest
from pathlib import Path
from unittest import mock
import capacity_candidate as c

PLAN=json.loads((Path(__file__).parent.parent/'multitask_validation_20261029/water/plan.json').read_text())
class CapacityTests(unittest.TestCase):
    def proof(self,free,spent=0):
        now=time.time();owned={'experiment':PLAN['experiment'],'output':PLAN['output'],
            'method':'committed_regular_files_no_external_hardlinks','observed_at':now,'committed_bytes':spent*c.GIB}
        quota={'rows':[{'quota_bytes':2048*c.GIB,'used_bytes':(2048-free)*c.GIB}],'observed_at':now,'method':'exact'}
        with mock.patch.object(c,'_owned',return_value=owned if spent else None):
            return c.evidence(PLAN,quota,True,False)
    def test_initial_full_reserve_unchanged(self):
        p=self.proof(90);self.assertEqual(p['required_remaining_bytes'],84*c.GIB);self.assertTrue(c.acceptable(p,PLAN))
    def test_committed_data_not_reserved_twice(self):
        p=self.proof(70,20);self.assertEqual(p['new_task_reserve_bytes'],64*c.GIB);self.assertEqual(p['required_remaining_bytes'],64*c.GIB);self.assertTrue(c.acceptable(p,PLAN))
    def test_real_insufficient_remaining_space_rejected(self):
        self.assertFalse(c.acceptable(self.proof(60,20),PLAN))
    def test_forged_credit_rejected(self):
        p=self.proof(70,20);p['committed_task_bytes']=40*c.GIB;self.assertFalse(c.acceptable(p,PLAN))
    def test_stale_measurement_rejected(self):
        p=self.proof(70,20);p['owned_storage']['observed_at']-=181;self.assertFalse(c.acceptable(p,PLAN))
    def test_other_task_measurement_rejected(self):
        p=self.proof(70,20);p['owned_storage']['experiment']='another';self.assertFalse(c.acceptable(p,PLAN))
    def test_queued_worker_can_use_consistent_snapshot(self):
        p=self.proof(70,20);self.assertTrue(c.acceptable(p,PLAN,now=time.time()+600))
    def test_bytes_written_after_quota_snapshot_not_credited(self):
        now=time.time();owned={'experiment':PLAN['experiment'],'output':PLAN['output'],
            'method':'committed_regular_files_no_external_hardlinks','observed_at':now,'committed_bytes':20*c.GIB}
        quota={'rows':[{'quota_bytes':2048*c.GIB,'used_bytes':1978*c.GIB}],'observed_at':now-1,'method':'exact'}
        with mock.patch.object(c,'_owned',return_value=owned):p=c.evidence(PLAN,quota,True,False)
        self.assertEqual(p['committed_task_bytes'],0);self.assertFalse(c.acceptable(p,PLAN))
if __name__=='__main__':unittest.main()
