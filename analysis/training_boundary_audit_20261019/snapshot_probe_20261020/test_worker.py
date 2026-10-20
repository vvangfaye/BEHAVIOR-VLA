import copy
import unittest
import worker


def sample():
    return {'object':{'position':[0,0,0],'orientation':[0,0,0,1]},
            '__robot__':{'joint_positions':[0,0],'joint_velocities':[0,0],
                         'held_objects':{'left':None},'grasp_counter':{'left':None},
                         'release_counter':{'left':None}}}


class ErrorContract(unittest.TestCase):
    def test_quaternion_sign_is_equivalent(self):
        a=sample();b=copy.deepcopy(a);b['object']['orientation']=[0,0,0,-1]
        self.assertTrue(worker.passed(worker.state_error(a,b),1e-5))

    def test_wrong_grasp_is_rejected(self):
        a=sample();b=copy.deepcopy(a);b['__robot__']['held_objects']['left']='jar'
        self.assertFalse(worker.passed(worker.state_error(a,b),1e-5))

    def test_missing_counter_restore_is_rejected(self):
        a=sample();b=copy.deepcopy(a);b['__robot__']['grasp_counter']['left']=2
        self.assertFalse(worker.passed(worker.state_error(a,b),1e-5))

    def test_matching_pose_does_not_hide_velocity_difference(self):
        a=sample();b=copy.deepcopy(a);b['__robot__']['joint_velocities'][0]=0.01
        self.assertFalse(worker.passed(worker.state_error(a,b),1e-5))

    def test_topology_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):worker.state_error(sample(),{})


if __name__=='__main__':unittest.main()
