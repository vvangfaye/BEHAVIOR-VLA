"""Regression test the real tensor conversion and wire serializer, without a GPU."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent/"runtime"
og=types.ModuleType("omnigibson")
og.__path__=[str(ROOT/"repo/BEHAVIOR-1K/OmniGibson/omnigibson")]
sys.modules["omnigibson"]=og
macros=types.ModuleType("omnigibson.macros")
macros.gm=types.SimpleNamespace(DEBUG=False)
sys.modules["omnigibson.macros"]=macros
from omnigibson.learning.utils.array_tensor_utils import torch_to_numpy
from omnigibson.learning.utils.network_utils import Packer,unpackb

spec=importlib.util.spec_from_file_location("audit_under_test",ROOT/"evaluation_audit.py")
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class Parent:
    def _preprocess_obs(self,obs):return dict(obs)


class Evaluator(audit.EvaluationAudit,Parent):
    env=types.SimpleNamespace(_current_step=0)


class ObservationTransportTest(unittest.TestCase):
    def test_original_numpy_marker_reproduces_failure(self):
        with self.assertRaises(AttributeError):
            torch_to_numpy({"_evaluation_step":np.array(0,dtype=np.int64)})

    def test_image_and_serf_observations_round_trip(self):
        for serf in [False,True]:
            for step in [0,1,20,15384]:
                with self.subTest(serf=serf,step=step):
                    obs={"robot_r1::proprio":torch.zeros(256),"task_id":torch.tensor([22])}
                    for camera in ["head","left","right"]:
                        obs[camera]=torch.zeros((224,224,3),dtype=torch.uint8)
                    if serf:
                        obs.update({"observation/points/xyz":torch.zeros(3,3),
                                    "observation/points/feat":torch.zeros(3,64),
                                    "observation/points/robot_mask":torch.zeros(3,dtype=torch.bool)})
                    e=Evaluator();e.env=types.SimpleNamespace(_current_step=step)
                    prepared=e._preprocess_obs(obs)
                    self.assertTrue(torch.is_tensor(prepared["_evaluation_step"]))
                    converted=torch_to_numpy(prepared)
                    wire=unpackb(Packer().pack(converted))
                    self.assertEqual(int(wire["_evaluation_step"]),step)
                    self.assertEqual(set(wire),set(obs)|{"_evaluation_step"})
                    for key in obs:
                        np.testing.assert_array_equal(wire[key],obs[key].numpy())


if __name__=="__main__":unittest.main()
