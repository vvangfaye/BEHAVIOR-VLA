import copy
import json
from pathlib import Path
import unittest

from mid_rewards import Milestones, Signals, box_gap

CONFIG=json.loads(Path(__file__).with_name('plan.json').read_text())['mid_reward']


class RewardTests(unittest.TestCase):
    def test_once_only_and_cap(self):
        reward=Milestones(CONFIG,Signals(2.,1.,False))
        amounts=[]
        for signal in [Signals(.8,1.,False)]*3+[Signals(.8,.01,False)]*3+[Signals(3.,3.,False)]*5+[Signals(.8,.01,False)]*10:
            amounts.append(reward.step(signal)[0])
        self.assertEqual(sum(amounts),1.5)
        self.assertEqual([e['milestone'] for e in reward.events],['near','finger'])

    def test_initially_satisfied_is_not_rewarded_later(self):
        reward=Milestones(CONFIG,Signals(.5,.01,False))
        for signal in [Signals(3.,3.,False)]*3+[Signals(.5,.01,False)]*5:
            self.assertEqual(reward.step(signal)[0],0.)

    def test_dwell_is_consecutive_and_hand_requires_near(self):
        reward=Milestones(CONFIG,Signals(3.,1.,False))
        for signal in [Signals(.5,1.,False)]*2+[Signals(3.,.0,False)]*5:
            self.assertEqual(reward.step(signal)[0],0.)
        for _ in range(2):self.assertEqual(reward.step(Signals(.5,.01,False))[0],0.)
        self.assertEqual(reward.step(Signals(.5,.01,False))[0],1.5)

    def test_environment_terminal_reward_is_not_duplicated(self):
        a=Milestones(CONFIG,Signals(3.,1.,False));b=Milestones(CONFIG,Signals(3.,1.,True))
        for _ in range(5):
            self.assertEqual(a.step(Signals(.5,.01,False))[0],b.step(Signals(.5,.01,True))[0])

    def test_fail_closed_on_invalid_state(self):
        for value in [float('nan'),float('inf'),-1.]:
            with self.assertRaises(ValueError):Milestones(CONFIG,Signals(value,1.,False))

    def test_box_gap_uses_switch_region(self):
        self.assertAlmostEqual(box_gap(([0,0,0],[1,1,1]),([2,1,1],[3,2,2])),1.)
        self.assertEqual(box_gap(([0,0,0],[1,1,1]),([.9,.9,.9],[2,2,2])),0.)


if __name__=='__main__':unittest.main()
