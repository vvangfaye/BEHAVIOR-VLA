"""Guard actual policy RNG against changes to simulator initialization."""
import json
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
import experiment


class Simulator:
    def __init__(self, plan, *args):
        experiment.seed_all(plan['seed'])
        self.seed = plan['seed']
        self.current = {'status': {'steps': 0}, 'observation': {
            'state': np.zeros(23), 'images': [np.zeros((2,2,3))]*3, 'prompt': 'test'}}
    def step(self, actions, text):
        self.current = {'status': {'steps': 16, 'score': 0., 'success': False},
                        'rewards': [0.], 'terminal': True, 'done': True}
        return self.current
    def close(self):
        pass


class Model:
    def predict_action_batch(self, *args, **kwargs):
        action = torch.randn(1,16,23) + random.random()
        return action, {'prev_values': torch.zeros(1), 'sample': action.clone()}


def main():
    experiment.start_simulator = lambda plan,*args: Simulator(plan)
    experiment.enforce_storage_budget = lambda *args: None
    experiment.label_text = lambda *args: 'test'
    with tempfile.TemporaryDirectory() as tmp:
        outputs = []
        for name, seed in [('a',123), ('b',124), ('c',123)]:
            plan={'seed':0,'output':str(Path(tmp)/name),'train_instances':[1],
                  'train_seeds':[[seed]],'train_steps_per_round':16}
            episode=experiment.collect(Model(),plan,1,0,lambda **kw:None)
            outputs.append(episode['transitions'][0]['sample'])
        assert not torch.equal(outputs[0],outputs[1])
        torch.testing.assert_close(outputs[0],outputs[2],atol=0,rtol=0)
    result={'passed':True,'different_episode_seeds_change_actual_sample':True,
            'same_episode_seed_reproduces_actual_sample':True,
            'sampler_code_changed':False,'false_alarm_corrected':True}
    Path(__file__).with_name('sampling_seed_test_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
