"""Run the native evaluator, changing only paired RNG seed and recording ACK."""
import json
import os
from pathlib import Path
from omnigibson.eval import eval as native
from seed_protocol import SeededClient, PROTOCOL

native.DEFAULT_EVAL_SEED = 0
Original = native.Evaluator


class Evaluator(Original):
    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        policy = self.policy
        if isinstance(policy, SeededClient):
            policy = policy.wrapped
        self.policy = SeededClient(policy, {'protocol': PROTOCOL, 'policy_seed': 0})
        return result

    def step(self, *args, **kwargs):
        terminated, truncated = super().step(*args, **kwargs)
        if terminated or truncated:
            proof = self.policy.proof()
            path = Path(os.environ['PAIRED_SEED_PROOF'])
            with path.open('x') as stream:
                json.dump(dict(proof, environment_seed=0), stream)
        return terminated, truncated


native.Evaluator = Evaluator
native.main()
