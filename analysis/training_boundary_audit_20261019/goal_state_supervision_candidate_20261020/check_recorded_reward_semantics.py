"""Check a candidate reward interpretation; do not assert historical provenance."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent

def main():
    raw_path = HERE/'recorded_rewards.npz'
    rows = []
    with np.load(raw_path,allow_pickle=False) as data:
        for ep in sorted(np.unique(data['episode_index'])):
            indices = np.flatnonzero(data['episode_index']==ep)
            indices = indices[np.argsort(data['frame_index'][indices])]
            rewards = data['next.reward'][indices].astype(float)
            flags = data['next.terminated'][indices].astype(bool)
            first = int(np.flatnonzero(flags)[0]) if flags.any() else None
            if first is None:
                continue
            rewards = rewards[:first+1]
            # Conditional test: r_t = Q_{t+1}-Q_t, Q in [0,1],
            # and the first terminated transition reaches Q=1.
            initial_q = 1.0-float(rewards.sum())
            qs = initial_q+np.concatenate(([0.0],np.cumsum(rewards)))
            events = np.flatnonzero(np.abs(rewards)>1e-8)
            rows.append({
                'episode':int(ep),
                'first_terminated_frame':int(data['frame_index'][indices[first]]),
                'reward_sum_through_first_termination':float(rewards.sum()),
                'implied_initial_Q_under_candidate_semantics':initial_q,
                'implied_Q_min':float(qs.min()),
                'implied_Q_max':float(qs.max()),
                'bounded_Q_interpretation_compatible':bool(qs.min()>=-1e-8 and qs.max()<=1+1e-8),
                'events':[{'frame':int(data['frame_index'][indices[i]]),'reward':float(rewards[i])} for i in events],
            })
    report={
        'source_npz_sha256':hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        'hypothesis_tested':'Recorded next.reward is a unit-scale difference of a [0,1] goal-completion score, and first next.terminated means score 1.',
        'results':rows,
        'conclusion':'The joint candidate interpretation fails for at least one inspected episode; do not integrate this field into exact goal-state labels without resolving reward and termination provenance.',
        'not_proven':['Dataset reward corruption','Actual goal regression at every negative reward','Which goal changed at an event','The historical exporter matches the current checkout'],
        'historical_source_limit':'The available merged info.json has no omnigibson_git_hash. Current and frozen local wrapper/task code are reference implementations, not proof of the exact July dataset producer.',
        'next_requirement':'Obtain original per-goal status or original reward/termination configuration, or independently recompute selected events from recorded simulator states.',
        'model_or_simulator_calls':0,
    }
    (HERE/'recorded_reward_semantics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    for row in rows:
        print(json.dumps({k:v for k,v in row.items() if k!='events'},ensure_ascii=False))

if __name__=='__main__':
    main()
