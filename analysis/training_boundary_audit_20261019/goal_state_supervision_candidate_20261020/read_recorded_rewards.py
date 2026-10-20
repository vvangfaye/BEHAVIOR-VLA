"""Fetch only recorded reward/termination columns for four selected episodes."""
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from prepare_probe_cache import ObjectReader, TransferBudget, open_source_bucket

def main():
    import numpy as np
    import pyarrow.parquet as pq
    p = json.loads((HERE/'natural_order_source_plan.json').read_text())
    target = HERE/'recorded_reward_audit.json'
    if target.exists() and json.loads(target.read_text()).get('complete'):
        print(target.read_text())
        return
    episodes = sorted({s['episode_index'] for s in p['samples']})
    data = [r for r in p['objects'] if r['relative_path'].startswith('data/')]
    assert len(data) == 1
    budget = TransferBudget(64*1024**2, 180)
    result = {'complete':False,'episodes':[],'model_calls':0,'simulator_calls':0}
    try:
        bucket = open_source_bucket(Path.home()/'.ossutilconfig', p['source_bucket'])
        table = pq.read_table(ObjectReader(bucket,data[0],budget),
            columns=['episode_index','frame_index','timestamp','next.reward','next.terminated','next.truncated'],
            filters=[('episode_index','in',episodes)], pre_buffer=False, use_threads=False).to_pydict()
        arrays = {k:np.asarray(v).reshape(-1) for k,v in table.items()}
        np.savez_compressed(HERE/'recorded_rewards.npz',**arrays)
        for ep in episodes:
            ids = np.flatnonzero(arrays['episode_index']==ep)
            ids = ids[np.argsort(arrays['frame_index'][ids])]
            rewards = arrays['next.reward'][ids]
            assert np.isfinite(rewards).all()
            nonzero = ids[np.abs(rewards)>1e-8]
            result['episodes'].append({
                'episode':ep,'frames':len(ids),'reward_min':float(rewards.min()),
                'reward_max':float(rewards.max()),'reward_sum':float(rewards.sum()),
                'unique_rewards':[float(x) for x in np.unique(rewards)],
                'reward_events':[{'frame':int(arrays['frame_index'][i]),'timestamp':float(arrays['timestamp'][i]),'reward':float(arrays['next.reward'][i])} for i in nonzero],
                'terminated_frames':[int(arrays['frame_index'][i]) for i in ids if arrays['next.terminated'][i]],
                'truncated_frames':[int(arrays['frame_index'][i]) for i in ids if arrays['next.truncated'][i]],
            })
        result['complete'] = True
    except BaseException as error:
        result['failure_type'] = type(error).__name__
    result['range_bytes_charged'] = budget.used_bytes
    result['range_requests'] = budget.requests
    result['interpretation'] = 'Recorded columns only. Reward semantics require verified exporter/task provenance; cumulative reward is not automatically a per-goal state vector.'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
