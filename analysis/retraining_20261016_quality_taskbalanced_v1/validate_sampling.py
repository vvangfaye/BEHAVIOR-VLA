"""Exhaustively audit the finite metadata plan, not the model or real data reader."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
BUNDLE=ROOT/'formal_stage/bundle'
sys.path.insert(0,str(BUNDLE/'openpi/src'))
from openpi.training.subtask_mixture import MixturePlan, consumed_contract, compare_consumed_contract


def main():
    started=time.time();p=MixturePlan.load(BUNDLE/'inputs/compiled/mixture.json')
    source_count,task_count,episode_count,window_count=Counter(),Counter(),Counter(),Counter()
    holds=json.loads((ROOT.parent/'retraining_20261015_comet_rft60_40/high_exposure_review/quality_mask.proposed.json').read_text())['proposed_hold_intervals']
    held=set(p._held_out)
    first_batches=[Counter(),Counter()]
    for i in range(p.num_samples):
        r=p.reference(i)
        assert r['segment_start']<=r['frame'] and r['frame']+32<=r['segment_end']
        assert (r['task_name'],r['task_instance_id']) not in held
        source_count[r['source_kind']]+=1;task_count[r['task_name']]+=1
        if i<384:first_batches[i//192][r['source_kind']]+=1
        if r['source_kind']=='full_episode':
            key=(r['root_id'],r['episode_index'])
            episode_count[key]+=1;window_count[(*key,r['frame'])]+=1
            assert all(h['root_id']!=r['root_id'] or not(r['frame']<h['end_frame'] and r['frame']+32>h['start_frame']) for h in holds)
    assert len(task_count)==100 and set(task_count.values())=={9600}
    assert source_count=={'replay':911171,'full_episode':48829}
    assert max(episode_count.values())<=256 and max(window_count.values())<=2
    assert all(set(b)=={'replay','full_episode'} for b in first_batches)
    # Independently reloaded workers must map the same global index identically.
    other=MixturePlan.load(BUNDLE/'inputs/compiled/mixture.json')
    probes=list(range(512))+list(range(p.num_samples-512,p.num_samples))
    for rank in range(16):
        for i in probes[rank::16]:assert p.reference(i)==other.reference(i)
    saved=consumed_contract(p,2);compare_consumed_contract(saved,other,2)
    for cap in [128,512]:
        mutated=copy.deepcopy(p.manifest);mutated['quality_sampling']['full_episode_draw_cap']=cap
        try:MixturePlan(mutated,base_dir=BUNDLE/'inputs/compiled')
        except ValueError:pass
        else:raise AssertionError('Changed quota accepted without new source budget')
    extended=copy.deepcopy(p.manifest);extended['phases'][0]['end_step']=10000
    try:MixturePlan(extended,base_dir=BUNDLE/'inputs/compiled')
    except ValueError:pass
    else:raise AssertionError('Finite sampler silently extended')
    old=ROOT.parent/'retraining_20261015_comet_rft60_40/formal_stage/bundle'
    original_hashes=json.loads((old/'bundle_manifest.json').read_text())
    for relative,h in original_hashes.items():assert hashlib.sha256((old/relative).read_bytes()).hexdigest()==h
    result={'passed':True,'scope':'All960000 metadata references; real decoding and GPU proof still pending',
        'samples':p.num_samples,'mixture_fingerprint':p.fingerprint,'source_draws':dict(source_count),
        'tasks':len(task_count),'samples_per_task':9600,'max_full_episode_draws':max(episode_count.values()),
        'max_full_window_repetitions':max(window_count.values()),'first_two_batches':[dict(b) for b in first_batches],
        'global_index_partition_replay_verified':True,'consumed_contract_restoration_verified':True,
        'quota_mutations_rejected':True,'budget_extension_rejected':True,'held_out_instances':len(held),
        'old_frozen_bundle_unchanged':True,'new_training_jobs':0,'elapsed_seconds':time.time()-started}
    (ROOT/'metadata_sampling_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
