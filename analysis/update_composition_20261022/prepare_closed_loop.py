"""Prepare existing demonstration subsets and matched, bounded trial plans.

Local preparation only: no upload, job, simulator or policy execution.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
AFS='/mnt/afs/260010165/behavior_1k'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    target=HERE/'closed_loop_pilot'
    if target.exists():raise ValueError('Refusing to overwrite an already prepared trial')
    source=ROOT/'analysis/action_support_20261021/vla_pilot'
    full=json.loads((source/'data_plan.json').read_text())
    receipt=json.loads((source/'image_receipt.json').read_text())
    assert sha(source/'data_plan.json')==receipt['data_plan_sha256']
    assert sha(source/full['raw_targets']['path'])==full['raw_targets']['sha256']
    assert sha(source/receipt['image_file'])==receipt['image_sha256']
    raw=np.load(source/full['raw_targets']['path'])
    images=np.load(source/receipt['image_file'],mmap_mode='r')
    rng=np.random.default_rng(17022)
    names=['tidying_living_room','polishing_shoes']
    heldout='unloading_the_car'
    pools={task: [i for i,s in enumerate(full['samples'])
                 if s['task_name']==task and int(raw['fold'][i]) in (2,3,4)]
           for task in names+[heldout]}
    def pair(task):
        first=int(rng.choice(pools[task]))
        second=int(rng.choice([i for i in pools[task] if raw['episode'][i]!=raw['episode'][first]]))
        return [first,second]
    rows=[];selected=set()
    for index in range(16):
        task=names[index%2];local_iteration=index//2;support=pair(task);selected.update(support)
        rows.append(dict(task=task,support_indices=support,
            instances=[2*local_iteration+1,2*local_iteration+2],max_steps=20000,
            fm_seed=8000+index,fm_times=rng.beta(1.5,1.,size=2).clip(.001,.999).tolist(),
            rollout_seeds=[18000+index*2,18001+index*2],
            environment_seeds=[9000+index*2,9001+index*2],
            cut_step=int(rng.integers(0,20000)),before_is_updated=bool(local_iteration%2)))
    evaluation_support=pair(heldout);selected.update(evaluation_support)
    ids=sorted(selected);mapping={old:new for new,old in enumerate(ids)}
    target.mkdir();data=target/'data';data.mkdir()
    np.save(data/'rgb.npy',np.array(images[ids]))
    np.savez_compressed(data/'raw_targets.npz',**{key:raw[key][ids] for key in raw.files})
    subset=copy.deepcopy(full)
    subset.update(samples=[full['samples'][i] for i in ids],sample_count=len(ids),
        raw_targets=dict(path='raw_targets.npz',sha256=sha(data/'raw_targets.npz')))
    write(data/'data_plan.json',subset)
    minimal_receipt=dict(receipt,data_plan_sha256=sha(data/'data_plan.json'),
        image_sha256=sha(data/'rgb.npy'),image_bytes=(data/'rgb.npy').stat().st_size,
        shape=list(images[ids].shape))
    write(data/'image_receipt.json',minimal_receipt)
    for row in rows:row['support_indices']=[mapping[i] for i in row['support_indices']]
    old_plan=json.loads((ROOT/'analysis/rlinf_adaptation_20261020/cloud_smoke/plan.json').read_text())
    environment={k:old_plan[k] for k in ['code','runtime','supplemental_library','supplemental_library_sha256']}
    environment.update(simulator_common=AFS+'/rlinf_adapter_20261020/smoke01/run/common.py',
        simulator_common_sha256=sha(ROOT/'analysis/rlinf_adaptation_20261020/cloud_smoke/common.py'),
        startup_timeout_seconds=1800)
    common=dict(admission_result=AFS+'/update_composition_20261022/plumbing04/output/result.json',
        checkpoint=AFS+'/rlinf_adapter_20261020/smoke01/checkpoint',
        tokenizer=old_plan['tokenizer'],data_directory=AFS+'/update_composition_20261022/pilot_data_v1',
        seed=17,rank=4,precision='bf16',inner_lr=.001,inner_epsilon=1e-6,
        outer_lr=.0001,outer_epsilon=1e-6,max_grad_norm=1.,compatibility_weight=.1,bc_weight=1.,
        max_training_episodes=32,max_seconds=86400,environment=environment,iterations=rows,
        trial_status='prepared_not_submitted',data_plan_sha256=sha(data/'data_plan.json'))
    for method in ('meta','maml','fixed_crossplay','rl_bc'):
        write(target/(method+'.json'),dict(common,method=method))
    eval_cases=[dict(instance=instance,environment_seed=seed,policy_seed=100000+instance*10+seed,
                     max_steps=20000) for instance in range(301,307) for seed in range(4)]
    evaluation={k:common[k] for k in ('checkpoint','tokenizer','data_directory','seed','rank','precision',
                                    'inner_lr','inner_epsilon','environment')}
    evaluation.update(meta_training_tasks=names,trial_status='prepared_not_submitted',
        tasks=[dict(task_name=heldout,adaptation_batches=[dict(indices=[mapping[i] for i in evaluation_support],
              seed=20999,times=[.25,.75])],cases=eval_cases)])
    write(target/'evaluate.json',evaluation)
    write(target/'provenance.json',dict(source_data_plan_sha256=sha(source/'data_plan.json'),
        source_image_sha256=receipt['image_sha256'],source_indices=ids,
        source_episode_ids=[int(raw['episode'][i]) for i in ids],
        selected_observations=len(ids),new_demonstrations=0,new_simulation_episodes=0,
        planned_training_episodes=128,planned_evaluation_episodes_per_arm=24,
        qualification='Three-task implementation trial; not a sufficient CVPR claim. Runtime admission and full comparison protocols still required.'))
    print(json.dumps(dict(prepared=str(target),selected_observations=len(ids),
                          rgb_bytes=(data/'rgb.npy').stat().st_size,submitted=False)))


if __name__=='__main__':main()
