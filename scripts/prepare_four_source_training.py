#!/usr/bin/env python3
"""Build frozen four-source indexes and LR contract. Never run training."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'openpi/src'))
from openpi.training.subtask_mixture import MixturePlan
from openpi.shared.learning_rate_contract import make_contract


def dump(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False)+'\n')


def convert_index(index, kind):
    index = copy.deepcopy(index)
    digest = hashlib.sha256(json.dumps({k:v for k,v in index.items() if k != 'fingerprint'},
                                      sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    if index.get('fingerprint') != digest:
        raise ValueError('Input index fingerprint mismatch')
    previous = index.get('source_kind')
    if previous != ('collected' if kind == 'full_episode' else kind):
        raise ValueError('Unexpected source kind in prepared index')
    if kind == 'full_episode':
        if index.get('provenance', {}).get('label_semantics') != 'self_generated_weak_subtask':
            raise ValueError('Full-episode index lacks explicit weak-text provenance')
        index['source_kind'] = kind
        index['provenance']['language_loss_enabled'] = False
    semantic = {k:v for k,v in index.items() if k != 'fingerprint'}
    index['fingerprint'] = hashlib.sha256(json.dumps(semantic, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    return index


def prepare(plan_path, data_manifest_path, descriptors, replay, recovered, output):
    plan = json.loads(Path(plan_path).read_text())
    raw_manifest = Path(data_manifest_path).read_bytes()
    data = json.loads(raw_manifest)
    if not data['data_ready'] or data['source_fingerprint'] != plan['dataset']['source_fingerprint']:
        raise ValueError('Data readiness or identity differs from the approved plan')
    if hashlib.sha256(raw_manifest).hexdigest() != plan['dataset']['cloud_manifest']['sha256']:
        raise ValueError('Data manifest bytes differ from the accepted cloud manifest')
    if data['missing_archives'] or data['duplicate_collected_content']:
        raise ValueError('Data inventory incomplete or duplicated')
    if output.exists():
        raise ValueError('Use a new output directory for the immutable training preparation')
    indexes = [('replay', json.loads(Path(replay).read_text())), ('recovered', json.loads(Path(recovered).read_text()))]
    descriptor_hashes = {}
    skipped = []
    for receipt in data['archives']:
        path = Path(descriptors)/(receipt['root_id']+'.json')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != receipt['index_target']['sha256']:
            raise ValueError('Prepared descriptor SHA mismatch: '+receipt['root_id'])
        descriptor_hashes[receipt['root_id']] = receipt['index_target']['sha256']
        descriptor = json.loads(raw)
        if descriptor['index'] is None:
            if receipt['training_eligible_episodes'] != 0:
                raise ValueError('Missing index for an eligible archive')
            skipped.append(receipt['root_id'])
            continue
        indexes.append(('full_episode' if receipt['kind']=='full_episode' else 'collected', descriptor['index']))
    weights = dict(replay=plan['sampling']['weights']['original_expert_replay'],
                   recovered=plan['sampling']['weights']['recovered_expert'],
                   collected=plan['sampling']['weights']['successful_subtask'],
                   full_episode=plan['sampling']['weights']['successful_full_episode'])
    output.mkdir(parents=True)
    (output/'indexes').mkdir()
    refs = {k:[] for k in weights}; counts = Counter(); roots = {}
    for number, (kind, index) in enumerate(indexes):
        if not index['episodes'] and not index['intervals']:
            skipped.append(index['root_id']);continue
        index = convert_index(index, kind)
        path = output/'indexes'/f'{number:04d}_{kind}.json';dump(path,index)
        refs[kind].append({'path':path.relative_to(output).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        counts[kind] += len(index['episodes']);roots[index['root_id']] = index['dataset_root']
    sources = [dict(kind=k, weight=weights[k], indexes=refs[k], group_sampling='task_balanced') for k in weights]
    manifest = dict(schema_version=2, kind='b1k_four_source_mixture', seed=plan['sampling']['seed'],
                    global_batch_size=plan['sampling']['global_batch_size'], action_horizon=32,
                    phases=[dict(start_step=0,end_step=plan['budget']['max_new_optimizer_updates'],sources=sources)])
    compiled = MixturePlan(manifest,base_dir=output)
    expected=plan['dataset']['eligible_episodes_by_source']
    if dict(counts) != dict(replay=expected['original_expert_replay'], recovered=expected['recovered_expert'],
                            collected=expected['successful_subtask'],full_episode=expected['successful_full_episode']):
        raise ValueError('Four-source eligible episode totals differ from plan')
    dump(output/'mixture.json',manifest);dump(output/'roots.json',roots)
    lr=plan['learning_rate']
    # Match the actual schedule dataclass without importing the model/JAX here.
    import dataclasses
    Schedule=dataclasses.make_dataclass('CosineDecaySchedule',[('warmup_steps',int),('peak_lr',float),('decay_steps',int),('decay_lr',float)])
    contract=make_contract(Schedule(lr['warmup_steps'],lr['peak_lr'],lr['decay_end_step'],lr['floor_lr']),peak_limit=lr['peak_lr'])
    dump(output/'learning_rate_contract.json',contract)
    sampled=Counter(compiled.reference(i)['source_kind'] for i in range(100000))
    result=dict(training_started=False,metadata_ready=True,real_training_batch_verified=False,
                source_manifest_cloud_sha256=plan['dataset']['cloud_manifest']['sha256'],
                local_data_manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
                mixture_fingerprint=compiled.fingerprint,eligible_episodes=dict(counts),skipped_empty_roots=skipped,
                sampled_100000=dict(sampled),roots_require_training_host_mapping=True,
                learning_rate_contract_sha256=contract['sha256'],source_descriptors=descriptor_hashes)
    dump(output/'preparation.json',result)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','data-manifest','descriptors','replay','recovered','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();result=prepare(a.plan,a.data_manifest,a.descriptors,a.replay,a.recovered,a.output)
    print(json.dumps({k:v for k,v in result.items() if k != 'source_descriptors'}))
