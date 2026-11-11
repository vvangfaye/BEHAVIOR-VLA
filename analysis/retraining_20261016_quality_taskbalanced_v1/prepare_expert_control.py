"""Prepare an isolated, unsubmitted two-update expert-only control bundle."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'smoke/bundle'
TARGET = ROOT / 'expert_control/smoke/bundle'
RUN = 'b1k-expert-control-1688977-20261016-smoke'


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def replace(path, old, new):
    text = path.read_text()
    if text.count(old) != 1:
        raise ValueError('Unexpected source at ' + str(path))
    path.write_text(text.replace(old, new))


def main():
    if TARGET.exists():
        raise FileExistsError('Inspect existing control; never overwrite it')
    manifest = json.loads((SOURCE / 'bundle_manifest.json').read_text())
    for name, digest in manifest.items():
        if hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Source bundle changed: ' + name)
    shutil.copytree(SOURCE, TARGET, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    sampler = TARGET / 'openpi/src/openpi/training/task_balanced_windows.py'
    replace(sampler, "if set(policy) != expected or policy['kind'] != 'task_balanced_capped_windows_v1':",
        "expected.add('expert_only_control')\n        if policy.get('expert_only_control') is not True or set(policy) != expected or policy['kind'] != 'task_balanced_capped_windows_v1':")
    replace(sampler, "target = math.floor(self.per_task * policy['within_task_full_ceiling'])", "target = 0")
    module = TARGET / 'openpi/src/openpi/training/subtask_mixture.py'
    replace(module,
        'if kinds != {"replay", "full_episode"} or any(s["weight"] <= 0 for s in compiled_sources):',
        'expert_control = manifest.get("quality_sampling", {}).get("expert_only_control") is True\n                weights = {s["kind"]: s["weight"] for s in compiled_sources}\n                valid_weights = weights == {"replay": 1, "full_episode": 0} if expert_control else all(s["weight"] > 0 for s in compiled_sources)\n                if kinds != {"replay", "full_episode"} or not valid_weights:')
    mixture_path = TARGET / 'inputs/compiled/mixture.json'
    mixture = json.loads(mixture_path.read_text())
    mixture['quality_sampling']['expert_only_control'] = True
    for source in mixture['phases'][0]['sources']:
        source['weight'] = 1 if source['kind'] == 'replay' else 0
    save(mixture_path, mixture)
    check = r'''
import json,sys
sys.path.insert(0,sys.argv[1])
from openpi.training.subtask_mixture import MixturePlan
p=MixturePlan.load(sys.argv[2])
assert p._quality_sampler.full_draws==0
for i in range(192*2):
 r=p.reference(i)
 assert r['source_kind']=='replay'
print(json.dumps({'fingerprint':p.fingerprint,'full_draws':p._quality_sampler.full_draws,'checked_windows':384}))
'''
    verified = json.loads(subprocess.check_output([sys.executable, '-c', check, str(TARGET/'openpi/src'), str(mixture_path)], text=True))
    jobpath = TARGET / 'nebula/smoke_job.json'
    job = json.loads(jobpath.read_text())
    job.update(run_id=RUN, output='/data/oss_bucket_0/wangfaye/result/behavior-challenge/expert_control/'+RUN,
        mixture_fingerprint=verified['fingerprint'], experiment='paired_expert_only_control',
        expected_source_counts={'replay':192,'full_episode':0})
    save(jobpath, job)
    smoke = TARGET / 'nebula/four_source_smoke.py'
    replace(smoke, 'real_two_source_batches=2,', 'real_expert_only_batches=2,')
    trainer = TARGET / 'openpi/scripts/b1k/train_b1k.py'
    replace(trainer, 'expected_source_ids = {0, 3} if mixture_plan.two_source else {0, 1, 2, 3}',
        'expert_control = mixture_plan.manifest.get("quality_sampling", {}).get("expert_only_control") is True\n            expected_source_ids = {0} if expert_control else ({0, 3} if mixture_plan.two_source else {0, 1, 2, 3})')
    replace(trainer, '"kind": "real_two_source_batch" if mixture_plan.two_source else "real_four_source_batch"',
        '"kind": "real_expert_only_batch" if expert_control else ("real_two_source_batch" if mixture_plan.two_source else "real_four_source_batch")')
    replace(smoke, "        restore=[x for x in reports if x['kind']=='checkpoint_restore']",
        "        batches=[x for x in reports if x['kind']=='real_expert_only_batch']\n        if len(batches)!=2 or any(x['source_counts']!={'0':192} or x['full_episode_text_target_tokens']!=0 or x['mixture_fingerprint']!=cfg['mixture_fingerprint'] for x in batches):\n            raise ValueError('Expert control requires two exact all-expert global batches')\n        restore=[x for x in reports if x['kind']=='checkpoint_restore']")
    manifest = {str(p.relative_to(TARGET)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(TARGET.rglob('*')) if p.is_file() and p.name!='bundle_manifest.json' and '__pycache__' not in p.parts}
    save(TARGET/'bundle_manifest.json', manifest)
    save(TARGET.parent/'preparation.json', {'run_id':RUN,'verification':verified,'cloud_jobs_created':0,
        'updates':2,'formal_training_authorized':False,'baseline_source_unchanged':True,
        'remaining':'real GPU batches, source counters, and exact checkpoint restore must pass before any formal control run'})
    print(json.dumps({'run_id':RUN, **verified, 'cloud_jobs_created':0}))


if __name__ == '__main__':
    main()
