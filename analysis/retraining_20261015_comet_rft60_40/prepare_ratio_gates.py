"""Prepare independent CPU/GPU acceptance bundles; never submit cloud jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'formal_stage/bundle/openpi/src'))
from openpi.training.subtask_mixture import MixturePlan


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def prepare(trial):
    if not trial or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in trial):
        raise ValueError('Unsafe trial name')
    parent = ROOT/'ratio_trials'/trial
    prepared = read(parent/'preparation.json')
    mixture = read(parent/'mixture.json')
    formal = read(parent/'worker_job.prepared.json')
    plan = MixturePlan(mixture, base_dir=ROOT/'formal_stage/bundle/inputs/compiled')
    if plan.fingerprint != prepared['mixture_fingerprint'] or formal['formal_training_authorized']:
        raise ValueError('Prepared mixture identity or inert launch state differs')
    for source in mixture['phases'][0]['sources']:
        if source['weight'] != prepared['weights'][source['kind']]:
            raise ValueError('Prepared ratio changed')
    output = parent/'gates'
    if output.exists():
        raise FileExistsError('Acceptance bundles already exist; never overwrite a potential attempt')
    # Validate every source byte before creating either new bundle.
    bases = {name: ROOT/base/'bundle' for name, base in [('cpu', 'cpu_repair'), ('smoke', 'gpu_smoke')]}
    manifests = {}
    for name, base in bases.items():
        manifests[name] = read(base/'bundle_manifest.json')
        for relative, digest in manifests[name].items():
            path = Path(relative)
            if path.is_absolute() or '..' in path.parts or (base/path).is_symlink() or sha(base/path) != digest:
                raise ValueError('Unsafe or changed acceptance source: '+relative)
        source_job = read(base/'nebula/smoke_job.json')
        expected = (0, 1) if name == 'cpu' else (2, 2)
        if (source_job['updates'], source_job['process_count']) != expected:
            raise ValueError('Acceptance topology/update budget changed: '+name)
    stages = {}
    for name, base in bases.items():
        bundle = output/name/'bundle'
        for relative in manifests[name]:
            dest = bundle/relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(base/relative, dest)
        write(bundle/'inputs/compiled/mixture.json', mixture)
        job = read(bundle/'nebula/smoke_job.json')
        job.update(run_id=formal['run_id']+'-'+name,
                   output=formal['output']+'-'+name,
                   mixture_fingerprint=plan.fingerprint,
                   formal_training_authorized=False,
                   gpu_smoke_prefix='PENDING_THIS_TRIAL_ACCEPTANCE',
                   resume=False)
        write(bundle/'nebula/smoke_job.json', job)
        if name == 'cpu':
            write(bundle/'probe_job.json', dict(run_id=job['run_id'], output=job['output']))
        actual = MixturePlan.load(bundle/'inputs/compiled/mixture.json')
        if actual.fingerprint != plan.fingerprint:
            raise ValueError('Acceptance bundle mixture differs')
        manifest = {str(p.relative_to(bundle)): sha(p) for p in bundle.rglob('*')
                    if p.is_file() and p.name != 'bundle_manifest.json'}
        allowed = {'inputs/compiled/mixture.json', 'nebula/smoke_job.json'}
        if name == 'cpu':
            allowed.add('probe_job.json')
        changed = {key for key in manifest if manifest[key] != manifests[name][key]}
        if changed != allowed:
            raise ValueError('Unexpected acceptance-code or input change')
        write(bundle/'bundle_manifest.json', manifest)
        stages[name] = dict(bundle_manifest_sha256=sha(bundle/'bundle_manifest.json'),
                           changed_files=sorted(changed), files=len(manifest),
                           entry='four_source_cpu_probe.py' if name == 'cpu' else 'nebula/four_source_smoke.py',
                           output=job['output'], run_id=job['run_id'],
                           allocated_gpus=1 if name == 'cpu' else 16,
                           uses_gpu=name != 'cpu', optimizer_updates=0 if name == 'cpu' else 2,
                           passed=False, submitted=False)
    receipt = dict(kind='inert_ratio_acceptance_bundles', trial=trial,
                   mixture_fingerprint=plan.fingerprint, stages=stages,
                   old_success_proofs_reused=False, formal_training_authorized=False,
                   cloud_jobs_created=0,
                   required_before_submission='Complete current fixed20 comparison and choose this ratio; then use new isolated attempts and current authorization.')
    write(output/'preparation.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.trial)))
