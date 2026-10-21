"""Prepare, but do not execute, the same panel on the available local GPUs."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Expected exactly one worker adaptation: ' + old)
    return text.replace(old, new)


def main():
    target = HERE / 'local_closed_loop_v1'
    target.mkdir(exist_ok=False)
    previous = HERE / 'closed_loop_v1'
    plan = json.loads((previous / 'plan.json').read_text())
    training_path = HERE / 'local_vla_pilot_ada01/plan.json'
    training = json.loads(training_path.read_text())
    runtime = Path('/mnt/data1/faye/b1k-rehearsal-1688977/runtime')
    old_source = runtime.parent / 'frozen_source'
    code = PROJECT / 'BEHAVIOR-VLAs'
    native_sources = {}
    for relative, expected in plan['critical_files'].items():
        if relative.startswith('BEHAVIOR-1K/'):
            path = old_source / relative
        elif any(relative.startswith(prefix) for prefix in (
                'openpi/src/openpi/configs/', 'openpi/src/openpi/shared/',
                'openpi/src/openpi/serving/', 'openpi/packages/openpi-client/')):
            path = code / relative
        else:
            continue
        if sha(path) != expected:
            raise ValueError('Native inference/evaluation source differs: ' + relative)
        native_sources[str(path)] = expected
    frozen = target / 'frozen'
    frozen.mkdir()
    for name in ['evaluator_entry.py', 'seed_protocol.py', 'serve_vla_native.py',
                 'vla_native_policy.py', 'vla_runtime.py', 'vla_support_adapter.py', 'support_flow.py']:
        shutil.copyfile(previous / 'frozen' / name, frozen / name)
    shutil.copyfile(HERE / 'local_native_env.py', frozen / 'local_native_env.py')
    helpers = (previous / 'frozen/native_helpers.py').read_text()
    needle = 'str(Path(payload["runtime"]) / "envs/b1k-eval/bin/python")'
    assert helpers.count(needle) == 2
    (frozen / 'native_helpers.py').write_text(helpers.replace(needle, 'payload["eval_python"]'))
    worker = (previous / 'frozen/closed_loop_worker.py').read_text()
    replacements = [
        ('from native_helpers import build_env, stop_process, wait_for_policy, verify_video\nfrom source_preflight import source_preflight',
         'from native_helpers import stop_process, wait_for_policy, verify_video\nfrom local_native_env import build_env, source_preflight'),
        ("Path(plan['runtime']) / 'omnigibson-data/2026-challenge-task-instances'", "Path(plan['dataset_root']) / '2026-challenge-task-instances'"),
        ("'--format=csv,noheader'], text=True)", "'--format=csv,noheader', '-i', '0'], text=True)"),
        ("('5090', 'N12LP')", "('RTX 6000 Ada',)"),
        ("'Expected one5090 GPU'", "'Expected the selected RTX6000Ada GPU'"),
        ("Path('/var/tmp/behavior-1k-runtime')", "Path('/tmp/b1k-action-support-native')"),
        (" or cache.stat().st_dev == Path(plan['runtime']).stat().st_dev", ''),
        ("load(Path(plan['supplemental_library']['root']) / '_READY')['library_path']", "'/usr/lib/x86_64-linux-gnu'"),
        ("str(runtime / 'envs/b1k-pi05/bin/python')", "plan['policy_python']"),
        ("str(runtime / 'envs/b1k-eval/bin/python')", "plan['eval_python']"),
    ]
    for before, after in replacements:
        worker = replace_once(worker, before, after)
    (frozen / 'closed_loop_worker.py').write_text(worker)
    plan.update(code=str(code), runtime=str(runtime), dataset_root=str(runtime / 'datasets'),
                policy_python=str(PROJECT / 'analysis/rlinf_adaptation_20261020/venv/bin/python'),
                eval_python=str(runtime / 'b1k-eval/bin/python'),
                checkpoint=training['checkpoint'], tokenizer=training['tokenizer'],
                openpi_cache=str(PROJECT / 'analysis/organizer_rehearsal_1688977_20261016/runtime/policy-bundle/rootfs/mnt/afs/260010165/behavior_1k/runtime_20261007_1306/openpi-cache'),
                upstream=str(PROJECT / 'analysis/rlinf_adaptation_20261020/upstream'),
                adapter=str(PROJECT / 'BEHAVIOR-VLAs/rlinf_adapter'),
                site=str(PROJECT / 'analysis/rlinf_adaptation_20261020/venv/lib/python3.11/site-packages'),
                afs_root=str(target / 'episodes'),
                training_roots={method: str(HERE / 'local_vla_pilot_ada01' / method / 'unused')
                                for method in training['methods']},
                native_source_sha256=native_sources, training_plan_sha256=sha(training_path),
                parent_cloud_plan_sha256=sha(previous / 'plan.json'),
                status='prepared_not_run; require all training complete and selected GPU idle',
                source_sha256={p.name: sha(p) for p in frozen.glob('*.py')})
    # The local runner already writes result/run/checkpoint directly under the
    # method folder; retain the worker's training_root/output convention via
    # an explicit local path adaptation rather than making filesystem links.
    worker = replace_once(worker, "Path(plan['training_roots'][method]) / 'output'", "Path(plan['training_roots'][method])")
    (frozen / 'closed_loop_worker.py').write_text(worker)
    plan['training_roots'] = {method: str(HERE / 'local_vla_pilot_ada01' / method) for method in training['methods']}
    plan['source_sha256']['closed_loop_worker.py'] = sha(frozen / 'closed_loop_worker.py')
    plan['resource_bound'].update(type='existing_local_gpu', quota_type=None, gpu_per_job=1)
    (target / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps({'prepared': True, 'native_sources_verified': len(native_sources), 'episodes_launched': 0}))


if __name__ == '__main__':
    main()
