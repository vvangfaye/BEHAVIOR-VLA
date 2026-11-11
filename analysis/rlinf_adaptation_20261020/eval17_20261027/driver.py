"""Evaluate immutable original and 17-update weights; never construct an optimizer."""
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import tarfile
import time
import traceback
import urllib.request

import torch

from common import ExecutionStopped, torch_model, write
from eval_utils import assigned_cases, paired_report, schedule
from experiment import evaluate, seed_all
from training import hash_parameters, load_overlay


def upload(url, data):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(urllib.request.Request(url, data=data, method='PUT'), timeout=20):
        pass


def main():
    plan = json.loads(Path(__file__).with_name('plan.json').read_text())
    assert plan['no_training'] and plan['checkpoint']['completed_updates'] == 17
    rank = int(os.environ['B1K_RANK'])
    root = Path(plan['remote_root'])/f'worker_{rank}'/'results'
    plan['output'] = str(root)
    root.mkdir(parents=True, exist_ok=False)
    write(root/'plan.json', plan)
    cases = assigned_cases(plan, rank)
    jobs = schedule(plan, rank)
    state = {'state': 'running', 'stage': 'loading', 'rank': rank,
             'job_name': os.environ['B1K_JOB_NAME'], 'started_at': time.time(),
             'no_training': True, 'new_optimizer_steps': 0, 'completed_new_episodes': 0,
             'expected_new_episodes': len(jobs), 'checkpoint_updates': 17}

    def publish(**fields):
        state.update(fields, updated_at=time.time())
        write(root/'status.json', state)
        try:
            upload(os.environ['B1K_STATUS_PUT'], json.dumps(state).encode())
        except Exception as error:
            write(root/'status_upload_error.json', {'type': type(error).__name__, 'at': time.time()})

    def report():
        value = paired_report(root, cases, plan['evaluation']['paired_initial_pose_tolerance'])
        value.update(rank=rank, no_training=True, checkpoint_updates=17, new_optimizer_steps=0)
        write(root/'comparison.json', value)
        return value

    def deadline(*_):
        raise ExecutionStopped('Frozen evaluation time budget exhausted or stopped')

    signal.signal(signal.SIGTERM, deadline)
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(plan['compute_timeout_seconds'])
    try:
        # Preserve verified baseline bytes. No completed baseline is rerun.
        reused = {tuple(x) for x in plan['reuse']['baseline_cases']}
        for instance, seed in cases:
            if (instance, seed) not in reused:
                continue
            prefix = f'i{instance}_s{seed}'
            for name in ('result.json', 'initial.json', 'spec.json'):
                relative = prefix+'/'+name
                data = (Path(plan['reuse']['baseline_root'])/relative).read_bytes()
                assert hashlib.sha256(data).hexdigest() == plan['reuse']['baseline_manifest'][relative]
                target = root/'baseline'/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream:
                    stream.write(data)
        publish(comparison=report())
        torch.set_grad_enabled(False)
        seed_all(plan['model_seed'])
        model = torch_model(plan)
        assert model.execution_interface.gripper_open_tolerance == 0.
        original_hash = hash_parameters((n,p) for n,p in model.named_parameters() if p.requires_grad)
        assert original_hash == plan['expected_original_trainable_sha256']
        frozen_hash = hash_parameters((n,p) for n,p in model.named_parameters() if not p.requires_grad)
        assert frozen_hash == plan['reuse']['frozen_hash']
        original = {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
        active = 'baseline'
        identities = {'baseline': original_hash, 'final': plan['checkpoint']['parameter_sha256']}
        write(root/'model_identity.json', {'original': original_hash, 'final': identities['final'],
                                         'frozen': frozen_hash, 'new_optimizer_steps': 0})
        for index, (phase, instance, seed) in enumerate(jobs):
            if phase != active:
                if phase == 'final':
                    load_overlay(model, plan['checkpoint']['directory'], identities[phase])
                else:
                    result = model.load_state_dict(original, strict=False)
                    assert not result.unexpected_keys
                    assert hash_parameters((n,p) for n,p in model.named_parameters() if p.requires_grad) == original_hash
                active = phase
            publish(stage=phase, instance=instance, seed=seed, active_parameter_sha256=identities[phase])
            local_plan = {**plan, 'evaluation': {**plan['evaluation'], 'cases': [[instance, seed]]}}
            evaluate(model, local_plan, phase, publish, write_summary=False)
            publish(completed_new_episodes=index+1, comparison=report())
        assert hash_parameters((n,p) for n,p in model.named_parameters() if not p.requires_grad) == frozen_hash
        final_report = report()
        assert final_report['comparison_complete']
        assert final_report['comparison_valid'], 'Initial state pairing failed'
        write(root/'checks.json', {'no_training': True, 'new_optimizer_steps': 0,
                                   'frozen_parameters_unchanged': True,
                                   'all_evaluations_check_parameters_before_and_after': True})
        publish(state='complete', stage='complete', comparison=final_report)
    except BaseException as error:
        failure = {'error_type': type(error).__name__, 'error': str(error),
                   'traceback': traceback.format_exc(), 'new_optimizer_steps': 0}
        write(root/'failure.json', failure)
        publish(state='failed', stage='failed', error_type=type(error).__name__, comparison=report())
        raise
    finally:
        signal.alarm(0)
        # Videos remain on AFS; export bounded JSON evidence after either outcome.
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
            for path in sorted(root.rglob('*.json')):
                archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
        try:
            upload(os.environ['B1K_RESULT_PUT'], buffer.getvalue())
        except Exception as error:
            write(root/'result_upload_error.json', {'type': type(error).__name__})


if __name__ == '__main__':
    main()
