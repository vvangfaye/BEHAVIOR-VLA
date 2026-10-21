"""One finite native episode with the frozen Torch policy. No job submission."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import time

from native_helpers import build_env, stop_process, wait_for_policy, verify_video
from source_preflight import source_preflight


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def validate_native(row, pair, budget, ack):
    task, instance = pair
    if (row['task'], row['instance_id'], row['rollout_id']) != (task, instance, 0):
        raise ValueError('Native result identity mismatch')
    if type(row['steps']) is not int or not 0 < row['steps'] <= budget + 1:
        raise ValueError('Native frame budget mismatch')
    if not (row['terminated'] or row['truncated']):
        raise ValueError('Incomplete native episode')
    if row['truncated'] and row['steps'] != budget + 1:
        raise ValueError('Truncation does not match the native budget')
    q = row['q_score']['final']
    if type(row['success']) is not bool or not math.isfinite(q) or not 0 <= q <= 1:
        raise ValueError('Invalid outcome fields')
    if ack != {'protocol': 'native_full_episode_seed_ack_v1', 'policy_seed': 0,
               'server_seed_acknowledged': True, 'environment_seed': 0}:
        raise ValueError('Missing applied seed acknowledgement')
    # Do not infer native success from q==1; these are separate endpoints.
    return {'task': task, 'instance_id': instance, 'q_score': q, 'success': row['success'],
            'steps': row['steps'], 'terminated': row['terminated'], 'truncated': row['truncated']}


def run(plan_path, method, task, instance):
    plan = load(plan_path)
    pair = [task, instance]
    if method not in plan['protocol']['methods'] or pair not in plan['pairs']:
        raise ValueError('Episode outside the frozen panel')
    bundle = Path(__file__).resolve().parent
    for name, expected in plan['source_sha256'].items():
        if sha(bundle / name) != expected:
            raise ValueError('Evaluation source changed: ' + name)
    root = Path(plan['afs_root']) / method / f'{task}-{instance}'
    root.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    deadline = start + plan['resource_bound']['max_episode_seconds'] - 120
    server = evaluator = None
    result = {'accepted': False, 'method': method, 'pair': pair,
              'protocol_sha256': plan['protocol_sha256'], 'plan_sha256': sha(plan_path)}
    try:
        preflight = source_preflight(plan, full_baseline=False)
        if preflight['task_metadata_sha256'] != plan['task_metadata_sha256']:
            raise ValueError('Task metadata changed')
        if preflight['native_task_budgets'] != plan['native_task_budgets']:
            raise ValueError('Native frame limits changed')
        asset = next(x for x in plan['instance_assets'] if x['pair'] == pair)
        state = Path(plan['runtime']) / 'omnigibson-data/2026-challenge-task-instances' / asset['relative_path']
        if state.stat().st_size != asset['bytes'] or sha(state) != asset['sha256']:
            raise ValueError('Instance state changed')
        weights = None
        if method != 'original1688977':
            training = Path(plan['training_roots'][method]) / 'output'
            receipt = load(training / 'result.json')
            manifest = load(training / 'run.json')
            if (not receipt['complete'] or receipt['completed_steps'] != 1000
                    or manifest['plan_sha256'] != plan['training_plan_sha256']
                    or manifest['method'] != method or manifest['seed'] != 17):
                raise ValueError('Training receipt differs from frozen selection')
            weights = training / 'checkpoint.pt'
            result['weights_sha256'] = sha(weights)
        gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total',
                                           '--format=csv,noheader'], text=True).strip().splitlines()
        if len(gpu_rows) != 1 or not any(name in gpu_rows[0] for name in ('5090', 'N12LP')):
            raise ValueError('Expected one5090 GPU')
        driver = gpu_rows[0].split(',')[2].strip()
        libraries = [p for directory in ('/usr/local/nvidia/lib64', '/usr/lib/x86_64-linux-gnu', '/usr/lib64', '/lib64')
                     for p in Path(directory).glob('libEGL_nvidia.so.' + driver)]
        if not libraries:
            raise ValueError('Matching host graphics driver missing')
        vk, egl = root / 'vk.json', root / 'egl.json'
        for path in (vk, egl):
            write(path, {'file_format_version': '1.0.0', 'ICD': {'library_path': str(libraries[0]),
                    **({'api_version': '1.3.242'} if path == vk else {})}})
        cache = Path('/var/tmp/behavior-1k-runtime') / ('support-' + method + '-' + task + '-' + str(instance))
        cache.mkdir(parents=True, exist_ok=False)
        if shutil.disk_usage(cache).free < 40 * 1024**3 or cache.stat().st_dev == Path(plan['runtime']).stat().st_dev:
            raise ValueError('Need separate worker-local simulator cache with40GiB free')
        for name in ('appdata', 'cache', 'config', 'warp'):
            (cache / name).mkdir()
        params = {**plan, 'port': 8930, 'startup_timeout_seconds': 1200}
        library_path = load(Path(plan['supplemental_library']['root']) / '_READY')['library_path']
        policy_env = build_env(params, cache, vk, egl, library_path, process='policy')
        eval_env = build_env(params, cache, vk, egl, library_path, process='evaluator')
        for env in (policy_env, eval_env):
            env['PYTHONPATH'] = str(bundle) + ':' + env['PYTHONPATH']
            env['PYTHONUNBUFFERED'] = '1'
        policy_env.update(RLINF_ROOT=plan['upstream'], B1K_ADAPTER_ROOT=plan['adapter'])
        policy_env['PYTHONPATH'] = ':'.join([str(bundle), plan['site'], plan['adapter'], plan['upstream'], policy_env['PYTHONPATH']])
        eval_env.update(PAIRED_SEED_PROOF=str(root / 'seed_ack.json'), REPEATABILITY_OUTPUT=str(root))
        token = secrets.token_hex(32)
        runtime = Path(plan['runtime'])
        output = root / 'results'
        server_cmd = [str(runtime / 'envs/b1k-pi05/bin/python'), str(bundle / 'serve_vla_native.py'),
                      '--base-checkpoint', plan['checkpoint'], '--tokenizer', plan['tokenizer'],
                      '--method', 'standard' if method == 'original1688977' else method,
                      '--native-code', plan['code'], '--task', 'b1k/' + task,
                      '--port', '8930', '--server-token', token, '--seed', '0']
        if weights:
            server_cmd += ['--trained-weights', str(weights)]
        eval_cmd = [str(runtime / 'envs/b1k-eval/bin/python'), str(bundle / 'evaluator_entry.py'),
                    '--task-name', task, '--host', '127.0.0.1', '--port', '8930', '--mode', 'public_test',
                    '--prompt-mode', 'whole_task', '--num-rollouts', '1', '--instance-indices', str(instance - 301),
                    '--expected-server-token', token, '--output-dir', str(output), '--write-video', '--headless']
        def interrupted(*_):
            raise InterruptedError('Evaluation worker terminated')
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        with (root / 'policy.log').open('x') as plog, (root / 'eval.log').open('x') as elog:
            server = subprocess.Popen(server_cmd, env=policy_env, cwd=Path(plan['code']) / 'openpi',
                                      stdout=plog, stderr=subprocess.STDOUT, start_new_session=True)
            metadata = wait_for_policy(params, server, token, eval_env, root / 'policy_metadata.json')
            if (metadata.get('full_episode_seed_protocol') != 'native_full_episode_seed_ack_v1'
                    or metadata.get('policy_rng_backend') != 'torch-separated-action-and-structure'):
                raise ValueError('Wrong policy runtime')
            evaluator = subprocess.Popen(eval_cmd, env=eval_env, cwd=Path(plan['code']) / 'BEHAVIOR-1K',
                                         stdout=elog, stderr=subprocess.STDOUT, start_new_session=True)
            evaluator.wait(timeout=max(1, deadline - time.monotonic()))
            if evaluator.returncode != 0 or server.poll() is not None:
                raise RuntimeError('Policy or evaluator failed; preserve partial outputs')
        native_path = output / 'json' / f'{task}_{instance}_0.json'
        if list((output / 'json').glob('*.json')) != [native_path]:
            raise ValueError('Expected exactly one episode')
        record = validate_native(load(native_path), pair, plan['native_task_budgets'][task], load(root / 'seed_ack.json'))
        video = output / 'videos' / f'{task}_{instance}_0.mp4'
        video_proof = verify_video(params, video, eval_env, root)
        result.update(accepted=True, record=record, gpu_rows=gpu_rows, video_proof=video_proof,
                      initial_state=load(root / 'initial_state.json'), action_trace=load(root / 'action_trace.json'),
                      result_sha256=sha(native_path))
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        stop_process(evaluator)
        stop_process(server)
        result['elapsed_seconds'] = time.monotonic() - start
        write(root / 'receipt.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--instance', type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.plan, args.method, args.task, args.instance)))
