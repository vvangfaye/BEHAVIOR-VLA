"""Exercise real worker control flow with isolated files and fake subprocesses."""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import tempfile
import types

ROOT = Path(__file__).resolve().parent


def run_case(role, fault=None):
    with tempfile.TemporaryDirectory(prefix='b1k-heldout-check-') as temporary:
        root = Path(temporary)
        runtime = root / 'runtime'
        runtime.mkdir()

        class TestPath(type(Path())):
            def __new__(cls, *parts):
                value = str(Path(*parts))
                for prefix in ('/var/tmp/behavior-1k-runtime', '/usr/local/nvidia/lib64',
                               '/usr/lib/x86_64-linux-gnu', '/usr/lib64', '/lib64'):
                    if value.startswith(prefix):
                        value = str(root / 'local' / value.lstrip('/'))
                        break
                return super().__new__(cls, value)

            def stat(self, *args, **kwargs):
                result = super().stat(*args, **kwargs)
                if str(self) == str(runtime):
                    values = list(result)
                    values[2] = result.st_dev + 1
                    return os.stat_result(values)
                return result

        lib = TestPath('/usr/local/nvidia/lib64/libEGL_nvidia.so.999.0')
        lib.parent.mkdir(parents=True)
        lib.write_text('test')
        digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        write = lambda p, value: Path(p).write_text(json.dumps(value))
        candidate = root / 'candidate'
        candidate.mkdir()
        (candidate / 'SHA256SUMS.inference').write_text('test inference manifest')
        expected = digest(candidate / 'SHA256SUMS.inference')
        write(candidate / '_READY', {'completed_updates': 1000,
              'manifest_sha256': 'bad' if fault == 'checkpoint' else expected})
        supplement = root / 'supplement'
        supplement.mkdir()
        write(supplement / '_READY', {'library_path': str(root)})
        asset = runtime / 'omnigibson-data/2026-challenge-task-instances/scenes/test.json'
        asset.parent.mkdir(parents=True)
        asset.write_text('{}')
        plan = {'pairs': [['test_task', 73]], 'afs_root': str(root / 'results'),
                'runtime': str(runtime), 'code': str(root / 'code'), 'repo_id': 'test_repo',
                'candidate_checkpoint': str(candidate), 'baseline_checkpoint': str(root / 'baseline'),
                'expected_checkpoint_manifest_sha256': expected, 'baseline_files': [],
                'task_metadata_sha256': 'metadata', 'native_task_budgets': {'test_task': 10},
                'supplemental_library': {'root': str(supplement)}, 'protocol_sha256': 'protocol',
                'instance_assets': [{'pair': ['test_task', 73], 'relative_path': 'scenes/test.json',
                                     'bytes': 2, 'sha256': 'bad' if fault == 'asset' else digest(asset)}]}
        calls, source_checks = [], []

        class Process:
            returncode = 0

            def __init__(self, server):
                self.server = server

            def wait(self, timeout):
                return 0

            def poll(self):
                return None if self.server else 0

        def popen(command, **kwargs):
            calls.append(command)
            server = '--policy.dir' in command
            if not server:
                output = Path(command[command.index('--output-dir') + 1])
                (output / 'json').mkdir(parents=True)
                (output / 'videos').mkdir()
                row = {'task': 'test_task', 'instance_id': 74 if fault == 'instance' else 73,
                       'rollout_id': 0, 'steps': 3 if fault == 'early_success' else (10 if fault == 'budget' else 11),
                       'terminated': fault == 'early_success', 'truncated': fault != 'early_success',
                       'q_score': {'final': 1.0 if fault == 'early_success' else 0.25},
                       'success': fault == 'early_success'}
                write(output / 'json/test_task_73_0.json', row)
                (output / 'videos/test_task_73_0.mp4').write_bytes(b'video')
                write(output.parent / 'seed_ack.json', {'protocol': 'native_full_episode_seed_ack_v1',
                      'policy_seed': 1 if fault == 'seed' else 0,
                      'server_seed_acknowledged': True, 'environment_seed': 0})
            return Process(server)

        def preflight(plan, full_baseline=False):
            source_checks.append(full_baseline)
            return {'task_metadata_sha256': 'bad' if fault == 'metadata' else 'metadata',
                    'native_task_budgets': {'test_task': 10}}

        namespace = {'__file__': str(ROOT / 'frozen/worker.py'), 'Path': TestPath,
            'source_preflight': preflight, 'sha': digest, 'load': lambda p: json.loads(Path(p).read_text()),
            'write': write, 'hashlib': hashlib, 'json': json, 'math': math, 'secrets': secrets,
            'shutil': types.SimpleNamespace(disk_usage=lambda p: types.SimpleNamespace(free=100 * 1024**3)),
            'subprocess': types.SimpleNamespace(Popen=popen, STDOUT=-2,
                check_output=lambda *a, **k: '0, NVIDIA RTX5090,999.0, 32768 MiB'),
            'build_env': lambda *a, **k: {'PYTHONPATH': ''},
            'signal': types.SimpleNamespace(signal=lambda *a: None, SIGTERM=15, SIGINT=2),
            'wait_for_policy': lambda *a: {'full_episode_seed_protocol': 'native_full_episode_seed_ack_v1'},
            'verify_video': lambda *a: {'valid': True}, 'stop_process': lambda p: None}
        tree = ast.parse((ROOT / 'frozen/worker.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate')
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'actual_evaluate', 'exec'), namespace)
        rejected = False
        try:
            result = namespace['evaluate']({'plan': plan, 'pair': ['test_task', 73],
                                           'role': role, 'id': 'episode', 'job_name': 'test'})
            assert result['role'] == role
        except ValueError:
            rejected = True
        assert rejected == (fault not in (None, 'early_success')), (role, fault, rejected)
        assert source_checks == [role == 'baseline']
        if not rejected:
            assert calls[0][calls[0].index('--policy.dir') + 1] == plan[role + '_checkpoint']
            assert calls[1][calls[1].index('--mode') + 1] == 'train'
            assert calls[1][calls[1].index('--instance-indices') + 1] == '73'
            assert '--max-steps' not in calls[1]
        return {'role': role, 'case': fault or 'valid_truncation', 'passed': True}


def main():
    cases = [run_case('baseline'), run_case('candidate'), run_case('candidate', 'early_success')]
    cases += [run_case('candidate', fault) for fault in ('checkpoint', 'asset', 'metadata', 'seed', 'instance', 'budget')]
    result = {'worker_sha256': hashlib.sha256((ROOT / 'frozen/worker.py').read_bytes()).hexdigest(),
              'cases': cases, 'all_passed': True, 'real_gpu_execution': False}
    (ROOT / 'worker_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
