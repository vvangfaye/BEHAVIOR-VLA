"""Exercise the actual submit packer with a fake object store and no cloud calls."""
import ast
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import shutil
import sys
import tarfile
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import zlib


class Store:
    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body):
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):
        return {'Body': io.BytesIO(self.objects[Key])}

    def generate_presigned_url(self, method, Params, ExpiresIn):
        return 'https://unit-test.invalid/' + Params['Key']

    def close(self):
        pass


def main():
    root = Path(sys.argv[1]).absolute()
    variant = sys.argv[2] if len(sys.argv) > 2 else 'balanced_gamma1_ppo'
    spec = importlib.util.spec_from_file_location('packer_under_test', root / 'deployment.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.campaign()
    role = sys.argv[3] if len(sys.argv) > 3 else 'eval'
    case = config['eval_cases'][0] if role == 'eval' else None
    store = Store()
    captured = {}
    def run(args, **kwargs):
        captured['command'] = next(a.split('=', 1)[1] for a in args if a.startswith('--command='))
        return SimpleNamespace(returncode=0)
    def describe(name):
        if name == 'b1k-repeat1688-1019-batch-00':
            return {'resource_pool': {'name': 'computing-cluster-5090-01g'}}
        return {'uid': 'unit-test-only', 'state': 'CREATING', 'scheduling': {'quota_type': 'SPOT'},
                'roles': [{'startup_script': captured['command']}]}
    with tempfile.TemporaryDirectory() as folder:
        temporary = Path(folder)
        for name in ['plan.json', 'verification.json', 'source_plan.json', 'source_training_receipts.json']:
            if (root / name).exists():
                shutil.copy2(root / name, temporary / name)
        (temporary / 'runtime').symlink_to(root / 'runtime', target_is_directory=True)
        with patch.object(module, 'HERE', temporary), patch.object(module, 'aoss_client', lambda: store), patch.object(module, 'status_client', lambda: store), \
             patch.object(module, 'describe', describe), patch.object(module, 'storage_proof', lambda _: {'passed': True}), \
             patch.object(module.subprocess, 'run', side_effect=run):
            if root.name == 'rl_combined_20261102':
                module._submit_unlocked(role, case)
            else:
                module._submit_unlocked(role, case, variant=variant)
    command = captured['command']
    assert len(command) < 64000
    expression = ast.parse(shlex.split(command)[-1])
    encoded = next(node.args[0].value for node in ast.walk(expression)
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'b64decode')
    startup_source = zlib.decompress(base64.b64decode(encoded)).decode()
    invocation = ast.parse(startup_source).body[-1].value
    assert invocation.func.id == 'guarded_run'
    startup = ast.literal_eval(invocation.args[0])
    assert 'bundle_inline' in startup and 'bundle_get' not in startup
    bundle = base64.b64decode(startup['bundle_inline'])
    assert len(bundle) == startup['bundle']['bytes']
    assert hashlib.sha256(bundle).hexdigest() == startup['bundle']['sha256']
    with tarfile.open(fileobj=io.BytesIO(bundle), mode='r:gz') as archive:
        names = archive.getnames()
        assert ('run/parallel_training.py' in names) == (role == 'train')
        assert 'run/driver.py' in names
        if role == 'eval':
            assert 'run/simulator.py' in names
        plan = json.loads(archive.extractfile('run/plan.json').read())
    if role == 'eval':
        assert plan['evaluation']['cases'] == [case]
    assert plan['comparison_variant'] == variant
    assert plan['seed'] == plan['model_seed'] == 0
    result = {'passed': True, 'real_cloud_submissions': 0, 'command_bytes': len(command),
              'bundle_bytes': len(bundle), 'files': len(names), 'variant': variant, 'role': role,
              'single_gpu_bootstrap_inline_contract_satisfied': True, 'fixed_case_preserved': True}
    (root / ('packaging_test_' + role + '_' + variant + '.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
