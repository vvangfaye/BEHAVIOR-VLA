"""SCO/AOSS adapter for the isolated checkpoint-2M, 1000-episode video run."""
from __future__ import annotations

import ast
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import inspect
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import time
import zlib

from video_eval_queue import TERMINAL_STATES, write_json

BUCKET = '260010165'
POOL = 'computing-cluster-5090-01g'
CLI = os.environ.get('BEHAVIOR_SCO_CLI', str(Path.home() / '.local/bin/sco'))
TEMPLATE = 'pt-y91th8k9'
TEMPLATE_UID = 'c14d0892eaa148d381c7f4424684c9d3'


def fetch_bootstrap(url, expected_sha256):
    """Fetch only through an explicit direct opener; retry read-only transport."""
    import hashlib
    import os
    import time
    import urllib.error
    import urllib.request
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        os.environ.pop(key, None)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    # The frozen worker's urllib readback must use the same verified route.
    urllib.request.install_opener(opener)
    for attempt in range(4):
        try:
            with opener.open(url, timeout=30) as response:
                data = response.read(32 * 1024**2 + 1)
            if len(data) > 32 * 1024**2 or hashlib.sha256(data).hexdigest() != expected_sha256:
                raise ValueError('Bootstrap content differs from its frozen SHA256')
            return data
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 3:
                raise RuntimeError('Bootstrap direct GET failed after four bounded attempts') from None
            time.sleep(min(2**attempt, 4))


def install_explicit_observer_launcher(namespace):
    """Keep the observer active when environment activation clears PYTHONPATH."""
    if '_environment' not in namespace:
        return
    import json
    from pathlib import Path
    import shlex
    original = namespace['_environment']

    def observed_environment(payload, source, root, cache, hook_dir):
        env, gpu_rows = original(payload, source, root, cache, hook_dir)
        actual = Path(payload['runtime']) / 'envs/b1k-eval'
        overlay = cache / 'observed_eval_env'
        (overlay / 'bin').mkdir(parents=True, exist_ok=False)
        for name in ('lib', 'share', 'etc', 'include'):
            if (actual / name).exists():
                (overlay / name).symlink_to(actual / name, target_is_directory=True)
        logs = root / 'logs'
        logs.mkdir(exist_ok=True)
        launcher = logs / 'explicit_progress_launcher.py'
        launcher.write_text('import runpy,sys\nfrom pathlib import Path\n'
            'target=sys.argv[1]\nsys.argv=sys.argv[1:]\nsys.path[0]=str(Path(target).parent)\n'
            'hook=Path(' + repr(str(hook_dir / 'sitecustomize.py')) + ')\n'
            'exec(compile(hook.read_bytes(),str(hook),"exec"),{"__name__":"explicit_progress_hook"})\n'
            'runpy.run_path(target,run_name="__main__")\n')
        python = overlay / 'bin/python'
        python.write_text('#!/bin/bash\nset -euo pipefail\n'
            'if [[ "${1:-}" == */eval_b1k_queue.py ]]; then\n exec '
            + shlex.quote(str(actual / 'bin/python')) + ' ' + shlex.quote(str(launcher)) + ' "$@"\n'
            'else\n exec ' + shlex.quote(str(actual / 'bin/python')) + ' "$@"\nfi\n')
        python.chmod(0o755)
        (logs / 'explicit_progress_python.sh').write_bytes(python.read_bytes())
        (logs / 'explicit_progress_identity.json').write_text(json.dumps(
            {'observer_only': True, 'actual_interpreter': str(actual / 'bin/python'),
             'original_evaluator_script': str(Path(payload['code']) / 'scripts/eval_b1k_queue.py'),
             'hook': str(hook_dir / 'sitecustomize.py'),
             'no_action_observation_or_episode_limit_changes': True}, indent=2) + '\n')
        env['EVAL_ENV'] = str(overlay)
        return env, gpu_rows

    namespace['_environment'] = observed_environment


def preserve_approved_empty_startup(payload):
    """Preserve a proved pre-rollout failure only with explicit user approval."""
    import hashlib
    import json
    import os
    from pathlib import Path
    recovery = payload.get('approved_empty_startup_recovery')
    if recovery is None:
        return
    assert recovery['user_authorization'] == '允许仅补交这一条'
    assert recovery['old_writer_stopped'] is True and recovery['old_platform_state'] == 'FAILED'
    assert recovery['model_rollouts_executed'] == 0
    root = Path(payload['assignment_root'])
    assert root.name == '0021' and payload['task'] == 'putting_away_Halloween_decorations'
    assert payload['instance_id'] == 302 and payload['checkpoint_step'] == 2000000
    assert sorted(p.name for p in root.iterdir()) == ['payload.json']
    original = (root / 'payload.json').read_bytes()
    assert hashlib.sha256(original).hexdigest() == recovery['old_payload_sha256']
    saved = json.loads(original)
    for key in ('run_id', 'task', 'instance_id', 'checkpoint_step',
                'source_manifest_sha256', 'checkpoint_ready_sha256'):
        assert saved[key] == payload[key]
    history = root.with_name('0021.startup-g000')
    assert not history.exists()
    os.rename(root, history)
    (history / 'approved_recovery.json').write_text(json.dumps(recovery, indent=2) + '\n')


def build_startup(url, payload_sha256, source):
    encoded_source = base64.b64encode(zlib.compress(source)).decode()
    return (inspect.getsource(fetch_bootstrap) + '\n' + inspect.getsource(install_explicit_observer_launcher) + '\n'
        + inspect.getsource(preserve_approved_empty_startup) + '\n'
        + 'import base64,hashlib,json,zlib\n'
        + 'b=fetch_bootstrap(' + repr(url) + ',' + repr(payload_sha256) + ')\n'
        + 's=zlib.decompress(base64.b64decode(' + repr(encoded_source) + '))\n'
        + 'assert hashlib.sha256(s).hexdigest()==' + repr(hashlib.sha256(source).hexdigest()) + '\n'
        + 'ns={"__name__":"video_eval_worker"}\n'
        + 'exec(compile(s,"video_eval_worker","exec"),ns)\n'
        + 'ns["_EXECUTED_WORKER_SOURCE"]=s\n'
        + 'install_explicit_observer_launcher(ns)\n'
        + 'p=json.loads(b)\npreserve_approved_empty_startup(p)\nns["run"](p)\n')


def aoss_client():
    import boto3
    from botocore.config import Config
    text = (Path(os.environ.get('SCO_CONFIG', str(Path.home() / '.config/sco'))) / 'profiles/default.toml').read_text()
    values = {k: ast.literal_eval(v) for k, v in re.findall(
        r'^(access_key_id|access_key_secret)\s*=\s*([\"\'].*[\"\'])\s*$', text, re.M)}
    return boto3.client('s3', endpoint_url='https://aoss.cn-sh-01b.sensecoreapi-oss.cn',
        region_name='cn-sh-01b', aws_access_key_id=values['access_key_id'],
        aws_secret_access_key=values['access_key_secret'], config=Config(
            signature_version='s3v4', s3={'addressing_style': 'path'}, proxies={},
            connect_timeout=8, read_timeout=30, retries={'total_max_attempts': 2}))


def environment():
    return dict(os.environ, SCO_CONFIG=os.environ.get('SCO_CONFIG', str(Path.home() / '.config/sco')))


def cli_json(args, timeout=40):
    result = subprocess.run([CLI] + args, env=environment(), capture_output=True,
                            text=True, timeout=timeout)
    if result.returncode:
        # Never propagate raw CLI output containing the private startup command.
        raise RuntimeError('SCO command failed with exit ' + str(result.returncode))
    return json.loads(result.stdout)


class Adapter:
    def __init__(self, root, plan):
        self.root = Path(root)
        self.plan = plan
        self.ready = json.loads((self.root / 'preparation_ready.json').read_text())
        assert self.ready['state'] == 'ready' and self.ready['checkpoint_step'] == 2000000
        self.client = aoss_client()
        self.prefix = plan['object_prefix'].rstrip('/')
        self.source = (self.root / 'frozen/video_eval_worker.py').read_bytes()
        self.driver_sha = hashlib.sha256(self.source).hexdigest()
        assert self.driver_sha == self.ready['worker_sha256']
        batch = self.root / 'frozen/video_eval_batch_worker.py'
        self.batch_source_versions = {}
        self.max_batch_episodes = 5
        policy_path = self.root / 'batch_policy.json'
        if policy_path.exists():
            policy = json.loads(policy_path.read_text())
            from video_eval_finalize import batch_version
            for digest in policy.get('worker_versions', {}):
                version = batch_version(policy, dict(batch_worker_sha256=digest))
                path = self.root / version['source_file']
                assert not path.is_symlink()
                raw = self.source + b'\n' + path.read_bytes()
                assert hashlib.sha256(raw).hexdigest() == digest
                self.batch_source_versions[digest] = raw
            if policy.get('worker_versions'):
                version = batch_version(policy, dict(batch_worker_sha256=policy['batch_worker_sha256']))
                batch = self.root / version['source_file']
                self.max_batch_episodes = version['max_episodes']
        if batch.exists():
            self.batch_source = self.source + b'\n' + batch.read_bytes()
            self.batch_worker_sha256 = hashlib.sha256(self.batch_source).hexdigest()
            self.batch_source_versions[self.batch_worker_sha256] = self.batch_source
        for field in ('source_manifest_sha256', 'checkpoint_ready_sha256', 'worker_sha256'):
            assert self.plan['immutable_inputs'][field] == self.ready[field], field
        (self.root / 'artifacts').mkdir(exist_ok=True)
        (self.root / 'launches').mkdir(exist_ok=True)

    def _describe_live(self, name):
        physical_name = name
        match = re.fullmatch(re.escape(self.plan['run_id']) + r'-e([0-9]{4})', name)
        if match:
            path = self.root / 'launches' / (match.group(1) + '.json')
            if path.exists():
                launch = json.loads(path.read_text())
                assert launch['name'] == name
                physical_name = launch.get('physical_name', name)
            else:
                saved = self.root / 'submissions' / (match.group(1) + '.json')
                if saved.exists():
                    record = json.loads(saved.read_text())
                    assert record['name'] == name
                    physical_name = record.get('physical_name', name)
        job = cli_json(['acp', 'jobs', 'describe', physical_name, '--workspace-name=share-space', '-o', 'json'])
        assert job['ownership']['user_name'] == BUCKET
        assert job['resource_pool']['name'] == POOL
        assert job['name'] == physical_name
        return dict(uid=job['uid'], name=name, state=job['state'], query_complete=True)

    def _receipt_live(self, record):
        try:
            response = self.client.get_object(Bucket=BUCKET, Key=self.keys(record)['receipt'])
        except self.client.exceptions.ClientError as exc:
            if exc.response.get('Error', {}).get('Code') in ('NoSuchKey', '404'):
                return None
            raise
        with response['Body'] as stream:
            raw = stream.read(8 * 1024**2 + 1)
        assert len(raw) <= 8 * 1024**2
        return raw

    def auxiliary_gpu_step(self, allow_submit=False):
        from video_eval_image_validation_controller import step
        return step(self, allow_submit=allow_submit)

    def begin_round(self, records):
        """Refresh all required job states/receipts with at most six reads at once."""
        self._description_cache = {}
        self._receipt_cache = {}
        with ThreadPoolExecutor(max_workers=6) as executor:
            pending = {}
            for record in records:
                if record.get('state') not in TERMINAL_STATES:
                    future = executor.submit(self._describe_live, record['name'])
                    pending[future] = (self._description_cache, record['name'])
                if record.get('inspection', {}).get('verified') is not True:
                    future = executor.submit(self._receipt_live, record)
                    pending[future] = (self._receipt_cache, record['index'])
            for future in as_completed(pending):
                cache, key = pending[future]
                try:
                    cache[key] = future.result()
                except Exception as exc:
                    # Preserve the failure for Controller's fail-closed observation.
                    cache[key] = exc

    def describe(self, name):
        cache = getattr(self, '_description_cache', {})
        if name not in cache:
            return self._describe_live(name)
        reply = cache[name]
        if isinstance(reply, Exception):
            raise reply
        return reply

    def keys(self, record):
        base = self.prefix + '/episodes/' + str(record['index']).zfill(4)
        if record.get('isolated_generation'):
            base += '/generations/g%03d' % record['infrastructure_generation']
        generation = record.get('infrastructure_generation', 0)
        private_name = 'startup.private.json' if not generation else 'startup-g%03d.private.json' % generation
        return {'archive': base + '/episode.tar.gz', 'receipt': base + '/receipt.json',
                'payload': base + '/' + private_name}

    def signed(self, op, key):
        return self.client.generate_presigned_url(op,
            Params={'Bucket': BUCKET, 'Key': key}, ExpiresIn=604800)

    def prepare_recovery(self, record):
        """Prove exact writer termination, then classify originals with a 0GPU job."""
        if record.get('inspection', {}).get('verified'):
            return None
        generation = int(record.get('infrastructure_generation', 0))
        directory = self.root / 'recovery_checks' / ('%04d' % record['index'])
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / ('g%03d.json' % generation)
        launch = json.loads((self.root / 'launches' / ('%04d.json' % record['index'])).read_text())
        physical = launch.get('physical_name', record['name'])
        job = cli_json(['acp', 'jobs', 'describe', physical, '--workspace-name=share-space', '-o', 'json'])
        assert job['uid'] == record['uid'] and job['ownership']['user_name'] == BUCKET
        assert job['resource_pool']['name'] == POOL
        assert job['state'] in TERMINAL_STATES - {'SUCCEEDED'}
        if not path.exists():
            workers = subprocess.run([CLI, 'acp', 'jobs', 'get-workers', physical,
                '--workspace-name=share-space'], env=environment(), capture_output=True, text=True, timeout=30)
            assert workers.returncode == 0 and ('pt-' + record['uid'] + '-worker-0') in workers.stdout
            phases = [v for v in re.findall(r'\|\s*([A-Za-z]+)\s*\|[ \t]*\n', workers.stdout) if v != 'PHASE']
            assert phases and all(v in ('Deleted', 'Failed', 'Succeeded') for v in phases)
            name = self.plan['run_id'] + '-probe-e%04d-g%03d' % (record['index'], generation)
            key = self.prefix + '/recovery_checks/%04d/g%03d.json' % (record['index'], generation)
            assignment = record.get('assignment_root', self.plan['afs_output_root'] + '/episodes/%04d' % record['index'])
            payload = dict(old_uid=record['uid'], old_name=physical, old_state=job['state'],
                old_writer_stopped=True, assignment_root=assignment, afs_output_root=self.plan['afs_output_root'],
                task=record['task'], instance_id=record['instance_id'], checkpoint_step=2000000,
                source_manifest_sha256=self.ready['source_manifest_sha256'],
                checkpoint_ready_sha256=self.ready['checkpoint_ready_sha256'], report_put=self.signed('put_object', key))
            body = json.dumps(payload, sort_keys=True).encode()
            payload_key = key + '.private'
            self.client.put_object(Bucket=BUCKET, Key=payload_key, Body=body)
            source = Path(__file__).with_name('video_eval_recovery_probe.py').read_bytes()
            startup = build_startup(self.signed('get_object', payload_key), hashlib.sha256(body).hexdigest(), source)
            probe = dict(name=name, state='UNKNOWN', old_uid=record['uid'], key=key,
                gpu_count=0, worker_phases=phases, old_writer_stopped=True,
                source_sha256=hashlib.sha256(source).hexdigest(), created_at=time.time())
            write_json(path, probe, exclusive=True)
            try:
                result = subprocess.run([CLI, 'acp', 'jobs', 'create', '--workspace-name=share-space',
                    '--aec2-name=share-cluster', '--name=' + name, '--job-name=' + name,
                    '--training-framework=pytorch',
                    '--container-image-url=registry.cn-sh-01g.sensecore.cn/lepton-trainingjob/ubuntu22.04-pytorch2.1.0-py310-cuda12.2:0630',
                    '--worker-spec=n6ls.iu.i40.8c32g', '--worker-nodes=1', '--quota-type=spot',
                    '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs',
                    '--command=timeout --signal=TERM --kill-after=15 1800 python3 -c ' + shlex.quote(startup)],
                    env=environment(), capture_output=True, text=True, timeout=90)
                probe['create_exit_code'] = result.returncode
            except subprocess.TimeoutExpired:
                probe['create_timeout'] = True
            write_json(path, probe)
        probe = json.loads(path.read_text())
        assert probe['old_uid'] == record['uid']
        check = cli_json(['acp', 'jobs', 'describe', probe['name'], '--workspace-name=share-space', '-o', 'json'])
        assert check['ownership']['user_name'] == BUCKET and check['resource_pool']['name'] == 'share-cluster'
        if probe.get('uid'):
            assert probe['uid'] == check['uid']
        probe.update(uid=check['uid'], state=check['state'])
        write_json(path, probe)
        if check['state'] != 'SUCCEEDED':
            return dict(eligible=False, reason='recovery_probe_' + check['state'].lower(), waiting=check['state'] not in TERMINAL_STATES)
        with self.client.get_object(Bucket=BUCKET, Key=probe['key'])['Body'] as stream:
            proof = json.loads(stream.read(16 * 1024**2))
        assert proof['old_uid'] == record['uid'] and proof['old_name'] == physical
        assert proof['old_writer_stopped'] is True and proof['gpu_count'] == 0
        assert proof['task'] == record['task'] and proof['instance_id'] == record['instance_id']
        proof.update(probe_uid=check['uid'], probe_state='SUCCEEDED', worker_phases=probe['worker_phases'])
        write_json(directory / ('g%03d-proof.json' % generation), proof)
        return proof

    def create(self, record):
        assert not (self.root / 'STOP').exists()
        assert record['checkpoint_step'] == 2000000
        template = cli_json(['acp', 'jobs', 'describe', TEMPLATE, '--workspace-name=share-space', '-o', 'json'])
        assert template['uid'] == TEMPLATE_UID and template['resource_pool']['name'] == POOL
        assert template['ownership']['user_name'] == BUCKET
        keys = self.keys(record)
        p = dict(checkpoint_step=2000000, checkpoint=self.ready['checkpoint'], code=self.ready['code'],
            runtime=self.ready['runtime'], source_manifest_sha256=self.ready['source_manifest_sha256'],
            checkpoint_ready_sha256=self.ready['checkpoint_ready_sha256'],
            task=record['task'], instance_id=record['instance_id'],
            run_id=record['name'], execution_name=record['name'], pilot=record['index'] == 0,
            assignment_root=record.get('assignment_root', self.plan['afs_output_root'] + '/episodes/' + str(record['index']).zfill(4)),
            artifact_puts={k: self.signed('put_object', keys[k]) for k in ('archive', 'receipt')},
            artifact_gets={'archive': self.signed('get_object', keys['archive'])},
            init_timeout_seconds=1800, stall_timeout_seconds=900, job_timeout_seconds=345600)
        if record.get('recovery_chain'):
            p['recovery_chain'] = record['recovery_chain']
            p['previous_model_attempts'] = record.get('previous_model_attempts', 0)
        if record.get('approved_empty_startup_recovery'):
            p['approved_empty_startup_recovery'] = record['approved_empty_startup_recovery']
        payload = json.dumps(p, sort_keys=True).encode()
        self.client.put_object(Bucket=BUCKET, Key=keys['payload'], Body=payload,
                               ContentType='application/json')
        # A private object avoids placing every upload capability in the visible command.
        url = self.signed('get_object', keys['payload'])
        startup = build_startup(url, hashlib.sha256(payload).hexdigest(), self.source)
        physical_name = record.get('physical_name', record['name'])
        launch = dict(name=record['name'], index=record['index'], task=record['task'],
            physical_name=physical_name, infrastructure_generation=record.get('infrastructure_generation', 0),
            instance_id=record['instance_id'], checkpoint_step=2000000,
            worker_sha256=self.driver_sha, source_manifest_sha256=self.ready['source_manifest_sha256'],
            payload_sha256=hashlib.sha256(payload).hexdigest(), keys=keys,
            bootstrap_sha256=hashlib.sha256(startup.encode()).hexdigest(), observer_launcher='explicit-hook-v1',
            quota_type='spot', gpu_count=1, created_at=time.time(),
            assignment_root=p['assignment_root'], recovery_chain=record.get('recovery_chain', []))
        write_json(self.root / 'launches' / f"{record['index']:04d}.json", launch, exclusive=True)
        command = 'timeout --signal=TERM --kill-after=30 345600 python3 -c ' + shlex.quote(startup)
        result = subprocess.run([CLI, 'acp', 'jobs', 'copy', '--workspace-name=share-space',
            '--copy-job-name=' + TEMPLATE, '--name=' + physical_name, '--job-name=' + physical_name,
            '--worker-spec=n12lp.nn.i10a.1', '--worker-nodes=1', '--quota-type=spot',
            '--env=NVIDIA_DRIVER_CAPABILITIES:all',
            '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs',
            '--command=' + command], env=environment(), capture_output=True, text=True, timeout=90)
        # Any error remains an intent and is reconciled by exact name. Never repeat copy.
        if result.returncode:
            raise RuntimeError('SCO copy outcome requires exact-name reconciliation')
        return self.describe(record['name'])

    def create_group(self, records):
        assert 2 <= len(records) <= getattr(self, 'max_batch_episodes', 5) and not (self.root / 'STOP').exists()
        name = records[0]['batch_id']
        assert all(r['batch_id'] == name and r['physical_name'] == name and r['task'] == records[0]['task'] for r in records)
        template = cli_json(['acp', 'jobs', 'describe', TEMPLATE, '--workspace-name=share-space', '-o', 'json'])
        assert template['uid'] == TEMPLATE_UID and template['resource_pool']['name'] == POOL
        assert template['ownership']['user_name'] == BUCKET
        children = []
        for record in records:
            keys = self.keys(record)
            children.append(dict(checkpoint_step=2000000, checkpoint=self.ready['checkpoint'], code=self.ready['code'],
                runtime=self.ready['runtime'], source_manifest_sha256=self.ready['source_manifest_sha256'],
                checkpoint_ready_sha256=self.ready['checkpoint_ready_sha256'], task=record['task'],
                instance_id=record['instance_id'], run_id=record['name'], execution_name=record['name'], pilot=False,
                assignment_root=self.plan['afs_output_root']+'/episodes/%04d' % record['index'],
                artifact_puts={k:self.signed('put_object',keys[k]) for k in ('archive','receipt')},
                artifact_gets={'archive':self.signed('get_object',keys['archive'])}))
        p=dict(batch_name=name,assignment_root=self.plan['afs_output_root']+'/batches/'+name,
               batch_worker_sha256=self.batch_worker_sha256,episodes=children)
        body=json.dumps(p,sort_keys=True).encode(); key=self.prefix+'/batches/'+name+'/startup.private.json'
        self.client.put_object(Bucket=BUCKET,Key=key,Body=body,ContentType='application/json')
        startup=build_startup(self.signed('get_object',key),hashlib.sha256(body).hexdigest(),self.batch_source)
        for record,child in zip(records,children):
            launch=dict(name=record['name'],index=record['index'],task=record['task'],physical_name=name,
                infrastructure_generation=0,instance_id=record['instance_id'],checkpoint_step=2000000,
                worker_sha256=self.driver_sha,source_manifest_sha256=self.ready['source_manifest_sha256'],
                payload_sha256=hashlib.sha256(body).hexdigest(),keys=self.keys(record),
                bootstrap_sha256=hashlib.sha256(startup.encode()).hexdigest(),observer_launcher='explicit-batch-hook-v1',
                quota_type='spot',gpu_count=1,created_at=time.time(),assignment_root=child['assignment_root'],
                batch_id=name,batch_indices=[r['index'] for r in records],batch_worker_sha256=self.batch_worker_sha256,
                batch_root=p['assignment_root'],recovery_chain=[])
            write_json(self.root/'launches'/('%04d.json' % record['index']),launch,exclusive=True)
        command='timeout --signal=TERM --kill-after=30 345600 python3 -c '+shlex.quote(startup)
        result=subprocess.run([CLI,'acp','jobs','copy','--workspace-name=share-space',
            '--copy-job-name='+TEMPLATE,'--name='+name,'--job-name='+name,
            '--worker-spec=n12lp.nn.i10a.1','--worker-nodes=1','--quota-type=spot',
            '--env=NVIDIA_DRIVER_CAPABILITIES:all',
            '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs','--command='+command],
            env=environment(),capture_output=True,text=True,timeout=90)
        if result.returncode:
            raise RuntimeError('Batch create outcome unknown; reconcile exact name')
        job=cli_json(['acp','jobs','describe',name,'--workspace-name=share-space','-o','json'])
        assert job['name']==name and job['ownership']['user_name']==BUCKET and job['resource_pool']['name']==POOL
        return dict(uid=job['uid'],state=job['state'],query_complete=True)

    def inspect(self, record):
        keys = self.keys(record)
        cache = getattr(self, '_receipt_cache', {})
        raw = cache[record['index']] if record['index'] in cache else self._receipt_live(record)
        if isinstance(raw, Exception):
            raise raw
        if raw is None:
            return None
        report = json.loads(raw)
        for key, value in {'checkpoint_step': 2000000, 'task': record['task'],
            'instance_id': record['instance_id'], 'run_id': record['name'],
            'execution_name': record['name'],
            'source_manifest_sha256': self.ready['source_manifest_sha256'],
            'checkpoint_ready_sha256': self.ready['checkpoint_ready_sha256']}.items():
            assert report[key] == value, key
        artifact_path = self.root / 'artifacts' / f"{record['index']:04d}.json"
        write_json(artifact_path, report)
        if report.get('state') != 'complete':
            return {'verified': False, 'reason': report.get('state', 'incomplete')}
        assert report['verified'] and report['full_video_decode_verified']
        assert report['archive_readback_verified']
        assert report['video_validation']['frames_equal_simulator_steps']
        head = self.client.head_object(Bucket=BUCKET, Key=keys['archive'])
        assert head['ContentLength'] == report['archive']['bytes']
        assert type(report['model_success']) is bool and 0 <= report['q_score'] <= 1
        recovered = {}
        if report.get('artifact_observer_recovery'):
            assert record['index'] == 0 and report['old_gpu_uid'] == record['uid']
            assert report['original_model_execution_complete'] is True
            assert report['failure_limited_to_post_rollout_observer'] is True
            assert report['no_model_or_simulator_rerun'] is True
            audit = json.loads((self.root / 'pilot_artifact_audit_submission.json').read_text())
            assert report['audit_job_name'] == audit['name'] and report['audit_job_uid'] == audit['uid']
            job = cli_json(['acp', 'jobs', 'describe', audit['name'], '--workspace-name=share-space', '-o', 'json'])
            assert job['uid'] == audit['uid'] and job['ownership']['user_name'] == BUCKET
            assert job['resource_pool']['name'] == 'share-cluster' and job['state'] == 'SUCCEEDED'
            recovered = dict(original_model_execution_complete=True, no_model_rerun=True,
                artifact_recovery_audit_uid=audit['uid'], artifact_recovery_audit_state='SUCCEEDED')
        if record['index'] == 0:
            self.verify_pilot_archive(record, report, keys['archive'])
        if record.get('batch_id'):
            launch=json.loads((self.root/'launches'/('%04d.json' % record['index'])).read_text())
            context=report.get('batch_context',{})
            assert context.get('batch_name')==record['batch_id']==launch['physical_name']
            assert context.get('batch_worker_sha256')==record['batch_worker_sha256']==launch['batch_worker_sha256']
            assert record['batch_worker_sha256'] in self.batch_source_versions
            assert context.get('root')==launch['batch_root']
            reuse=report.get('batch_reuse',{})
            assert reuse.get('single_model_server') is True and reuse.get('single_evaluator_process') is True
            assert record['instance_id'] in reuse.get('reset_instance_ids',[])
            from video_eval_finalize import validate_batch_provenance
            policy=json.loads((self.root/'batch_policy.json').read_text())
            validate_batch_provenance(self.plan, record, report, launch, policy)
            if record['index'] in policy['pilot_indices']:
                self.verify_pilot_archive(record, report, keys['archive'])
            recovered['batch_completed_original']=True
        return dict(verified=True, model_success=report['model_success'], q_score=report['q_score'], **recovered,
                    artifact_manifest_path=str(artifact_path))

    def verify_pilot_archive(self, record, report, key):
        dest = (self.root / 'batch_pilot_artifacts' / ('%04d' % record['index'])
                if record.get('batch_id') else self.root / 'pilot_artifacts')
        dest.mkdir(parents=True, exist_ok=True)
        archive = dest / 'episode.tar.gz'
        if not archive.exists():
            temporary = archive.with_suffix('.partial')
            self.client.download_file(BUCKET, key, str(temporary))
            temporary.replace(archive)
        sha = hashlib.sha256()
        with archive.open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024**2), b''):
                sha.update(block)
        assert sha.hexdigest() == report['archive']['sha256']
        expected = {'episode/' + r['path']: r for r in report['files']}
        seen = set()
        with tarfile.open(archive, 'r:gz') as tf:
            for item in tf:
                path = Path(item.name)
                assert not path.is_absolute() and '..' not in path.parts
                if item.isdir():
                    continue
                assert item.isfile() and item.name not in seen
                seen.add(item.name)
                contents = tf.extractfile(item).read()
                if item.name in expected:
                    assert len(contents) == expected[item.name]['bytes']
                    assert hashlib.sha256(contents).hexdigest() == expected[item.name]['sha256']
                else:
                    assert path.name in ('ARTIFACT_MANIFEST.json', 'RESULT_MANIFEST.json')
                target = dest / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(contents)
        assert set(expected) <= seen
        # Full decoding already ran independently from the simulator before upload.
        # A second local decode is required for the pilot before the controller opens admission.
        videos = list(dest.rglob('*.mp4'))
        assert len(videos) == 1
        decoder = self.plan['pilot_python']
        script = ('import cv2,json,sys; c=cv2.VideoCapture(sys.argv[1]);'
                  'assert c.isOpened();n=0\n'
                  'while True:\n ok,f=c.read()\n if not ok: break\n assert f.size>0\n n+=1\n'
                  'assert n==int(sys.argv[2]);print(json.dumps({"frames":n}))')
        result = subprocess.run([decoder, '-c', script, str(videos[0]), str(report['steps'])],
                                capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, 'Independent pilot decode failed'
        receipt_path = dest / 'accepted.json' if record.get('batch_id') else self.root / 'pilot_accepted.json'
        write_json(receipt_path, dict(verified=True, index=record['index'],
            checkpoint_step=2000000, source_manifest_sha256=self.ready['source_manifest_sha256'],
            archive_sha256=sha.hexdigest(), full_readback=True, full_decode=json.loads(result.stdout),
            accepted_at=time.time(), model_success=report['model_success'], q_score=report['q_score'],
            canonical_result_kept_regardless_of_score=True))


def factory(root, plan):
    return Adapter(root, plan)
