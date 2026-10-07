"""OSS identity, bounded allocation, submission ambiguity and publication checks."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from nebula import subtask_finetune as job
from test_stage_subtask_replay import inputs
from test_finetune_subtask_collection import checkpoint
import test_subtask_improvement as fixture
from collect_b1k_subtasks import validate_collection


class Client:
    bucket_name = 'fixture'
    def __init__(self, objects):
        self.objects = objects; self.bucket = self; self.reads = []; self.puts = []; self.truncated = False
    def key(self, uri): return job.object_key(uri, 'fixture')
    def head_object(self, key):
        data = self.objects[key]
        return SimpleNamespace(content_length=len(data), etag=hashlib.sha256(data).hexdigest())
    def read(self, key): self.reads.append(key); return self.objects[key]
    def inventory(self, key): return {k:len(v) for k,v in self.objects.items() if k.startswith(key + '/')}
    def unused(self, key, directory=True):
        if key in self.objects or any(k.startswith(key + ('/' if directory else '.')) for k in self.objects):
            raise ValueError('Existing remote output')
    def get_object(self, key, headers):
        data = self.objects[key]; etag = self.head_object(key).etag
        assert headers['If-Match'] == etag
        response = io.BytesIO(data[:-1] if self.truncated else data)
        response.etag = etag; response.resp = SimpleNamespace(response=response)
        return response
    def put_new(self, key, path): self.put_object_from_file(key, str(path), {'x-oss-forbid-overwrite':'true'})
    def put_object_from_file(self, key, path, headers):
        assert headers['x-oss-forbid-overwrite'] == 'true'
        if key in self.objects: raise ValueError('Object already exists')
        self.objects[key] = Path(path).read_bytes(); self.puts.append(key)
    def delete_object(self, key): self.objects.pop(key, None)


@pytest.fixture
def source(inputs, checkpoint, tmp_path, monkeypatch):
    metadata, bucket, replay = inputs
    base, identity = checkpoint
    panel, summary = fixture.inputs()
    collection = fixture.plan_for(panel, summary, checkpoint_identity=identity,
                                   held_out_instances=replay['held_out_instances'])
    root = tmp_path / 'collection'
    for index, trial in enumerate(collection['trials']): fixture.write_trial(root, collection, panel, trial, value=index+1)
    report = validate_collection(root, panel, collection)
    (root / '_SUCCESS').write_text(json.dumps(dict(kind='subtask_success_collection', complete=True, ready_for_export=True, plan_fingerprint=collection['fingerprint'])))
    (root / 'collection_summary.json').write_text(json.dumps(report))
    (root / 'collection_contract.json').write_text(json.dumps(dict(kind='subtask_success_collection', run_id='synthetic-fixture', parent_panel_fingerprint=panel['fingerprint'], plan_fingerprint=collection['fingerprint'], checkpoint_identity=identity)))
    objects = dict(bucket.objects)
    objects.update({'collection/' + p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()})
    objects.update({'model/' + p.relative_to(base).as_posix(): p.read_bytes() for p in base.rglob('*') if p.is_file()})
    objects.update({'runtime/policy.tar.gz': b'fixture archive', 'runtime/cache.tar.gz': b'fixture cache'})
    labels = tmp_path / 'labels.parquet'; labels.write_bytes(b'fixture labels')
    client = Client(objects); monkeypatch.setenv('OSS_ENDPOINT', 'https://fixture.invalid')
    kwargs = dict(panel=panel, collection_plan=collection, collection_source='oss://fixture/collection',
        metadata=metadata, replay_plan=replay, labels=labels, checkpoint='oss://fixture/model',
        base_repo_id='base', policy_archive='oss://fixture/runtime/policy.tar.gz',
        cache_archive='oss://fixture/runtime/cache.tar.gz', output='oss://fixture/new/experiment', run_id='bounded-test')
    return client, kwargs


def test_freezes_completed_collection_exact_model_and_indexed_replay_without_large_reads(source):
    client, kwargs = source; plan = job.freeze(client, **kwargs)
    job.validate_plan(plan)
    assert plan['steps'] == 1000 and plan['devices'] == 4 and plan['batch_size'] == 8
    assert plan['replay_plan']['held_out_instances'] == plan['collection_plan']['held_out_instances']
    assert any(i['path'] == 'replay/data/chunk-000/file-002.parquet' for i in plan['objects'])
    assert plan['checkpoint_bytes'] > 0
    assert not any(k.endswith('.hdf5') or k.endswith('.mp4') or k.endswith('.tar.gz') for k in client.reads)


@pytest.mark.parametrize('damage', ['pending', 'zero_success', 'failed', 'wrong_contract', 'wrong_checkpoint', 'holdout', 'overlap', 'uncovered_params'])
def test_preflight_rejects_invalid_real_input_before_submission(source, damage):
    client, kwargs = source
    if damage == 'pending':
        client.objects['collection/_RUNNING'] = b'running'
    elif damage == 'zero_success':
        value = json.loads(client.objects['collection/_SUCCESS']); value['ready_for_export'] = False
        client.objects['collection/_SUCCESS'] = json.dumps(value).encode()
    elif damage == 'failed': client.objects['collection/_FAILED'] = b'failed'
    elif damage == 'wrong_contract': client.objects['collection/collection_contract.json'] = b'{}'
    elif damage == 'wrong_checkpoint':
        value = json.loads(client.objects['model/_READY']); value['step'] = 0
        client.objects['model/_READY'] = json.dumps(value).encode()
    elif damage == 'holdout': kwargs['collection_plan'] = {**kwargs['collection_plan'], 'held_out_instances': [['task',13]]}
    elif damage == 'overlap': kwargs['output'] = 'oss://fixture/collection/experiment'
    else: client.objects['model/params/uncovered'] = b'not manifested'
    with pytest.raises(ValueError): job.freeze(client, **kwargs)
    assert not client.puts


@pytest.mark.parametrize('values', [dict(devices=3), dict(steps=5001), dict(steps=True), dict(batch_size=9), dict(run_id='../escape')])
def test_bounded_job_limits_cannot_be_expanded(source, values):
    client, kwargs = source
    with pytest.raises(ValueError): job.freeze(client, **{**kwargs, **values})
    assert not client.puts


@pytest.mark.parametrize('damage', ['changed', 'truncated', 'sha256', 'escape'])
def test_streamed_download_cannot_accept_mutated_partial_or_escaping_input(tmp_path, damage):
    client = Client({'input': b'original input'})
    item = job.snapshot(client, 'input', 'collection/input', hashlib.sha256(b'original input').hexdigest())
    if damage == 'changed': client.objects['input'] = b'changed'
    elif damage == 'truncated': client.truncated = True
    elif damage == 'sha256': item['sha256'] = '0'*64
    else: item['path'] = '../escape'
    with pytest.raises(ValueError): job.download(client, item, tmp_path)
    assert not (tmp_path/'collection/input').exists()


def test_real_bundle_has_no_config_secrets_or_unrelated_openloop_changes(source, tmp_path):
    client, kwargs = source; plan = job.freeze(client, **kwargs)
    payload = job.bundle(plan, kwargs['metadata'], kwargs['labels'], tmp_path/'prepared')
    assert not (payload/'nebula/config.sh').exists()
    assert not (payload/'scripts/eval_openloop.py').exists()
    assert (payload/'openpi/scripts/b1k/train_b1k.py').is_file()
    assert job.inventory(payload) == job.read(payload.parent/'bundle_manifest.json')
    assert job.read(payload/'nebula/cluster.json')['worker']['gpu'] == 400
    assert not (payload/'openpi/.venv').exists()


def prepare_submission(source, tmp_path, monkeypatch):
    client, kwargs = source; plan = job.freeze(client, **kwargs)
    payload = job.bundle(plan, kwargs['metadata'], kwargs['labels'], tmp_path/'prepared')
    for key,value in dict(AI_HUB_TOKEN='fixture-private-token', AI_HUB_NAMESPACE='fixture', OSS_ACCESS_ID='fixture-private-id', OSS_ACCESS_KEY='fixture-private-key', OSS_ENDPOINT='https://fixture.invalid').items(): monkeypatch.setenv(key,value)
    return client, payload.parent


def test_submission_redacts_private_cli_output_and_never_repeats_same_attempt(source, tmp_path, monkeypatch):
    client, prepared = prepare_submission(source,tmp_path,monkeypatch); calls=[]
    def fake(command, **kwargs):
        calls.append(command)
        assert '--worker_count=1' in command and '--max_failover_times=0' in command
        return SimpleNamespace(returncode=0,stdout='fixture-private-token fixture-private-key\n{"iid":217999,"task_id":"01atest"}\n')
    monkeypatch.setattr(job.subprocess,'run',fake)
    receipt=job.submit(client,prepared)
    assert receipt['submission_state']=='submitted' and receipt['job_identifiers']==['01atest','217999']
    text=(prepared/'submission.stdout.log').read_text()
    assert 'fixture-private' not in text and '[REDACTED]' in text
    with pytest.raises(ValueError,match='Prior submission'):job.submit(client,prepared)
    assert len(calls)==1


def test_cli_exception_persists_unknown_attempt_for_exact_task_investigation(source,tmp_path,monkeypatch):
    client, prepared = prepare_submission(source,tmp_path,monkeypatch)
    def fail(*args,**kwargs):raise OSError('fixture-private-key')
    monkeypatch.setattr(job.subprocess,'run',fail)
    with pytest.raises(OSError):job.submit(client,prepared)
    receipt=job.read(prepared/'submission_receipt.json')
    assert receipt['submission_state']=='unknown' and receipt['resubmit_allowed'] is False
    assert all('fixture-private' not in data.decode(errors='ignore') for key,data in client.objects.items() if 'submission_' in key)
    with pytest.raises(ValueError):job.submit(client,prepared)


def test_mutated_bundle_rejected_before_remote_claim_or_cli(source,tmp_path,monkeypatch):
    client, prepared = prepare_submission(source,tmp_path,monkeypatch)
    (prepared/'bundle/scripts/train_pi05_b1k.sh').write_text('mutated')
    monkeypatch.setattr(job.subprocess,'run',lambda *a,**k:pytest.fail('CLI called before bundle check'))
    with pytest.raises(ValueError,match='code or inputs changed'):job.submit(client,prepared)
    assert not client.puts


def test_tar_extraction_blocks_escape_without_running_conda(tmp_path,monkeypatch):
    archive=tmp_path/'bad.tar.gz'
    with tarfile.open(archive,'w:gz') as f:
        info=tarfile.TarInfo('../escaped');info.size=3;f.addfile(info,io.BytesIO(b'bad'))
    monkeypatch.setattr(job.subprocess,'run',lambda *a,**k:pytest.fail('unpack before safe extraction'))
    with pytest.raises(tarfile.FilterError):job.restore_policy(archive,tmp_path/'policy')
    assert not (tmp_path/'escaped').exists()


def staged_output(source,tmp_path):
    client,kwargs=source;plan=job.freeze(client,**kwargs);runtime=tmp_path/'worker';runtime.mkdir()
    model=runtime/'inference_checkpoint';model.mkdir();(model/'params').mkdir()
    (model/'params/array').write_bytes(b'new fixture weights')
    manifest=hashlib.sha256(b'new fixture weights').hexdigest()+'  params/array\n'
    (model/'SHA256SUMS.inference').write_text(manifest)
    checksum=hashlib.sha256(manifest.encode()).hexdigest()
    job.write(model/'_READY',dict(manifest_sha256=checksum))
    provenance=[dict(source_kind='collected',task_name='task',task_instance_id=2)]
    job.write(runtime/'dataset/meta/collection_provenance.json',provenance)
    training=dict(dataset_files=job.inventory(runtime/'dataset'),run_id=plan['run_id'])
    training['fingerprint']=job.digest(training)
    job.write(runtime/'training/training_plan.json',training)
    receipt=dict(complete=True,inference_staging_complete=True,training_plan_fingerprint=training['fingerprint'],
        output_checkpoint_manifest_sha256=checksum,training_run_id=plan['run_id'],conditioning='task_and_subtask',
        collection_plan_fingerprint=plan['collection_plan']['fingerprint'],export_provenance_sha256=job.digest(provenance),
        init_checkpoint_manifest_sha256=plan['collection_plan']['checkpoint_identity']['checkpoint_manifest_sha256'])
    job.write(runtime/'training/training_receipt.json',receipt)
    job.write(runtime/'pipeline_completion.json',dict(complete=True,plan_fingerprint=plan['fingerprint'],training_plan_fingerprint=training['fingerprint'],training_receipt=receipt))
    client.objects['new/experiment/_RUNNING']=b'running'
    return client,plan,runtime


def test_completion_marker_is_last_and_only_after_real_staged_bytes_upload(source,tmp_path):
    client,plan,runtime=staged_output(source,tmp_path)
    job.publish(client,plan,runtime)
    assert client.puts[-1]=='new/experiment/_SUCCESS'
    assert 'new/experiment/_RUNNING' not in client.objects
    assert 'new/experiment/inference_checkpoint/params/array' in client.objects
    assert 'new/experiment/dataset/meta/collection_provenance.json' in client.objects


@pytest.mark.parametrize('damage',['unfinished','mutated_weights','wrong_manifest','mutated_dataset','wrong_training_run'])
def test_invalid_training_or_model_cannot_publish_success(source,tmp_path,damage):
    client,plan,runtime=staged_output(source,tmp_path)
    if damage=='unfinished':
        value=job.read(runtime/'pipeline_completion.json');value['complete']=False
        (runtime/'pipeline_completion.json').write_text(json.dumps(value))
    elif damage=='mutated_weights':(runtime/'inference_checkpoint/params/array').write_bytes(b'mutated')
    elif damage=='wrong_manifest':(runtime/'inference_checkpoint/SHA256SUMS.inference').write_text('different')
    elif damage=='mutated_dataset':(runtime/'dataset/meta/collection_provenance.json').write_text('[]')
    else:
        receipt=job.read(runtime/'training/training_receipt.json');receipt['training_run_id']='different'
        (runtime/'training/training_receipt.json').write_text(json.dumps(receipt))
        proof=job.read(runtime/'pipeline_completion.json');proof['training_receipt']=receipt
        (runtime/'pipeline_completion.json').write_text(json.dumps(proof))
    with pytest.raises(ValueError):job.publish(client,plan,runtime)
    assert not client.puts and 'new/experiment/_SUCCESS' not in client.objects


def test_default_training_queue_card_and_receipt_are_frozen_not_inherited(source,tmp_path,monkeypatch):
    client,prepared=prepare_submission(source,tmp_path,monkeypatch)
    plan=job.read(prepared/'bundle/nebula/finetune_job.json')
    assert plan['resources']==dict(queue='demand_specific_na175_h20',gpu_card_type='h20',worker_count=1)
    monkeypatch.setenv('QUEUE','amap_common_aios_nm125_l20');monkeypatch.setenv('GPU_CARD_TYPE','l20')
    commands=[]
    def fake(command,**kw):
        commands.append(command)
        return SimpleNamespace(returncode=0,stdout='{"iid":217999}\n')
    monkeypatch.setattr(job.subprocess,'run',fake)
    receipt=job.submit(client,prepared)
    assert '--queue=demand_specific_na175_h20' in commands[0] and '--gpu_card_type=h20' in commands[0]
    assert receipt['resources']==plan['resources']


@pytest.mark.parametrize('queue,card',[('demand_specific_na175_h20','l20'),('amap_common_aios_nm125_l20','h20'),('bad\nqueue','h20')])
def test_queue_card_disagreement_and_unsafe_queue_cannot_be_prepared(source,queue,card):
    client,kw=source
    with pytest.raises(ValueError):job.freeze(client,**kw,queue=queue,gpu_card_type=card)
    assert not client.puts


def test_explicit_l20_training_alternative_keeps_its_own_frozen_contract(source):
    client,kw=source;plan=job.freeze(client,**kw,queue='amap_common_aios_nm125_l20',gpu_card_type='l20')
    assert plan['resources']['queue']=='amap_common_aios_nm125_l20' and plan['resources']['gpu_card_type']=='l20'


@pytest.mark.parametrize('names,valid', [('NVIDIA H20\n'*4,True),('NVIDIA H20\n'*3,False),('NVIDIA H200\n'*4,False),('NVIDIA L20\n'*4,False)])
def test_h20_runtime_guard_checks_actual_gpu_names_and_count(source,monkeypatch,names,valid):
    client,kw=source;plan=job.freeze(client,**kw)
    monkeypatch.setattr(job.subprocess,'check_output',lambda *a,**k:names)
    if valid:
        proof=job.verify_training_cards(plan)
        assert proof['verified'] and proof['requested_devices']==4 and len(proof['allocated_gpu_names'])==4
    else:
        with pytest.raises(ValueError,match='Allocated GPUs'):job.verify_training_cards(plan)


@pytest.mark.parametrize('overrides,expected_queue,expected_card', [({},'demand_specific_na175_h20','h20'),({'TRAIN_QUEUE':'amap_common_aios_nm125_l20','TRAIN_GPU_CARD_TYPE':'l20'},'amap_common_aios_nm125_l20','l20'),({'QUEUE':'amap_common_aios_nm125_l20','GPU_CARD_TYPE':'l20'},'amap_common_aios_nm125_l20','l20')])
def test_training_launcher_preserves_public_caller_profile_over_private_config(tmp_path,overrides,expected_queue,expected_card):
    import os
    import subprocess
    folder=tmp_path/'launcher';folder.mkdir()
    launcher=folder/'submit_subtask_finetune.sh';launcher.write_text((job.REPO/'nebula/submit_subtask_finetune.sh').read_text())
    # Deliberately a fixture config, never the user's actual credential file.
    (folder/'config.sh').write_text('QUEUE=private_config_queue\nGPU_CARD_TYPE=private_config_card\n')
    (folder/'subtask_finetune.py').write_text('import os,json\nprint(json.dumps({k:os.environ[k] for k in ("TRAIN_QUEUE","TRAIN_GPU_CARD_TYPE")}))\n')
    env=dict(os.environ)
    for k in ('QUEUE','GPU_CARD_TYPE','TRAIN_QUEUE','TRAIN_GPU_CARD_TYPE'):env.pop(k,None)
    env.update(overrides);env['NEBULA_SUBMIT_PYTHON']=sys.executable
    p=subprocess.run(['bash',str(launcher),'prepare'],env=env,capture_output=True,text=True,check=True)
    assert json.loads(p.stdout)=={'TRAIN_QUEUE':expected_queue,'TRAIN_GPU_CARD_TYPE':expected_card}
