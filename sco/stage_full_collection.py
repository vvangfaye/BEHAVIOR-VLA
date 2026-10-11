"""Create an immutable descendant source, test it, then audit the full requested scope."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import traceback


def run(payload, overlay):
    code, parent, output = (Path(payload[k]) for k in ('code_root','parent_code_root','staging_root'))
    output.mkdir(parents=True, exist_ok=False)
    status = dict(state='staging', started_at=time.time(), code_root=str(code), output_root=str(output))
    def publish(**updates):
        status.update(updates, updated_at=time.time())
        (output/'status.json').write_text(json.dumps(status, indent=2)+'\n')
        print('STATUS_JSON='+json.dumps(status), flush=True)
    try:
        source = (parent/'_CODE_SNAPSHOT_READY').read_bytes()
        assert hashlib.sha256(source).hexdigest() == payload['parent_source_manifest_sha256']
        marker = json.loads(source)
        assert not code.exists()
        shutil.copytree(parent, code, symlinks=True, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        assert hashlib.sha256(overlay).hexdigest() == payload['overlay_sha256']
        with tarfile.open(fileobj=io.BytesIO(overlay),mode='r:gz') as archive:
            for member in archive.getmembers():
                name = Path(member.name)
                assert member.isfile() and not name.is_absolute() and '..' not in name.parts
                assert member.name in payload['overlay_files_sha256']
                target = code/name
                assert target.resolve().is_relative_to(code.resolve())
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(archive.extractfile(member).read());target.chmod(member.mode & 0o777)
        marker['overlay_files_sha256'].update(payload['overlay_files_sha256'])
        for name, expected in marker['overlay_files_sha256'].items():
            assert hashlib.sha256((code/name).read_bytes()).hexdigest() == expected
        marker.update(code_root=str(code),parent_code_root=str(parent),
            parent_source_manifest_sha256=payload['parent_source_manifest_sha256'],created_at=time.time())
        new_source=(json.dumps(marker,indent=2)+'\n').encode()
        (code/'_CODE_SNAPSHOT_READY').write_bytes(new_source)
        identity=hashlib.sha256(new_source).hexdigest()
        publish(state='testing',source_manifest_sha256=identity)
        runtime=Path(payload['runtime_root']);python=runtime/'envs/b1k-eval/bin/python'
        env=os.environ.copy();env.pop('PYTHONHOME',None)
        env.update(PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='',OMNIGIBSON_NO_OMNIVERSE='1',
            PYTHONPATH=str(code/'sco')+':'+str(code/'scripts'),
            LD_LIBRARY_PATH=str(runtime/'envs/b1k-eval/lib')+':'+env.get('LD_LIBRARY_PATH',''))
        with (output/'pytest.log').open('x') as log:
            subprocess.run([str(python),'-m','pytest','sco/test_accepted_subtask_collection.py',
                'sco/test_full_subtask_collection.py','sco/test_additional_training_collection.py',
                'sco/test_collection_object_checkpoint.py','sco/test_collection_resume.py',
                'sco/test_history_artifact_accounting.py','scripts/test_subtask_improvement.py','-q'],
                cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        inputs=dict(payload['inputs'],code_root=str(code),source_manifest_sha256=identity)
        (output/'inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
        publish(state='auditing_sources')
        driver='import json;from pathlib import Path;from prepare_full_subtask_collection import run;p=Path('+repr(str(output/'inputs.json'))+');run(json.loads(p.read_text()))'
        with (output/'source_audit.log').open('x',buffering=1) as log:
            subprocess.run([str(python),'-u','-c',driver],cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        report=json.loads((Path(inputs['output_root'])/'campaign.json').read_text())
        publish(state=report['state'],prepared_group_count=report['prepared_group_count'],
            expected_trials=report['expected_trials'],campaign_path=str(Path(inputs['output_root'])/'campaign.json'))
    except BaseException as e:
        (output/'error.log').write_text(traceback.format_exc())
        publish(state='failed',error_type=type(e).__name__,error=str(e))
        raise
    return status
