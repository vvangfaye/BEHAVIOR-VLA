"""Check real synthetic export and the OpenPI reader in their archived envs."""
import json
import hashlib
import os
from pathlib import Path
import subprocess


def run(payload):
    root=Path(payload['output_root']);root.mkdir(parents=True,exist_ok=False)
    code,runtime=Path(payload['code_root']),Path(payload['runtime_root'])
    marker=(code/'_CODE_SNAPSHOT_READY').read_bytes()
    assert hashlib.sha256(marker).hexdigest()==payload['source_manifest_sha256']
    for name, expected in json.loads(marker)['overlay_files_sha256'].items():
        assert hashlib.sha256((code/name).read_bytes()).hexdigest()==expected
    env=os.environ.copy();env.pop('PYTHONHOME',None)
    env.update(PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='',OMNIGIBSON_NO_OMNIVERSE='1',
        PYTHONPATH=':'.join(str(code/p) for p in ('scripts','sco','BEHAVIOR-1K','BEHAVIOR-1K/OmniGibson','openpi/src')),
        LD_LIBRARY_PATH=str(runtime/'envs/b1k-eval/lib')+':'+env.get('LD_LIBRARY_PATH',''))
    # Keep the existing complete round-trip fixture, then use the environment
    # containing the actual training dependencies for its final reader phase.
    phase1='''import json
from pathlib import Path
import pytest
from test_finetune_subtask_collection import checkpoint
from test_subtask_collection_roundtrip import test_real_mixed_export_and_training_horizon_readback
root=Path(ROOT);fixture=root/'fixture';fixture.mkdir()
patch=pytest.MonkeyPatch()
try:
    test_real_mixed_export_and_training_horizon_readback(fixture,patch,checkpoint.__wrapped__(fixture))
except ModuleNotFoundError as error:
    if error.name!='etils':raise
    assert (fixture/'export/_VERIFIED_EXPORT').is_file()
finally:patch.undo()
(root/'export_phase.json').write_text(json.dumps(dict(real_lerobot_export_verified=True)))
'''.replace('ROOT',repr(str(root)))
    (root/'export_phase.py').write_text(phase1)
    reader=root/'reader_phase.py'
    reader.write_text(payload['reader_source']+'\n'+
        'result=run('+repr(dict(dataset_root=str(root/'fixture/export'),repo_id='fixture/exported'))+')\n'+
        'Path('+repr(str(root/'training_reader.json'))+').write_text(json.dumps(result,indent=2)+"\\n")\n')
    for phase,python in (('export_phase','b1k-eval'),('reader_phase','b1k-pi05')):
        with (root/(phase+'.log')).open('x') as log:
            result=subprocess.run([str(runtime/'envs'/python/'bin/python'),str(root/(phase+'.py'))],
                cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            (root/'status.json').write_text(json.dumps(dict(state='failed',phase=phase,returncode=result.returncode)))
            raise RuntimeError('Format check failed: '+phase)
    report=json.loads((root/'training_reader.json').read_text())
    assert report['complete'] and report['actual_openpi_training_reader']
    report.update(state='complete',synthetic_format_only=True,real_simulator_collection=False)
    (root/'status.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
