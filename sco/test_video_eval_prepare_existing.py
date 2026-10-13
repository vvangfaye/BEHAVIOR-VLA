import hashlib,io,json,shutil,sys
from pathlib import Path
import pytest
sys.path.insert(0,'/Users/faye/code/Behavior-1k Challenge/BEHAVIOR-VLAs/sco')
from video_eval_prepare_existing import finish_existing
import video_eval_prepare

def fixture(tmp_path,monkeypatch):
 runtime=tmp_path/'runtime';runtime.mkdir();(runtime/'_RUNTIME_READY').write_bytes(b'ready');(runtime/'envs/b1k-pi05').mkdir(parents=True);(runtime/'graphics-runtime').mkdir();(runtime/'graphics-runtime/_READY').write_bytes(b'graphics')
 lib=tmp_path/'lib';lib.mkdir();(lib/'_READY').write_text(json.dumps({'libraries_sha256':{},'library_path':str(lib)}))
 old=tmp_path/'old';old.mkdir()
 for name in ['scripts/eval_b1k.sh','scripts/pi05_eval_queue.py','sco/video_eval_worker.py']:
  path=old/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(name.encode())
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 base={'overlay_files_sha256':{},'graphics_runtime_manifest_sha256':sha(runtime/'graphics-runtime/_READY')};(old/'_CODE_SNAPSHOT_READY').write_text(json.dumps(base))
 oldprov={'base_source_sha256':sha(old/'_CODE_SNAPSHOT_READY'),'files_sha256':{},'supplemental_library':{'root':str(lib),'ready_manifest_sha256':sha(lib/'_READY')}};(old/'_PUBLIC_TEST_SOURCE_READY').write_text(json.dumps(oldprov))
 code=tmp_path/'code';shutil.copytree(old,code);(code/'openpi').mkdir();(code/'openpi/.venv').symlink_to(runtime/'envs/b1k-pi05',target_is_directory=True)
 ckpt=tmp_path/'checkpoint';ckpt.mkdir();items=[]
 for i in range(24):
  name='assets/subtask_conditioning.json' if i==0 else 'params/data'+str(i)
  p=ckpt/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'{"action_conditioning":"task_and_subtask"}' if i==0 else ('data'+str(i)).encode());items.append({'relative_path':name,'bytes':p.stat().st_size,'sha256':sha(p)})
 captures=[]
 class Opener:
  def open(self,req,timeout):captures.append(json.loads(req.data));return io.BytesIO()
 monkeypatch.setattr('urllib.request.build_opener',lambda *_:Opener())
 p={'checkpoint':str(ckpt),'code':str(code),'runtime':str(runtime),'oldcode':str(old),'old_source_sha256':sha(old/'_PUBLIC_TEST_SOURCE_READY'),'items':items,'total_bytes':sum(x['bytes'] for x in items),'run_id':'test','report_put':'https://test.invalid','overlay_files_sha256':{'sco/video_eval_worker.py':sha(code/'sco/video_eval_worker.py')},'overlay_sha256':'a'*64,'scope':{'expected_rollouts':1000},'preparation_recovery':{'old_writer_stopped':True},'plan_sha256':'b'*64}
 return p,captures

def test_expected_environment_link_and_full_checkpoint_readback(tmp_path,monkeypatch):
 p,reports=fixture(tmp_path,monkeypatch);finish_existing(p);r=reports[-1]
 assert r['state']=='ready' and r['files']==24 and r['checkpoint_full_sha256_readback']
 assert r['source_manifest']['shared_runtime_links']['openpi/.venv']['target']==str(tmp_path/'runtime/envs/b1k-pi05')
 assert hashlib.sha256((tmp_path/'checkpoint/_READY').read_bytes()).hexdigest()==r['checkpoint_ready_sha256']

def test_arbitrary_external_link_is_rejected(tmp_path,monkeypatch):
 p,reports=fixture(tmp_path,monkeypatch);(tmp_path/'code/escape').symlink_to(tmp_path/'runtime')
 with pytest.raises(RuntimeError):finish_existing(p)
 assert reports[-1]['state']=='failed' and reports[-1]['stage']=='source_freeze'
 assert not (tmp_path/'checkpoint/_READY').exists()

def test_corrupt_existing_checkpoint_does_not_get_ready(tmp_path,monkeypatch):
 p,reports=fixture(tmp_path,monkeypatch);(tmp_path/'checkpoint/params/data1').write_bytes(b'xxxxx')
 with pytest.raises(RuntimeError):finish_existing(p)
 assert reports[-1]['stage']=='checkpoint_readback'
 assert not (tmp_path/'checkpoint/_READY').exists()

def test_cli_uses_saved_name_and_verifies_uid(monkeypatch):
 calls=[]
 class Reply:
  returncode=0
  stdout=json.dumps({'name':'saved-name','uid':'saved-uid','ownership':{'user_name':'260010165'},'state':'RUNNING'})
 def fake(args):calls.append(args);return Reply()
 monkeypatch.setattr(video_eval_prepare,'cli',fake)
 assert video_eval_prepare.describe_preparation({'name':'saved-name','uid':'saved-uid'})['platform_state']=='RUNNING'
 assert calls[0][3]=='saved-name'
 with pytest.raises(ValueError):video_eval_prepare.describe_preparation({'name':'saved-name','uid':'different'})
