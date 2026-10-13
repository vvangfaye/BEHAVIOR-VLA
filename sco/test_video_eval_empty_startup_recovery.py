import hashlib,json
import pytest
from video_eval_adapter import preserve_approved_empty_startup


def payload(tmp_path):
 root=tmp_path/'0021';root.mkdir()
 p=dict(assignment_root=str(root),run_id='b1k-video2m-20261013-e0021',task='putting_away_Halloween_decorations',instance_id=302,checkpoint_step=2000000,source_manifest_sha256='a'*64,checkpoint_ready_sha256='b'*64)
 raw=json.dumps(p).encode();(root/'payload.json').write_bytes(raw)
 p['approved_empty_startup_recovery']=dict(user_authorization='允许仅补交这一条',old_writer_stopped=True,old_platform_state='FAILED',model_rollouts_executed=0,old_payload_sha256=hashlib.sha256(raw).hexdigest())
 return root,p,raw


def test_preserves_empty_original_and_only_once(tmp_path):
 root,p,raw=payload(tmp_path);preserve_approved_empty_startup(p)
 assert not root.exists()
 assert (tmp_path/'0021.startup-g000/payload.json').read_bytes()==raw
 with pytest.raises(FileNotFoundError):preserve_approved_empty_startup(p)


def test_refuses_any_model_queue_or_output(tmp_path):
 root,p,raw=payload(tmp_path);(root/'queue.sqlite3').touch()
 with pytest.raises(AssertionError):preserve_approved_empty_startup(p)
 assert (root/'payload.json').read_bytes()==raw
 assert not (tmp_path/'0021.startup-g000').exists()


def test_refuses_changed_original_or_missing_approval(tmp_path):
 root,p,raw=payload(tmp_path);p['approved_empty_startup_recovery']['user_authorization']='继续'
 with pytest.raises(AssertionError):preserve_approved_empty_startup(p)
 p['approved_empty_startup_recovery']['user_authorization']='允许仅补交这一条'
 (root/'payload.json').write_bytes(raw+b' ')
 with pytest.raises(AssertionError):preserve_approved_empty_startup(p)
 assert root.exists()
