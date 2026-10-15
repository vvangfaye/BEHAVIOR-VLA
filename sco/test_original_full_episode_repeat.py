import copy
import json
from pathlib import Path
import pytest
from original_full_episode_repeat import validate_campaign,validate_original_repeat,aggregate,gate_allowed,MODEL
from full_episode_scope import validate_final1000

CAMPAIGN=Path(__file__).resolve().parents[1]/'eval_results/sco_original1688977_collection3000_20261015'

def load():return json.loads((CAMPAIGN/'authorization.json').read_text())

def test_three_rounds_share_finite_scope_and_unique_new_seeds():
 plans=validate_campaign(CAMPAIGN,load());assert len(plans)==3
 seeds=[s for p in plans for s in p['policy_seeds'].values()]
 assert len(seeds)==len(set(seeds))==3000
 old=json.loads((CAMPAIGN.parent/'sco_full_episode_collection_20261014_final1000_round3/round-3/plan.json').read_text())
 assert not set(seeds)&set(old['policy_seeds'].values())
 for root,p in zip(load()['roots'],plans):
  assert validate_final1000(p) and gate_allowed(root,p)
  assert p['checkpoint']==MODEL and p['checkpoint_identity']==old['checkpoint_identity']

@pytest.mark.parametrize('field,value',[('checkpoint','wrong'),('sampling_round',4),('publish_compact_metrics',False),('cpu_thread_policy','modified')])
def test_scope_rejects_wrong_model_and_protocol(field,value):
 p=json.loads((CAMPAIGN/'round-6/plan.json').read_text());p[field]=value
 with pytest.raises(AssertionError):validate_original_repeat(p)

def test_aggregate_repeats_are_distinct_not_deduplicated():
 rows=[dict(completed_episodes=2,successful_episodes=1,mean_q_score=.6,competition_sr=.5,bc_eligible_episodes=1),dict(completed_episodes=1,successful_episodes=0,mean_q_score=.3,competition_sr=0,bc_eligible_episodes=0)]
 result=aggregate(rows)
 assert result['completed_episodes']==3 and result['target_episodes']==3000
 assert result['success_rate']==pytest.approx(1/3) and result['mean_q_score']==pytest.approx(.5)
 assert result['missing_episodes']==2997

def test_frozen_policy_keeps_original_settings_and_full_rgb_capture():
 old=CAMPAIGN.parent/'sco_full_episode_collection_20261014_final1000_round3/round-3/overlay_sources'
 for rnd in (6,7,8):
  source=CAMPAIGN/f'round-{rnd}/overlay_sources'
  for file in ['sco/run_public_test_assignment.py','scripts/eval_b1k.sh','scripts/full_episode_trajectory.py','scripts/eval_b1k_queue.py','openpi/scripts/b1k/serve_full_episode_repeat.py']:
   assert (source/file).read_bytes()==(old/file).read_bytes()
  assert "upload_json(metrics,manifest['metrics_object_storage'][key])" in (source/'scripts/full_episode_checkpoint.py').read_text()
  for file in source.rglob('*.py'):compile(file.read_bytes(),str(file),'exec')

@pytest.mark.parametrize('failure',['capacity','query','timeout'])
def test_precreate_query_failure_preserves_evidence_without_cloud_copy(tmp_path,monkeypatch,failure):
 import hashlib
 import subprocess
 import full_episode_collection as c
 root=tmp_path/'leaf';root.mkdir();(root/'submissions').mkdir();(root/'overlay.tar.gz').write_bytes(b'overlay')
 source=tmp_path/'source/sco';source.mkdir(parents=True);driver=b'def run(payload): pass\n';(source/'run_full_episode_collection.py').write_bytes(driver)
 p=dict(run_id='test-run',expected_episodes=2000,shards=[dict(index=0,task='turning_on_radio',instance_ids=[301],pilot=True)],overlay_sha256=hashlib.sha256(b'overlay').hexdigest(),overlay_files_sha256={'sco/run_full_episode_collection.py':hashlib.sha256(driver).hexdigest()})
 p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();(root/'plan.json').write_text(json.dumps(p));obs=tmp_path/'capacity.json';obs.write_text('{}')
 monkeypatch.setattr(c,'REPO',source.parent)
 monkeypatch.setattr(c,'describe',lambda *a:dict(uid='c14d0892eaa148d381c7f4424684c9d3',resource_pool={'name':c.POOL}))
 monkeypatch.setattr(c.subprocess,'check_output',lambda *a,**k:'2 TB 0 TB')
 called=[]
 def no_copy(*a,**k):called.append('copy');raise AssertionError('must not create')
 monkeypatch.setattr(c.subprocess,'run',no_copy)
 count=[]
 def gate(*a):
  count.append(1)
  if len(count)==2:
   if failure=='capacity':raise c.CapacityUnavailable('full')
   if failure=='query':raise subprocess.CalledProcessError(70,['list'])
   raise subprocess.TimeoutExpired(['list'],30)
 monkeypatch.setattr(c,'require_capacity',gate)
 class Storage:
  def generate_presigned_url(self,*a,**k):return 'https://test.invalid/private'
  def put_object(self,**k):pass
  def create_multipart_upload(self,**k):return {'UploadId':'test'}
 with pytest.raises((c.CapacityUnavailable,subprocess.CalledProcessError,subprocess.TimeoutExpired)):
  c.submit(root,0,obs,'cli',Storage())
 assert not called and not (root/'submissions/0000.json').exists()
 saved=list((root/'unsubmitted_preparations').glob('*.json'));assert len(saved)==1
 intent=json.loads(saved[0].read_text());assert intent['state']=='preparing_objects' and 'created_at' not in intent
