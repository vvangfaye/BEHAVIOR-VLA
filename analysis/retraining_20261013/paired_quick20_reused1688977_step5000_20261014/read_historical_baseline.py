"""Reuse accepted archives; retain only JSON/config/code, never duplicate videos."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]/'BEHAVIOR-VLAs'
sys.path.insert(0,str(REPO/'sco'))
from video_eval_adapter import aoss_client


class DigestStream:
    def __init__(self,raw):self.raw=raw;self.sha=hashlib.sha256();self.size=0
    def read(self,n=-1):
        b=self.raw.read(n);self.sha.update(b);self.size+=len(b);return b


def read_one(item):
    receipt=item['receipt'];index=Path(item['receipt_path']).stem
    root=ROOT/'baseline_evidence'/index;root.mkdir(parents=True,exist_ok=True)
    expected={f['path']:f for f in receipt['files']}
    cached=root/'archive_readback.json'
    if cached.exists():
        c=json.loads(cached.read_text())
        if c['archive_bytes']!=receipt['archive_size'] or c['archive_sha256']!=receipt['archive_sha256']:
            raise ValueError('Cached archive identity changed')
        for name,entry in c['kept'].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=entry['sha256']:
                raise ValueError('Cached evidence changed')
        return finish(item,root,c['kept'],c['archive_bytes'],c['archive_sha256'])
    client=aoss_client()
    body=client.get_object(Bucket='260010165',Key=item['archive_key'])['Body'];stream=DigestStream(body)
    kept={}
    try:
        with tarfile.open(fileobj=stream,mode='r|gz') as archive:
            for member in archive:
                name=member.name.removeprefix('episode/')
                wanted=(name.startswith(('json/','metadata/','config/','wrapper/','code/scripts/'))
                        or (name.startswith('logs/') and name.endswith('.log')))
                if not member.isfile() or not wanted:continue
                if name not in expected or member.size!=expected[name]['bytes'] or member.size>32*1024**2:
                    raise ValueError('Unmanifested or oversized historical evidence: '+name)
                raw=archive.extractfile(member).read()
                if hashlib.sha256(raw).hexdigest()!=expected[name]['sha256']:
                    raise ValueError('Historical evidence SHA mismatch')
                p=root/name
                if not p.resolve().is_relative_to(root.resolve()):raise ValueError('Unsafe archive path')
                p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);os.chmod(p,0o600)
                kept[name]=expected[name]
        while stream.read(8*1024**2):pass
    finally:body.close()
    if stream.size!=receipt['archive_size'] or stream.sha.hexdigest()!=receipt['archive_sha256']:
        raise ValueError('Historical full archive readback differs')
    (root/'archive_readback.json').write_text(json.dumps({'archive_bytes':stream.size,'archive_sha256':stream.sha.hexdigest(),'kept':kept},indent=2)+'\n')
    return finish(item,root,kept,stream.size,stream.sha.hexdigest())


def finish(item,root,kept,archive_size,archive_sha):
    receipt=item['receipt']
    for name,field in [('metadata/source_manifest.json','source_manifest_sha256'),('metadata/checkpoint_ready.json','checkpoint_ready_sha256')]:
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=receipt[field]:
            raise ValueError('Historical source/checkpoint identity chain differs')
    eval_logs=[p for p in (root/'logs').rglob('eval_*.log') if 'metrics' not in p.name]
    if not eval_logs or not any('Seeded Python, NumPy, and Torch with seed=0' in p.read_text() for p in eval_logs):
        raise ValueError('Actual historical evaluator seed0 log missing')
    task,instance=item['task'],item['instance_id']
    result_name=f'json/{task}_{instance}_0.json';result=json.loads((root/result_name).read_text())
    if result['task']!=task or result['instance_id']!=instance or result['rollout_id']!=0 or result['steps']<=0:
        raise ValueError('Native historical result identity mismatch')
    if not (result['terminated'] or result['truncated']):raise ValueError('Historical episode incomplete')
    q=result['q_score']['final']
    if not math.isfinite(q) or not 0<=q<=1 or result['success']!=(q==1):raise ValueError('Invalid actual historical success/Q')
    if result['success']!=receipt['model_success'] or q!=receipt['q_score'] or result['steps']!=receipt['steps']:
        raise ValueError('Receipt and native JSON differ')
    execution=json.loads((root/'metadata/execution.json').read_text());env=execution['env']
    expected_env={'MODE':'public_test','PROMPT_MODE':'whole_task','NUM_ROLLOUTS':'1','WRITE_VIDEO':'1',
        'SUBTASK_PREDICTION':'1','EXECUTE_HORIZON':'16','MAX_TOKEN_LEN':'512','MAX_SUBTASK_LEN':'64',
        'PI05_REPO_ID':'b1k_full100_joint_subtask_v1_drop_episode_s1'}
    if any(env.get(k)!=v for k,v in expected_env.items()) or execution['max_steps'] is not None:
        raise ValueError('Historical inference/termination settings differ')
    if execution['wrapper']!='omnigibson.eval.wrappers.DefaultWrapper' or execution['robot_config']!='omnigibson/eval/r1pro.yaml':
        raise ValueError('Historical robot or wrapper differs')
    reuse=json.loads((root/'metadata/batch_reuse.json').read_text())
    # Preserve evidence verbatim. The first loaded/reset instance is required;
    # later instances cannot reconstruct their already-consumed policy RNG.
    loaded=reuse.get('loaded_instance_ids') if isinstance(reuse,dict) else None
    if loaded is None:
        events=reuse if isinstance(reuse,list) else reuse.get('events',[])
        loaded=[e['instance_id'] for e in events if e.get('event')=='load']
    if not loaded or loaded[0]!=instance:
        raise ValueError('Historical episode was not first in its evaluator process')
    conditioning=result.get('conditioning',{})
    if conditioning.get('prompt_mode','whole_task')!='whole_task':raise ValueError('Historical oracle conditioning')
    proof={'task':task,'instance_id':instance,'archive_key':item['archive_key'],'archive_sha256':archive_sha,
        'archive_bytes':archive_size,'native_result_sha256':kept[result_name]['sha256'],'evidence_files':kept,
        'execution_matches_native_contract':True,'first_instance_in_fresh_process':True,
        'baseline_seed_acknowledged':False,'seed_evidence':'frozen source default key0 + verified fresh process first instance',
        'accepted_at':time.time(),
        'record':dict(task=task,instance_id=instance,artifact_accepted=True,policy_seed=0,environment_seed=0,
            policy_seed_acknowledged=False,policy_seed_source_verified=True,steps=result['steps'],
            model_success=result['success'],q_score=q,terminated=result['terminated'],truncated=result['truncated'],
            result_sha256=kept[result_name]['sha256'])}
    (root/'reuse_proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof


def main():
    selection=json.loads((ROOT/'selection_frozen.json').read_text())
    with ThreadPoolExecutor(max_workers=4) as pool:
        proofs=list(pool.map(read_one,selection['baseline_artifacts']))
    (ROOT/'historical_baseline_proofs.json').write_text(json.dumps(proofs,indent=2)+'\n')
    print(json.dumps({'verified_old_archives':len(proofs),'bytes_read':sum(p['archive_bytes'] for p in proofs),
                      'videos_duplicated':False,'new_model_episodes':0}))


if __name__=='__main__':main()
