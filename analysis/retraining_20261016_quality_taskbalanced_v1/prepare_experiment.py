"""Build isolated CPU, two-step GPU, and formal bundles; never submit resources."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'retraining_20261015_comet_rft60_40'
SOURCE=PREVIOUS/'formal_stage/bundle'
REVIEW=PREVIOUS/'high_exposure_review'
RUN='b1k-quality-taskbalanced-1688977-20261016'

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True)+'\n')

def sampler_code(original):
    x=original
    before='        self.fingerprint = _digest(self._semantic_plan())'
    after='''        self._quality_sampler = None
        if "quality_sampling" in manifest:
            from openpi.training.task_balanced_windows import TaskBalancedWindowSampler
            self.sampler_version = 4
            self._quality_sampler = TaskBalancedWindowSampler(self, manifest["quality_sampling"])
        self.fingerprint = _digest(self._semantic_plan())'''
    assert x.count(before)==1;x=x.replace(before,after)
    before='        phase = self._phases[bisect.bisect_right(self._ends, global_index)]'
    after='''        if self._quality_sampler is not None:
            return self._quality_sampler.reference(global_index)
        phase = self._phases[bisect.bisect_right(self._ends, global_index)]'''
    assert x.count(before)==1;x=x.replace(before,after)
    before='        return {"sampler_version": self.sampler_version'
    assert x.count(before)==1;x=x.replace(before,'        semantic = {"sampler_version": self.sampler_version')
    tail='''\n        if self._quality_sampler is not None:
            semantic["quality_sampling"] = self.manifest["quality_sampling"]
        return semantic
'''
    boundary='\n\ndef _prefix('
    assert x.count(boundary)==1;x=x.replace(boundary,tail+boundary)
    compile(x,'subtask_mixture.py','exec')
    return x

def cut_spans(intervals,holds,horizon):
    output=[]
    for s in intervals:
        spans=[(s['start'],s['end'])]
        for h in holds:
            next_spans=[]
            for lo,hi in spans:
                if h['end_frame']<=lo or h['start_frame']>=hi:next_spans.append((lo,hi))
                else:
                    if lo<h['start_frame']:next_spans.append((lo,h['start_frame']))
                    if h['end_frame']<hi:next_spans.append((h['end_frame'],hi))
            spans=next_spans
        output.extend(dict(s,start=lo,end=hi) for lo,hi in spans if hi-lo>=horizon)
    return output

def build():
    if (ROOT/'preparation.json').exists():raise FileExistsError('Already prepared; inspect existing output')
    auth=read(ROOT/'execution_authorization.json');assert auth['formal_training_allowed'] and auth['max_completed_updates']==5000
    masks=read(REVIEW/'quality_mask.proposed.json')['proposed_hold_intervals']
    base_manifest=read(SOURCE/'bundle_manifest.json')
    for f,h in base_manifest.items():
        if sha(SOURCE/f)!=h:raise ValueError('Frozen base changed: '+f)
    base_index_paths=read(SOURCE/'inputs/compiled/mixture.json')['phases'][0]['sources']
    fullroots={read(SOURCE/'inputs/compiled'/r['path'])['root_id'] for s in base_index_paths if s['kind']=='full_episode' for r in s['indexes']}
    assert all(h['root_id'] in fullroots for h in masks)
    stage_summary={}
    for stage,base in [('cpu',PREVIOUS/'cpu_repair/bundle'),('smoke',PREVIOUS/'gpu_smoke/bundle'),('formal',SOURCE)]:
        dest=ROOT/('formal_stage' if stage=='formal' else stage)/'bundle'
        if dest.exists():raise FileExistsError(str(dest))
        source_hashes=read(base/'bundle_manifest.json')
        for f,h in source_hashes.items():
            if sha(base/f)!=h:raise ValueError('Stage base changed: '+stage+'/'+f)
            target=dest/f;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/f,target)
        module=dest/'openpi/src/openpi/training/subtask_mixture.py'
        module.write_text(sampler_code(module.read_text()))
        shutil.copy2(ROOT/'task_balanced_windows.py',module.with_name('task_balanced_windows.py'))
        mixture=read(dest/'inputs/compiled/mixture.json')
        mixture['quality_sampling']={'kind':'task_balanced_capped_windows_v1','task_count':100,
            'within_task_full_ceiling':.2,'full_episode_draw_cap':256,'window_stride':32,'maximum_full_window_passes':2,'finite_budget_updates':5000}
        mixture['phases'][0]['end_step']=5000
        for source in mixture['phases'][0]['sources']:
            for item in source['indexes']:
                path=dest/'inputs/compiled'/item['path'];idx=read(path)
                if source['kind']=='full_episode':
                    matched=[h for h in masks if h['root_id']==idx['root_id']]
                    idx['intervals']=cut_spans(idx['intervals'],matched,32)
                    idx['provenance']['quality_review']={'status':'partial_review_provisional_holds','full270_semantics_verified':False,
                        'mask_sha256':sha(REVIEW/'quality_mask.proposed.json'),'held_ranges':matched,
                        'original_index_sha256':item['sha256'],'labels_changed':False}
                    idx['provenance']['valid_anchors']=sum(x['end']-x['start']-31 for x in idx['intervals'])
                    idx['fingerprint']=digest({k:v for k,v in idx.items() if k!='fingerprint'})
                    save(path,idx);item['sha256']=sha(path)
        # Compute the exact capped budget from the modified interval metadata.
        by_task={}
        for source in mixture['phases'][0]['sources']:
            if source['kind']!='full_episode':continue
            for item in source['indexes']:
                idx=read(dest/'inputs/compiled'/item['path'])
                for ep in idx['episodes']:
                    windows=sum((s['end']-s['start'])//32 for s in idx['intervals'] if s['episode_index']==ep['episode_index'])
                    by_task.setdefault(ep['task_name'],{}).setdefault(ep['task_instance_id'],[]).append(windows)
        total=sum(min(256,1920//len(instances)//len(eps),2*w) for instances in by_task.values() for eps in instances.values() for w in eps)
        for source in mixture['phases'][0]['sources']:
            source['weight']=total/960000 if source['kind']=='full_episode' else 1-total/960000
        save(dest/'inputs/compiled/mixture.json',mixture)
        # Import in a separate interpreter so each stage is checked against its own code.
        import subprocess
        code="import sys;sys.path.insert(0,sys.argv[1]);from openpi.training.subtask_mixture import MixturePlan;p=MixturePlan.load(sys.argv[2]);print(p.fingerprint);print(p._quality_sampler.full_draws)"
        result=subprocess.run([sys.executable,'-c',code,str(dest/'openpi/src'),str(dest/'inputs/compiled/mixture.json')],capture_output=True,text=True,check=True)
        fingerprint,draws=result.stdout.strip().splitlines();assert int(draws)==total
        run=RUN if stage=='formal' else RUN+'-'+stage
        output='/data/oss_bucket_0/wangfaye/result/behavior-challenge/quality_taskbalanced/'+run
        jobpath=dest/('nebula/worker_job.json' if stage=='formal' else 'nebula/smoke_job.json')
        job=read(jobpath);job.update(run_id=run,output=output,mixture_fingerprint=fingerprint,
            gpu_smoke_prefix='/data/oss_bucket_0/wangfaye/result/behavior-challenge/quality_taskbalanced/'+RUN+'-smoke',resume=False)
        if stage=='formal':
            job.update(mode='formal_finetune',formal_training_authorized=True,attempt_id='stage5000-g0',max_steps=5000,stage_end_step=5000)
            save(dest/'inputs/formal_authorization.json',auth)
            contract=read(dest/'inputs/formal_run_contract.json');contract.update(run_id=run,mixture_fingerprint=fingerprint)
            contract['quality_sampling_policy']=mixture['quality_sampling'];save(dest/'inputs/formal_run_contract.json',contract)
        else:
            job['formal_training_authorized']=False
            assert job['updates']==(0 if stage=='cpu' else 2)
        save(jobpath,job)
        if stage=='cpu':save(dest/'probe_job.json',{'run_id':run,'output':output})
        assert sha(dest/'inputs/compiled/learning_rate_contract.json')==sha(SOURCE/'inputs/compiled/learning_rate_contract.json')
        manifest={str(p.relative_to(dest)):sha(p) for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='bundle_manifest.json' and '__pycache__' not in p.parts}
        save(dest/'bundle_manifest.json',manifest)
        stage_summary[stage]={'run_id':run,'output':output,'mixture_fingerprint':fingerprint,
            'bundle_manifest_sha256':sha(dest/'bundle_manifest.json'),'files':len(manifest),'full_draws':total,'submitted':False}
    assert len({x['mixture_fingerprint'] for x in stage_summary.values()})==1
    result={'kind':'quality_taskbalanced_preparation','created_at':time.time(),'stages':stage_summary,
        'full_training_draws':total,'original_training_draws':960000-total,'full_fraction':total/960000,
        'seed':42,'completed_update_budget':5000,'global_batch':192,
        'learning_rate_contract_sha256':sha(SOURCE/'inputs/compiled/learning_rate_contract.json'),
        'partial_quality_review_disclosed':True,'old_STOP_files_preserved':True,'cloud_jobs_created':0}
    save(ROOT/'preparation.json',result);print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':build()
