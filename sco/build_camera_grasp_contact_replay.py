"""Freeze a finite model-free replay against camera attempt003's actual actions."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from build_countertop_contact_replay import DRIVER, ROOT


def build(output, quota='reserved'):
    if quota not in ('reserved','spot'):raise ValueError('Invalid quota')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    evidence=ROOT/'eval_results/codex_subtask_camera_20261013'
    manifest=json.loads((evidence/'trial003-v1/manifest.json').read_text())
    producer=json.loads((evidence/'trial003-v1/artifact_download.json').read_text())
    if producer['state']!='producer_sha256_verified':raise ValueError('Unverified original attempt artifacts')
    source=(ROOT/'scripts/replay_codex_camera_grasp.py').read_text()
    compile(source,'camera_grasp_replay_worker','exec')
    config=dict(kind='camera_grasp_contact_replay_v1',collection_payload=manifest['payload'],
        output_root='/mnt/afs/260010165/behavior_1k/codex_subtask_camera_20261013/'+output.name,
        plan_path=manifest['payload']['plan_path'],plan_fingerprint=manifest['plan_fingerprint'],
        attempt_root=manifest['payload']['run_root']+'/attempt',robot_config=manifest['payload']['robot_config'],
        worker_sha256=hashlib.sha256(source.encode()).hexdigest(),expected_steps=628,
        max_gpu_count=1,quota=quota,max_job_seconds=6000,diagnostic_only=True,model_calls=0,training_eligible=False)
    driver=DRIVER.replace('countertop_replay_job_v1','camera_grasp_replay_job_v1')
    driver=driver.replace(" if hashlib.sha256((states/'on_top.py').read_bytes()).hexdigest()!=config['on_top_sha256']:raise ValueError('Original OnTop source changed')\n",'')
    driver=driver.replace("time.monotonic()+4200","time.monotonic()+5700")
    driver=driver.replace("report['summary']['executed_steps']!=960","report['summary']['executed_steps']!=config['expected_steps']")
    marker=" report['state']='starting_replay';publish()"
    assert driver.count(marker)==1
    driver=driver.replace(marker," env['B1K_LEGACY_HISTORY_RECOVERY']='recorded_actions_before_start_native_grasp_recovery_v1'\n"+marker)
    driver=re.sub(r'\b(CONFIG|SOURCE)\b',lambda m:repr(config if m[0]=='CONFIG' else source),driver)
    compile(driver,'camera_grasp_replay_driver','exec')
    (output/'driver.py').write_text(driver)
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    return config


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--quota',choices=('reserved','spot'),default='reserved')
    args=p.parse_args();config=build(args.output,args.quota)
    print(json.dumps({k:config[k] for k in ('kind','output_root','expected_steps','worker_sha256','max_job_seconds','model_calls')}))
