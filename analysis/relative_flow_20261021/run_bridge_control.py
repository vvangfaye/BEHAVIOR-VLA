"""Launch the registered bridge-point control only after the four-arm gate."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'action_support_20261021'))
from analyze_vla_pilot import score,compare,sha

PARENT=HERE/'pilot_ada01'
ROOT=HERE/'bridge_control_ada01'


def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')


def main():
    plan=read(ROOT/'plan.json')
    if not (PARENT/'completion.json').exists():
        print(json.dumps(dict(launched=False,reason='Waiting for four-arm completion.')));return
    assert sha(PARENT/'plan.json')==plan['parent_plan_sha256']
    assert sha(HERE/'analyze.py')==read(PARENT/'analysis_plan.json')['analyzer_sha256']
    subprocess.run([sys.executable,str(HERE/'analyze.py')],check=True,stdout=subprocess.DEVNULL)
    parent=read(PARENT/'reanalysis.json')
    if not parent['screen_passed']:
        write(ROOT/'skipped.json',dict(launched=False,reason='Original four-arm screen failed.',
              parent_result_sha256=sha(PARENT/'reanalysis.json')))
        print(json.dumps(dict(launched=False,reason='Original four-arm screen failed.')));return
    if (ROOT/'launch.json').exists():raise ValueError('Refusing repeated launch')
    for name,expected in plan['source_sha256'].items():assert sha(name)==expected
    assert read(HERE/'bridge_control_check.json')['passed']
    processes=subprocess.check_output(['nvidia-smi','--id=0','--query-compute-apps=pid',
                                      '--format=csv,noheader,nounits'],text=True).strip()
    if processes:raise RuntimeError('GPU0 is occupied')
    write(ROOT/'launch.json',dict(controller_pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),
                                 gpu=0,parent_result_sha256=sha(PARENT/'reanalysis.json')))
    for stage in ['smoke','formal']:
        config=ROOT/('smoke_plan.json' if stage=='smoke' else 'plan.json')
        settings=read(config);out=ROOT/stage;out.mkdir(exist_ok=False)
        record=dict(stage=stage,method='bridge_single',gpu=0,started_at=datetime.now(timezone.utc).isoformat(),
                    plan_sha256=sha(config),max_seconds=settings['max_seconds']+180,no_retry=True)
        start=time.monotonic()
        with (out/'worker.log').open('x') as stream:
            child=subprocess.Popen([sys.executable,'-u',str(HERE/'train_bridge_control.py'),
                  '--plan',str(config),'--method','bridge_single','--output',str(out/'output')],
                  cwd=HERE,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4'),
                  stdout=stream,stderr=subprocess.STDOUT)
            record['worker_pid']=child.pid;write(out/'launch.json',record)
            print(json.dumps(dict(event='launch',**record)),flush=True)
            try:code=child.wait(timeout=record['max_seconds'])
            except subprocess.TimeoutExpired:
                child.terminate()
                try:code=child.wait(timeout=60)
                except subprocess.TimeoutExpired:child.kill();code=child.wait()
        result=read(out/'output/result.json') if (out/'output/result.json').exists() else {}
        accepted=code==0 and result.get('complete') and result.get('completed_steps')==settings['steps']
        receipt=dict(**record,returncode=code,accepted=bool(accepted),elapsed_seconds=time.monotonic()-start,
                     result_sha256=sha(out/'output/result.json') if result else None)
        write(out/'completion.json',receipt)
        print(json.dumps(dict(event='finished',**receipt)),flush=True)
        if not accepted:raise SystemExit(1)
        reference=PARENT/(stage+'_relative_difference')/'output'
        ref_result=read(reference/'result.json');ref_run=read(reference/'run.json')
        run=read(out/'output/run.json')
        for key in ['data_trace_sha256','noise_trace_sha256','training_expert_forwards','training_prefix_forwards']:
            assert result[key]==ref_result[key],key
        assert run['initial_attention_sha256']==ref_run['initial_attention_sha256']
    with np.load(ROOT/'formal/output/test.npz',allow_pickle=False) as z:
        point={k:z[k].copy() for k in ['samples','target','index','frame','episode','noise_sha256']}
    with np.load(PARENT/'formal_relative_difference/output/test.npz',allow_pickle=False) as z:
        relative={k:z[k].copy() for k in point}
    for key in ['target','index','frame','episode','noise_sha256']:
        np.testing.assert_array_equal(point[key],relative[key])
    metrics={name:score(a['samples'],a['target']) for name,a in [('point',point),('relative',relative)]}
    means={name:{k:float(v.mean()) for k,v in values.items()} for name,values in metrics.items()}
    delta={k:compare(metrics['relative'][k],metrics['point'][k],point['episode']) for k in metrics['point']}
    passed=(delta['energy_score']['anchor_mean_delta']<0
            and delta['energy_score']['anchor_weighted_cluster_bootstrap95'][1]<0
            and means['relative']['mean_action_mse']<=1.05*means['point']['mean_action_mse'])
    report=dict(ready=True,all_gates_passed=bool(parent['screen_passed'] and passed),means=means,
                relative_minus_bridge_point=delta,parent_result_sha256=sha(PARENT/'reanalysis.json'),
                control_plan_sha256=sha(ROOT/'plan.json'),
                scope='Additional registered attribution control. Passing is not a closed-loop or novelty claim.',
                goal_complete=False)
    write(ROOT/'reanalysis.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
