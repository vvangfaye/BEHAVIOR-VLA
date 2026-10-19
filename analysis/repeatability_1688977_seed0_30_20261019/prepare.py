"""Freeze the authorized 10-instance x 3-repeat seed0 experiment. No jobs submitted."""
from pathlib import Path
import hashlib
import json
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
REFERENCE = PROJECT / 'analysis/retraining_20261016_quality_taskbalanced_v1/heldout20_baseline_vs_expert1k'
ASSETS = PROJECT / 'analysis/organizer_rehearsal_1688977_20261016/runtime/datasets/2026-challenge-task-instances'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(name, value):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as out:
        json.dump(value, out, indent=2, ensure_ascii=False, allow_nan=False)
        out.write('\n')


def main():
    if (ROOT / 'plan.json').exists():
        raise SystemExit('Plan already frozen; do not overwrite an experiment')
    source = json.loads((REFERENCE / 'plan.json').read_text())
    tasks = ['turning_on_radio', 'bringing_water', 'installing_a_modem',
             'putting_dirty_dishes_in_sink', 'make_rose_centerpieces']
    unique_pairs = [[task, instance] for task in tasks for instance in [301, 302]]
    task_rows = {r['task_name']:r for line in (ASSETS/'metadata/task.jsonl').read_text().splitlines()
                 if line.strip() for r in [json.loads(line)]}
    budgets = {task:int(1.5*task_rows[task]['length']) for task in tasks}
    assets = []
    for task, instance in unique_pairs:
        paths = [p for p in (ASSETS/'scene_test/public').rglob('*.json')
                 if task in p.name and f'_{instance}_' in p.name and p.name.endswith('-tro_state.json')]
        assert len(paths)==1, (task,instance,paths)
        p=paths[0]
        assets.append(dict(pair=[task,instance],relative_path=str(p.relative_to(ASSETS)),
                           bytes=p.stat().st_size,sha256=sha(p.read_bytes())))
    protocol = dict(checkpoint=1688977, experiment='fixed_seed_repeatability_phase1',
        unique_pairs=unique_pairs, repeats_per_pair=3, policy_seed=0, environment_seed=0,
        python_hash_seed=0, evaluator_mode='public_test', prompt_mode='whole_task',
        external_agent=False, subtask_prediction=True, subtask_temperature=0.0,
        subtask_action_conditioning='task_and_subtask', executed_action_horizon=16,
        max_token_len=512,max_subtask_len=64,write_video=True,
        fresh_policy_process_each_repeat=True,fresh_evaluator_process_each_repeat=True,
        same_gpu_for_three_repeats=True,independent_cache_each_repeat=True,
        native_task_budgets=budgets,instance_assets=assets,
        initial_state_recorded_before_action=True,action_trace_recorded=True,
        no_retry_or_score_based_filter=True,all_initial_state_differences_reported=True,
        interpretation='Fixed-input/run repeatability diagnostic; not a model quality benchmark.',
        scope='Only phase1: thirty new episodes, no training, no other checkpoint, no other campaign modified.')
    write('protocol.json',protocol)
    plan={k:source[k] for k in ['code','runtime','baseline_checkpoint','repo_id','norm_sha256',
          'baseline_files','source_manifest_sha256','critical_files','supplemental_library','image']}
    plan.update(run_id='b1k-repeat1688-seed0-30-1019',
        afs_root='/mnt/afs/260010165/behavior_1k/repeatability_1688977_seed0_30_20261019',
        object_prefix='behavior1k/repeatability-1688977-seed0-30-20261019',
        task_metadata_sha256=sha((ASSETS/'metadata/task.jsonl').read_bytes()),
        native_task_budgets=budgets,instance_assets=assets,
        protocol_sha256=sha(json.dumps(protocol,sort_keys=True,separators=(',',':')).encode()),
        pairs=[pair for pair in unique_pairs for _ in range(3)],
        groups=[dict(id=f'batch-{g:02d}',name=f'b1k-repeat1688-1019-batch-{g:02d}',
                     members=[dict(role='baseline',index=3*g+r,repeat=r) for r in range(3)])
                for g in range(10)],
        member_walltime_seconds=8*3600,job_walltime_seconds=24*3600+1800,
        creation_deadline_epoch=time.time()+72*3600,
        template='b1k-expert1k-1016-5x4-batch-03',template_uid='af01e81fcc2749ecac41845ca2e0cce2',
        workspace='share-space',pool='computing-cluster-5090-01g',worker_spec='n12lp.nn.i10a.1',
        volume_id='01a04263-91e5-7603-bc01-c67e503da6b5',quota_type='spot',
        expected_episodes=30,max_attempts_per_episode=1,policy_seed=0,environment_seed=0,
        cpu_threads=4,protocol=protocol)
    frozen=ROOT/'frozen';frozen.mkdir()
    for name in ['worker.py','native_helpers.py','server_entry.py','seed_protocol.py','supervisor.py','batch_runner.py']:
        text=(REFERENCE/'frozen'/name).read_text()
        if name=='worker.py':
            old="'--mode','train','--prompt-mode','whole_task','--num-rollouts','1','--instance-indices',str(instance)"
            new="'--mode','public_test','--prompt-mode','whole_task','--num-rollouts','1','--instance-indices',str(instance-301)"
            assert text.count(old)==1
            text=text.replace(old,new)
            text=text.replace("if row['success'] != (q == 1):", "if False:") if False else text
            old="or row['success'] != (q == 1)"
            assert text.count(old)==1
            text=text.replace(old,'')
            old="if [task, instance] not in plan['pairs'] or payload['role'] not in ('baseline', 'candidate'):"
            assert text.count(old)==1
            text=text.replace(old,"if [task, instance] not in plan['pairs'] or payload['role'] != 'baseline':")
            # Readbacks and traces do not alter observations/actions or select outcomes.
            old="    token = secrets.token_hex(32)"
            assert text.count(old)==1
            text=text.replace(old,"    eval_env['REPEATABILITY_OUTPUT'] = str(own)\n"+old)
            old="        write(own/'acceptance.json',result)"
            assert text.count(old)==1
            text=text.replace(old,"        result.update(repeat=payload['repeat'], initial_state=load(own/'initial_state.json'), action_trace=load(own/'action_trace.json'), gpu_rows=gpu_rows, runtime_contract=plan['protocol'])\n"+old)
        if name=='native_helpers.py':
            old='    if process == "policy":'
            assert text.count(old)==1
            text=text.replace(old,'    env.update(PYTHONHASHSEED="0", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="4")\n'+old)
        if name=='batch_runner.py':
            assert text.count('len(results) == 4')==1
            text=text.replace('len(results) == 4','len(results) == 3')
            text=text.replace('Four fixed episodes','Three repeated episodes')
        (frozen/name).write_text(text)
    # Evaluator entry is intentionally authored separately for observable reset states.
    entry=ROOT/'evaluator_entry.py'
    assert entry.exists()
    (frozen/'evaluator_entry.py').write_bytes(entry.read_bytes())
    hashes={p.name:sha(p.read_bytes()) for p in frozen.glob('*.py')}
    write('frozen_sha256.json',hashes)
    write('plan.json',plan)
    write('authorization.json',dict(user_request='先用1688977跑第一阶段的30个回合，固定seed和运行配置',
        authorized_episodes=30,authorized_checkpoint=1688977,training_authorized=False,
        protocol_sha256=plan['protocol_sha256'],timestamp=time.time()))
    print(json.dumps(dict(pairs=len(unique_pairs),episodes=30,gpu_jobs=10,budgets=budgets,protocol_sha256=plan['protocol_sha256'])))


if __name__=='__main__':main()
