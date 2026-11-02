"""Freeze two task-specific replications of the combined PPO method."""
import copy
import hashlib
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
ADAPT = PROJECT / 'analysis/rlinf_adaptation_20261020'
COMBINED = HERE.parent / 'rl_combined_20261102'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def unique_file(source, relative):
    matches = sorted(source.glob('retrieved/*/' + relative))
    assert matches, (source, relative)
    bodies = [p.read_bytes() for p in matches]
    assert all(b == bodies[0] for b in bodies), ('Conflicting completed evidence', relative)
    return bodies[0]


WORKER_PLAN = '''def worker_plan(role, case=None, variant='balanced_gamma1_ppo'):
    config = campaign()
    assert variant in config['evaluation_variants']
    plan = json.loads((HERE / 'source_plan.json').read_text())
    source_output = plan['output']
    source_contract = hashlib.sha256(json.dumps({'goals': plan['goal_contract'], 'shaping': plan['shaping']}, sort_keys=True).encode()).hexdigest()
    plan.update(experiment=config['experiment'] + '_' + variant, comparison_variant=variant,
                comparison_role=role, worker_prefix='cross_', rounds=1, baseline_sources={})
    plan['optimizer']['updates_per_round'] = [17]
    plan['optimizer']['gamma'] = plan['shaping']['gamma'] = 1. if variant == config['variant'] else .99
    plan['source_rollout_root'] = source_output
    receipts = json.loads((HERE / 'source_training_receipts.json').read_text())
    assert len(receipts) == 32 and all(r['contract'] == source_contract for r in receipts)
    assert hashlib.sha256((HERE / 'source_training_receipts.json').read_bytes()).hexdigest() == config['source_rollout_receipts_sha256']
    plan['source_rollout_receipts'] = [{k: r[k] for k in ['episode','instance','seed','path','sha256','policy_hash','contract','chunks']} for r in receipts]
    plan['source_reward_contract_sha256'] = source_contract
    if 'task_bddl_sha256' not in plan['source_identity']:
        plan['source_identity']['task_bddl_sha256'] = plan['goal_contract']['bddl_sha256']
    plan['storage_admission'] = config['storage_admission']
    assert plan['optimizer']['actor_lr'] == config['actor_lr'] and plan['optimizer']['value_lr'] == config['value_lr']
    if role == 'train':
        assert variant == config['variant']
        plan['eval_instances'] = []
        plan['output'] = config['remote_root'] + '/training/results'
        plan['runtime_storage']['max_persistent_bytes'] = config['training_storage_limit_bytes']
        plan['phase_limits_seconds']['train1'] = config['train_timeout_seconds']
        plan['distributed'] = {'world_size':4,'preflight_only':False,'backend':'gloo_cpu_reduction','kl_batch_size':8,
            'runtime_signature':hashlib.sha256(json.dumps(config['runtime_sha256'],sort_keys=True).encode()).hexdigest()}
        gate = json.loads((OLD / 'kl_batch_gate.json').read_text())
        assert gate['passed'] and gate['selected_batch_size'] == 8
        for name in ['kl_batch.py','b1k_rlinf/model.py','b1k_rlinf/precision.py']:
            assert (HERE / 'runtime' / name).read_bytes() == (OLD / 'runtime' / name).read_bytes()
    else:
        assert case in config['eval_cases']
        plan['output'] = config['remote_root'] + f'/evaluations/{variant}/i{case[0]}_s{case[1]}/results'
        plan['comparison_checkpoint'] = (config['remote_root'] + '/training/results/checkpoints/round_01'
            if variant == config['variant'] else config['control_checkpoint']['directory'])
        plan['evaluation']['cases'] = [case]
        plan['evaluation_runtime_order'] = [case]
        plan['eval_instances'] = [case[0]]
        plan['runtime_storage']['max_persistent_bytes'] = config['evaluation_storage_limit_bytes']
        plan['phase_limits_seconds']['final'] = config['eval_timeout_seconds']
        assert plan['model_seed'] == plan['seed'] == 0
    plan['runtime_storage']['shared_budget_roots'] = [plan['output']]
    return plan


'''


def deployment_source():
    code = (COMBINED / 'deployment.py').read_text()
    code = code.replace('PROJECT = HERE.parents[1]', 'PROJECT = HERE.parents[2]')
    code = code.replace("SOURCE = HERE.parent / 'rl_scheme_comparison_20261101'", "SOURCE = PROJECT / 'analysis/rl_scheme_comparison_20261101'")
    code = code.replace("OLD = HERE.parent / 'rl_multigpu_20261101'", "OLD = PROJECT / 'analysis/rl_multigpu_20261101'")
    code = code.replace("ADAPT = HERE.parent / 'rlinf_adaptation_20261020'", "ADAPT = PROJECT / 'analysis/rlinf_adaptation_20261020'")
    start, end = code.index('def worker_plan('), code.index('def storage_proof(')
    code = code[:start] + WORKER_PLAN + code[end:]
    code = code.replace("reserve + margin", "reserve + config['other_remaining_commitments_bytes'] + margin")
    code = code.replace("'other_remaining_commitments_bytes': 0", "'other_remaining_commitments_bytes': config['other_remaining_commitments_bytes']")
    code = code.replace("'36 GiB new-branch reservation: 16 GiB learner plus forty 512 MiB evaluation outputs; inputs already stored'", "'80 GiB shared reserve for both cross-task learners and eighty evaluation outputs; plus 36 GiB for the water combination'")
    code = code.replace('def submit(role, case=None, attempt=1):', "def submit(role, case=None, attempt=1, variant='balanced_gamma1_ppo'):")
    code = code.replace("f'eval-{case[0]}s{case[1]}-a{attempt}'", "f'eval-{\"combo\" if variant == config[\"variant\"] else \"control\"}-{case[0]}s{case[1]}-a{attempt}'")
    code = code.replace('plan = worker_plan(role, case)', 'plan = worker_plan(role, case, variant)')
    code = code.replace("name = 'b1k-combo-1102-' + ident", "name = 'b1k-xrl-1102-' + config['task_label'] + '-' + ident")
    code = code.replace("'variant': config['variant']", "'variant': variant")
    return code


def controller_source():
    code = (COMBINED / 'controller.py').read_text()
    code = code.replace("update['chunks'] == 23908", "update['chunks'] == campaign()['source_chunks']")
    code = code.replace("destination = HERE / 'evidence' / f'i{instance}_s{seed}'", "destination = HERE / 'evidence' / row['variant'] / f'i{instance}_s{seed}'")
    code = code.replace('def comparison():', 'def comparison(variant):')
    code = code.replace("original = SOURCE / 'evidence/original' / key", "original = HERE / 'evidence/original' / key")
    code = code.replace("new = HERE / 'evidence' / key", "new = HERE / 'evidence' / variant / key")
    code = code.replace("'expected_pairs': 40", "'expected_pairs': len(config['eval_cases'])")
    code = code.replace("'complete': n == 40", "'complete': n == len(config['eval_cases'])")
    code = code.replace("result = comparison()\n                    report = json.loads((SOURCE / 'comparisons.json').read_text())\n                    report[config['variant']] = result",
                        "report = {v: comparison(v) for v in config['evaluation_variants']}\n                    result = report[config['variant']]")
    code = code.replace("if result['complete']:", "if all(r['complete'] for r in report.values()):")
    code = code.replace("same forty held-out cases", "same task-specific fixed held-out cases")
    old = """if not (HERE / 'checkpoint.json').exists():
                        tasks = [('train', None)]
                    else:
                        tasks = [('eval', case) for case in config['eval_cases']]"""
    new = """tasks = []
                    if not (HERE / 'checkpoint.json').exists():
                        tasks.append(('train', None, config['variant']))
                    if config['control_needs_evaluation']:
                        tasks.extend(('eval', case, 'control') for case in config['eval_cases'])
                    if (HERE / 'checkpoint.json').exists():
                        tasks.extend(('eval', case, config['variant']) for case in config['eval_cases'])"""
    assert old in code
    code = code.replace(old, new)
    code = code.replace('for role, case in tasks:', 'for role, case, variant in tasks:')
    code = code.replace("r.get('case') == case]", "r.get('case') == case and r['variant'] == variant]")
    code = code.replace("count = 4 if role == 'train' else 1\n                        if global_active", "count = 4 if role == 'train' else 1\n                        if sum(r['gpu_count'] for r in active) + count > config['eval_parallel_gpus']:\n                            continue\n                        if global_active")
    code = code.replace('submit(role, case, 1 if last is None else last[\'attempt\'] + 1)',
                        "submit(role, case, 1 if last is None else last['attempt'] + 1, variant)")
    return code


def main():
    settings = [('trash', ADAPT / 'multitask_validation_20261029/trash'),
                ('modem', ADAPT / 'modem_parallel_20261029')]
    water = json.loads((COMBINED / 'jobs/train-a1/worker_plan.json').read_text())
    plans = {}
    for label, source in settings:
        root = HERE / label
        assert not (root / 'plan.json').exists(), 'Keep the frozen cross-task campaign intact'
        root.mkdir(parents=True, exist_ok=True)
        original = json.loads((source / 'plan.json').read_text())
        save(root / 'source_plan.json', original)
        receipts = [json.loads(unique_file(source, f'training_receipts/round_01/episode_{i:02}.json'))
                    for i in range(1, 33)]
        assert all(r['policy_hash'] == original['source_identity']['cold_trainable_hash'] for r in receipts)
        assert all(r['round'] == 1 for r in receipts)
        contract = hashlib.sha256(json.dumps({'goals': original['goal_contract'], 'shaping': original['shaping']}, sort_keys=True).encode()).hexdigest()
        assert all(r['contract'] == contract for r in receipts)
        save(root / 'source_training_receipts.json', receipts)
        cases = original['evaluation']['heldout_cases']
        for instance, seed in cases:
            key = f'i{instance}_s{seed}'
            for name in ['result.json', 'initial.json', 'spec.json', 'actions.json', 'goal_summary.json']:
                destination = root / 'evidence/original' / key / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(unique_file(source, 'baseline/' + key + '/' + name))
                if label == 'trash':
                    destination = root / 'evidence/control' / key / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(unique_file(source, 'final/' + key + '/' + name))
        ready = json.loads(unique_file(source, 'checkpoints/round_01/ready.json'))
        assert ready['optimizer_steps'] == ready['cumulative_optimizer_steps'] == 17
        assert ready['collection_policy_hash'] == original['source_identity']['cold_trainable_hash']
        runtime = root / 'runtime'
        for path in (COMBINED / 'runtime').rglob('*.py'):
            target = runtime / path.relative_to(COMBINED / 'runtime')
            target.parent.mkdir(parents=True, exist_ok=True)
            body = path.read_text()
            if path.name in ['bootstrap.py', 'single_bootstrap.py']:
                assert "('picking_up_trash','bringing_water')" in body
                body = body.replace("('picking_up_trash','bringing_water')", "('picking_up_trash','bringing_water','installing_a_modem')")
            target.write_text(body)
        # These two task-dependent files are copied unchanged from the validated task runtime.
        for name in ['goals.py', 'simulator.py']:
            (runtime / name).write_bytes((source / name).read_bytes())
        for name in ['common.py', 'experiment.py', 'training.py', 'helpers.py', 'eval_utils.py', 'b1k_rlinf/model.py']:
            assert (runtime / name).read_bytes() == (source / name).read_bytes()
        plan = {
            'experiment': 'rl_cross_task_' + label + '_20261102', 'created_at': time.time(),
            'authorization': 'User asked to test the current combined trajectory-weighted gamma=1 method on other tasks.',
            'task_label': label, 'task': original['task'], 'source_campaign': str(source),
            'variant': 'balanced_gamma1_ppo', 'start_checkpoint': 'original1688977',
            'source_trajectories': 32, 'source_chunks': sum(r['chunks'] for r in receipts),
            'source_successes': sum(r['status']['success'] for r in receipts),
            'optimizer_steps': 17, 'actor_lr': 1e-7, 'value_lr': 1e-4, 'gae_lambda': .95,
            'gamma': 1., 'shaping_gamma': 1., 'weighting': 'N/(32*T_i)',
            'train_gpus': 4, 'eval_parallel_gpus': 6, 'max_total_live_gpus': 32, 'quota_type': 'spot',
            'train_timeout_seconds': 43200, 'eval_timeout_seconds': 5400, 'attempt_cap': 2,
            'max_total_gpu_hours': 192, 'max_wall_hours': 48,
            'eval_cases': cases, 'evaluation_variants': ['balanced_gamma1_ppo', 'control'],
            'control_checkpoint': ready, 'control_needs_evaluation': label == 'modem',
            'reuse_existing_baselines': True, 'no_new_training_rollouts': True,
            'remote_root': '/mnt/afs/260010165/behavior_1k/rl_cross_task_20261102/' + label,
            'aoss_prefix': 'behavior1k/rl-cross-task-20261102/' + label + '/',
            'training_storage_limit_bytes': 16 * 1024**3, 'evaluation_storage_limit_bytes': 512 * 1024**2,
            'campaign_storage_reserve_bytes': 80 * 1024**3, 'other_remaining_commitments_bytes': 36 * 1024**3,
            'storage_margin_bytes': 20 * 1024**3, 'storage_admission': water['storage_admission'],
            'source_rollout_receipts_sha256': hashlib.sha256((root / 'source_training_receipts.json').read_bytes()).hexdigest(),
            'runtime_sha256': {str(p.relative_to(runtime)): hashlib.sha256(p.read_bytes()).hexdigest() for p in runtime.rglob('*.py')},
            'baseline_result_sha256': {f'i{i}_s{s}': hashlib.sha256((root / 'evidence/original' / f'i{i}_s{s}' / 'result.json').read_bytes()).hexdigest() for i, s in cases},
        }
        save(root / 'plan.json', plan)
        (root / 'deployment.py').write_text(deployment_source())
        (root / 'controller.py').write_text(controller_source())
        plans[label] = {k: plan[k] for k in ['task', 'source_trajectories', 'source_chunks', 'source_successes', 'eval_cases', 'control_needs_evaluation']}
        print(label, '32 reused trajectories;', plan['source_chunks'], 'chunks;', len(cases), 'held-out cases')
    save(HERE / 'plan.json', {'created_at': time.time(), 'tasks': plans, 'method': 'trajectory-weighted PPO with gamma=1',
                            'max_active_cross_task_gpus': 12, 'max_active_cross_task_training_gpus': 8,
                            'max_active_including_water_and_prior_comparison': 32,
                            'new_training_trajectories': 0, 'protected_existing_data': True})


if __name__ == '__main__':
    main()
