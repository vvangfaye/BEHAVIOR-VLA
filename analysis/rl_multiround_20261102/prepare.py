"""Freeze plan.json for rl_multiround_20261102 (local only; submits nothing).

Usage: python3 prepare.py [--rounds N] [--learner-pool h100|5090] [--learner-quota reserved_if_free|spot]
Defaults describe the 3-round modem pipeline validation; nothing is launched by this script.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
CROSS = ANALYSIS / 'rl_cross_task_20261102/modem'
POWER = ANALYSIS / 'rl_eval_power_lr_20261102'

parser = argparse.ArgumentParser()
parser.add_argument('--rounds', type=int, default=3)
parser.add_argument('--learner-pool', choices=['h100', '5090'], default='h100')
parser.add_argument('--learner-quota', choices=['reserved_if_free', 'spot', 'reserved'], default='reserved_if_free')
parser.add_argument('--actor-lr', type=float, default=1e-6)
args = parser.parse_args()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


GIB = 1024**3
runtime = {str(p.relative_to(HERE / 'runtime')): sha(p)
           for p in sorted((HERE / 'runtime').rglob('*.py')) if '__pycache__' not in p.parts}
train_instances = list(range(1, 17))
rollouts_per_instance = 4
eval_cases = [[i, s] for i in range(301, 321) for s in range(5)]
final_eval_cases = [[i, s] for i in range(301, 321) for s in range(10)]  # user 2026-11-03: final round 200
plan = {
    'experiment': 'rl_multiround_20261102',
    # After launch (launch.json), regenerating the plan keeps the campaign's start (its wall-clock deadline).
    'created_at': (json.loads((HERE / 'plan.json').read_text())['created_at'] if (HERE / 'launch.json').exists()
                   else time.time()),
    'authorization': ('User 2026-11-02: prepare the code for "4 standard H100 train + Spot 5090 continuous sampling". '
                      'User 2026-11-03 launch go-ahead: "启动吧，最后一轮用200局" (3 rounds, actor LR 1e-6, '
                      '32 Spot 5090 samplers, final-round evaluation on 200 paired cases).'),
    'task': 'installing_a_modem',
    'source_plan_sha256': sha(CROSS / 'source_plan.json'),
    'rounds': args.rounds,
    'train_instances': train_instances,
    'rollouts_per_instance': rollouts_per_instance,
    'episodes_per_round': len(train_instances) * rollouts_per_instance,
    'min_episodes_per_round': int(0.95 * len(train_instances) * rollouts_per_instance),
    # User 2026-11-03: use 32 Spot 5090 samplers per round (64 episodes -> 2 each, ~40 min per round
    # instead of 16 x 4 = ~75 min).
    'episodes_per_sampler': 2,
    'seed_base': 400000,
    'actor_lr': args.actor_lr, 'value_lr': 1e-4, 'gamma': 0.99, 'gae_lambda': 0.95, 'optimizer_steps': 17,
    'carry_optimizer': True,
    'learner_pool': args.learner_pool,
    'learner_quota_policy': args.learner_quota,
    # NCCL on H100 (NVLink). On RTX 5090 nodes NCCL all-reduce hit 'CUDA error: an illegal memory access'
    # (validation 2026-11-03, val-learn-5090-full-r1); keep the proven gloo CPU reduction there.
    'collective_backend': {'h100': 'nccl', '5090': 'gloo_cpu_reduction'},
    'kl_batch_size': 8,
    # User 2026-11-03 ("就按你说的办"): cache the frozen prefix once per round (prefix_cache_probe_20261103:
    # 98% of each forward, bitwise-equal outputs), check the full KL after every step (cheap with the cache;
    # removes the sampled-KL early stop), save a resume checkpoint every 4 verified steps.
    'fast_learner': {'microbatch_size': 4, 'sampled_kl_chunks': 2048, 'full_kl_every': 1,
                     'sampled_kl_threshold_fraction': 1.0, 'checkpoint_every': 4, 'prefix_cache': True},
    'replay_tolerance': {'max_ratio_error': 0.01, 'max_mean_kl': 1e-5},
    'eval_cases': eval_cases,
    'final_eval_cases': final_eval_cases,
    # User 2026-11-03 11:2x ("改成只评最后一版吧"): only the final policy is evaluated (200 paired cases); earlier
    # policies are tracked by the next round's sampling success and can be evaluated afterwards if needed.
    'eval_rounds': [args.rounds],
    'reuse_original_evidence': str(POWER / 'evidence/orig'),
    'sample_timeout_seconds': 2 * 3600,  # 2 episodes of ~17 min each; also bounds the GPU-hour accounting
    'learner_timeout_seconds': 2 * 3600,  # cached learner round ~15 min; uncached fallback ~70 min
    'eval_timeout_seconds': 5400,
    'round_sampling_storage_bytes': 16 * GIB,
    'learner_storage_bytes': 24 * GIB,
    'evaluation_storage_bytes': 512 * 1024**2,
    'campaign_storage_reserve_bytes': args.rounds * 40 * GIB + 60 * GIB,
    'storage_margin_bytes': 20 * GIB,
    'max_campaign_live_gpus': 32,
    'max_pending_jobs': 8,
    'final_max_pending_jobs': 32,  # user 2026-11-03 15:5x: congested Spot queue during the final evaluation
    'attempt_cap': 3,
    'preemption_attempt_cap': 10,
    'max_total_gpu_hours': 80 * args.rounds,
    'max_wall_hours': 18 * args.rounds,
    'remote_root': '/mnt/afs/260010165/behavior_1k/rl_multiround_20261102',
    'aoss_prefix': 'behavior1k/rl-multiround-20261102/',
    'job_prefix': 'b1k-mr-1102-',
    'runtime_sha256': runtime,
    'protected': ['existing checkpoints and expert data', 'other campaigns', 'SkyWM jobs'],
}
(HERE / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: plan[k] for k in ['rounds', 'episodes_per_round', 'learner_pool', 'learner_quota_policy',
                                       'actor_lr', 'max_total_gpu_hours']}, ensure_ascii=False))
