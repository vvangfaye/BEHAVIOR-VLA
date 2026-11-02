"""Freeze plan.json for rl_eval_power_lr_20261102 (local only; submits nothing)."""
import hashlib
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
CROSS = ANALYSIS / 'rl_cross_task_20261102/modem'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


runtime = {str(p.relative_to(HERE / 'runtime')): sha(p)
           for p in sorted((HERE / 'runtime').rglob('*.py')) if '__pycache__' not in p.parts}
cases = [[instance, seed] for instance in range(301, 321) for seed in range(10)]
repeat_cases = [c for c in cases if c[1] < 5]
control = json.loads((CROSS / 'control_restore_verified.json').read_text())
inputs = json.loads((CROSS / 'input_restore_verified.json').read_text())
assert control['passed'] and inputs['passed'] and inputs['all_sha256_verified']

plan = {
    'experiment': 'rl_eval_power_lr_20261102',
    'created_at': time.time(),
    'authorization': ('User 2026-11-02: "请帮我分别做1和2" — (1) make evaluation accurate: about 150-300 paired '
                      'episodes per arm, iterate on a short task first; (2) make updates bigger: scan actor LR '
                      '1e-6..5e-6 and check clip_fraction / KL against the 0.01 rollback line. Durable research '
                      'authorization: AGENTS.md (2026-10-20); prefer free Reserved, else Spot (2026-10-24).'),
    'task': 'installing_a_modem',
    'task_label': 'modem',
    'start_checkpoint': 'original1688977',
    'source_campaign': str(CROSS),
    'source_plan_sha256': sha(CROSS / 'source_plan.json'),
    'source_rollout_receipts_sha256': sha(CROSS / 'source_training_receipts.json'),
    'restored_receipts_sha256': sha(CROSS / 'restored_training_receipts.json'),
    'source_trajectories': 32,
    'source_chunks': 5677,
    # Workstream 2: same 32 fixed on-policy rollouts, plain PPO (gamma .99), only the actor LR changes.
    'training_arms': {
        'lr1e7': {'actor_lr': 1e-07, 'evaluate': False, 'purpose': 'same-runtime KL/clip reference at the current LR'},
        'lr1e6': {'actor_lr': 1e-06, 'evaluate': True},
        'lr5e6': {'actor_lr': 5e-06, 'evaluate': True},
    },
    'training_common': {'variant': 'control', 'value_lr': 1e-4, 'gamma': 0.99, 'gae_lambda': 0.95,
                        'optimizer_steps': 17, 'clip_ratio': 0.2, 'kl_target': 0.01,
                        'kl_rollback': True, 'allow_kl_early_stop': True, 'train_gpus': 4, 'kl_batch_size': 8},
    # Workstream 1: high-power paired evaluation on one fixed case set.
    'eval_cases': cases,
    'eval_arms': {
        'orig': {'checkpoint': None, 'cases': 'all', 'label': 'original 1688977'},
        'origrep': {'checkpoint': None, 'cases': 'seeds_0_4', 'label': 'original 1688977, independent repeat (A/A noise floor)'},
        'old17': {'checkpoint': control['directory'], 'trainable_sha256': control['source_trainable_hash'],
                  'cases': 'all', 'label': 'existing modem plain PPO, 17 updates, LR 1e-7 (behind the 25->19 result)'},
        'lr1e6': {'checkpoint': 'training:lr1e6', 'cases': 'all', 'label': 'plain PPO, 17 updates, LR 1e-6'},
        'lr5e6': {'checkpoint': 'training:lr5e6', 'cases': 'all', 'label': 'plain PPO, up to 17 updates, LR 5e-6'},
    },
    'repeat_cases': repeat_cases,
    'primary_test': 'two-sided exact McNemar on paired success vs orig; paired sign-flip test on score',
    'max_campaign_live_gpus': 32,
    'combo_guard_account_total_gpus': 30,
    'reserved_quota_gpus': 4,
    'train_quota_type': 'spot',
    'eval_quota_policy': 'reserved when own active Reserved GPUs + 1 <= 4, else spot',
    'train_timeout_seconds': 21600,
    'eval_timeout_seconds': 5400,
    'eval_attempt_cap': 3,
    'train_attempt_cap': 3,
    'max_total_gpu_hours': 360,
    'max_wall_hours': 36,
    'remote_root': '/mnt/afs/260010165/behavior_1k/rl_eval_power_lr_20261102',
    'aoss_prefix': 'behavior1k/rl-eval-power-lr-20261102/',
    'job_prefix': 'b1k-plr-1102-',
    'training_storage_limit_bytes': 16 * 1024**3,
    'evaluation_storage_limit_bytes': 512 * 1024**2,
    'campaign_storage_reserve_bytes': 110 * 1024**3,
    'storage_margin_bytes': 20 * 1024**3,
    'runtime_sha256': runtime,
    'runtime_changes': {
        'parallel_training.py': 'read-only PPO diagnostics: RLinf actor/clip_fraction during gradient, and post-update '
                                'full-batch fraction of chunk ratios outside [0.8, 1.2] via the existing full_kl observe hook',
        'distributed_driver.py': 'control variant may early-stop at the 0.01 KL rollback when allow_kl_early_stop; '
                                 'ready.json records early_stopped and actor_lr',
        'comparison_driver.py': 'final phase can evaluate the original policy without an overlay; writes policy_identity.json',
    },
    'phase2_gate': ('Multi-round sample->update is NOT launched here. After this campaign, pick the largest LR whose '
                    'paired success change vs orig is not significantly negative and whose clip fraction is nonzero; '
                    'then run multi-round online PPO with that LR as a separate recorded campaign.'),
    'protected': ['existing checkpoints and expert data', 'rl_combined_20261102 running evaluation',
                  'SkyWM jobs', 'source rollouts (read-only)'],
}
(HERE / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'eval_cases': len(cases), 'repeat_cases': len(repeat_cases), 'runtime_files': len(runtime),
                  'eval_episodes_total': 200 * 4 + 100}, ensure_ascii=False))
