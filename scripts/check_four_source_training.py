#!/usr/bin/env python3
"""Check frozen metadata and the real LR schedule. Does not start a trainer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['JAX_PLATFORMS'] = 'cpu'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'openpi/src'))
from openpi.training.subtask_mixture import MixturePlan
from openpi.training.optimizer import CosineDecaySchedule
from openpi.shared.learning_rate_contract import check_schedule, expected_rate


def check(prepared, plan_path):
    plan=json.loads(plan_path.read_text());mixture=MixturePlan.load(prepared/'mixture.json')
    s=plan['learning_rate'];schedule=CosineDecaySchedule(warmup_steps=s['warmup_steps'],peak_lr=s['peak_lr'],
                                                       decay_steps=s['decay_end_step'],decay_lr=s['floor_lr'])
    contract=check_schedule(schedule,prepared/'learning_rate_contract.json')
    init=Path(plan['initialization']['params_path'])
    if not init.is_dir():
        raise ValueError('Initial checkpoint parameters are not available')
    norm=init.parent/'assets/b1k_full100_joint_subtask_v1_drop_episode_s1/norm_stats.json'
    if hashlib.sha256(norm.read_bytes()).hexdigest()!=plan['model']['normalization_sha256']:
        raise ValueError('Normalization differs from the approved checkpoint')
    if not mixture.four_source or mixture.num_steps!=plan['budget']['max_new_optimizer_updates'] or mixture.global_batch_size!=192:
        raise ValueError('Mixture update budget or batch differs from plan')
    argv=['pi05_b1k','--exp-name','four_source_1688977_v1','--batch-size','192','--num-train-steps',str(mixture.num_steps),
          '--seed','42','--save-interval','1000','--keep-period','5000','--log-interval','10',
          '--model.train-subtask-prediction','--model.subtask-action-conditioning','task_and_subtask',
          '--model.max-token-len','512','--model.max-subtask-len','64','--model.subtask-loss-weight','1.0',
          '--data.subtask-whole-task-prob','0','--data.frame-stride','1','--data.video-backend','pyav',
          '--data.subtask-mixture-path',str((prepared/'mixture.json').resolve()),
          '--data.subtask-mixture-roots-path',str((prepared/'roots.json').resolve()),
          '--data.assets.assets-dir',str(norm.parent.parent), '--data.assets.asset-id',norm.parent.name,
          '--learning-rate-contract-path',str((prepared/'learning_rate_contract.json').resolve()),
          '--lr-schedule.warmup-steps',str(s['warmup_steps']),'--lr-schedule.peak-lr',str(s['peak_lr']),
          '--lr-schedule.decay-steps',str(s['decay_end_step']),'--lr-schedule.decay-lr',str(s['floor_lr']),
          '--weight-loader.params-path',str(init)]
    (prepared/'training_argv.json').write_text(json.dumps({'argv':argv,'launch_authorized':False,
          'root_mapping_must_be_resolved_on_training_host':True},indent=2)+'\n')
    return dict(training_started=False,metadata_and_lr_checks_passed=True,mixture_fingerprint=mixture.fingerprint,
                learning_rate_sha256=contract['sha256'],checkpoint_directory_step=1688977,
                expected_initial_optimizer_step=0,normalization_verified=True,
                full_train_state_initialized=False,
                lr_milestones={str(i):expected_rate(contract,i) for i in [0,200,1000,5000,10000,20000,40000]},
                training_host_batch_and_gpu_smoke_pending=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    a=p.parse_args();r=check(a.prepared,a.plan);(a.prepared/'cpu_preflight.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
