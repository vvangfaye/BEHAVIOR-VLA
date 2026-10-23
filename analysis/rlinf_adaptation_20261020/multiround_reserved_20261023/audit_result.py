"""Audit experiment completion separately from whether PPO improved performance."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE/'retrieved'


def read(name):
    return json.loads((ROOT/name).read_text())


def main():
    plan=json.loads((HERE/'plan.json').read_text())
    status=read('status.json')
    comparison=read('comparison.json')
    training=read('training_summary.json')
    artifacts=read('artifact_manifest.json')
    baseline, final = read('baseline/summary.json'), read('final/summary.json')
    rounds=training['rounds']
    checks={
        'worker_completed':status['state']=='complete',
        'five_rounds':len(rounds)==plan['rounds'],
        'full_fixed_evaluation':len(baseline)==len(final)==len(plan['evaluation']['cases']),
        'paired_initial_states':comparison['comparison_valid'],
        'frozen_weights_unchanged':training['frozen_sha256_before']==training['frozen_sha256_after'],
        'evaluation_weights_unchanged':all(read(f'{phase}/checks.json')['weights_unchanged'] for phase in ('baseline','final')),
        'real_ppo_updates':all(r['update']['passed'] and r['update']['optimizer_steps']>0 and r['update']['expert_gradient_nonzero'] and r['update']['value_gradient_nonzero'] for r in rounds),
        'checkpoint_round_trips':all(r['checkpoint']['round_trip'] for r in rounds),
        'checkpoints_archived':all(f'checkpoints/round_{i:02d}/trainable.pt' in artifacts for i in range(1,plan['rounds']+1)) and f'checkpoints/round_{plan["rounds"]:02d}/optimizer.pt' in artifacts,
    }
    report={'passed':all(checks.values()),'checks':checks,'comparison':comparison,
            'total_train_controls':training['total_controls'],'total_optimizer_steps':training['total_optimizer_steps'],
            'training_reward_sums':[r['update']['reward_sum'] for r in rounds],
            'performance_conclusion':comparison['conclusion'],'scope':'Single RTX5090 fixed small evaluation panel; completion does not imply improvement.'}
    (HERE/'acceptance_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
