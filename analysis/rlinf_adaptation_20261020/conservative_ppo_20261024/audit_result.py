"""Distinguish protocol completion, trust-region checks, and performance outcome."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
R=HERE/'retrieved'
def read(name):return json.loads((R/name).read_text())
def main():
    status=read('status.json')
    if status['state']!='complete':
        report={'passed':False,'worker_state':status,'reason':'Incomplete experiment; inspect saved failure and checkpoints.'}
    else:
        plan=json.loads((HERE/'plan.json').read_text());training=read('training_summary.json');comparison=read('comparison.json');old=read('previous_comparison.json');warm=read('critic_warmup.json');artifacts=read('artifact_manifest.json')
        checks={'two_rounds':len(training['rounds'])==2,'four_episodes_per_round':all(len(r['rollouts'])==4 for r in training['rounds']),
                'full_fixed_evaluation':len(read('baseline/summary.json'))==len(read('final/summary.json'))==6,
                'old_missing_evaluation_completed':len(old['pairs'])==6 and old['comparison_valid'],
                'initial_states_paired':comparison['comparison_valid'],'critic_only_warmup':warm['actor_unchanged'],
                'frozen_parameters_unchanged':training['frozen_sha256_before']==training['frozen_sha256_after'],
                'kl_bound_respected':all(r['update']['final_batch_kl']['mean']<=plan['optimizer']['kl_target']+1e-6 for r in training['rounds']),
                'checkpoints_verified':all(r['checkpoint']['round_trip'] for r in training['rounds']),
                'checkpoints_uploaded':all(f'checkpoints/round_{i:02d}/trainable.pt' in artifacts for i in [1,2]) and 'checkpoints/round_02/optimizer.pt' in artifacts}
        report={'passed':all(checks.values()),'checks':checks,'comparison':comparison,'old_comparison':old,
                'total_controls':training['total_controls'],'total_optimizer_steps':training['total_optimizer_steps']}
    (HERE/'acceptance_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
