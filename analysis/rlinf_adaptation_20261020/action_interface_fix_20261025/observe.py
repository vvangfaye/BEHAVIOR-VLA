"""Read bounded status and reports; never change or resubmit cloud jobs."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client


def main():
    client = aoss_client()
    observations = []
    reports = []
    try:
        for rank in (0, 1):
            target = HERE / 'attempt02' / f'r{rank}'
            if not (target / 'submission.json').exists():
                continue
            receipt = json.loads((target / 'submission.json').read_text())
            result = subprocess.run([str(Path.home() / '.local/bin/sco'), 'acp', 'jobs', 'describe',
                receipt['job'], '--workspace-name=share-space', '-o', 'json'],
                capture_output=True, text=True, check=True, timeout=45)
            job = json.loads(result.stdout)
            assert job['uid'] == receipt['uid'] and job['ownership']['user_name'] == '260010165'
            row = {'rank': rank, 'job': receipt['job'], 'uid': receipt['uid'], 'state': job['state'],
                   'instance': receipt['instance']}
            for name in ('status.json', 'report.json'):
                try:
                    response = client.get_object(Bucket='260010165', Key=receipt['aoss_prefix'] + name)
                except client.exceptions.ClientError as error:
                    if error.response['Error']['Code'] in ('NoSuchKey', '404'):
                        continue
                    raise
                raw = response['Body'].read()
                (target / name).write_bytes(raw)
                item = json.loads(raw)
                row[name[:-5]] = item
                if name == 'report.json':
                    reports.append(item)
            observations.append(row)
    finally:
        client.close()
    current = {'observed_at': datetime.now(timezone.utc).isoformat(), 'workers': observations}
    (HERE / 'latest.json').write_text(json.dumps(current, ensure_ascii=False, indent=2) + '\n')
    active_path = HERE.parent / 'active_run.json'
    if active_path.exists():
        active = json.loads(active_path.read_text())
        validation = active.setdefault('interface_validation', {})
        validation.update(observed_at=current['observed_at'], workers=observations,
                          status='results_received' if len(reports) == 2 and all(r.get('passed') for r in reports)
                          else 'incomplete' if any(r['state'] in ('FAILED', 'SUSPENDED') for r in observations)
                          else 'running')
        active_path.write_text(json.dumps(active, ensure_ascii=False, indent=2) + '\n')
    if len(reports) == 2 and all(r.get('passed') for r in reports):
        plan = json.loads((HERE / 'plan.json').read_text())
        for report in reports:
            assert report['no_weight_updates'] and report['frozen_parameters_unchanged']
            assert report['base_hash'] == plan['expected_original_trainable_sha256']
            assert report['ppo_hash'] == plan['audit']['checkpoints'][-1]['hash']
            assert report['same_interface']['gripper_open_tolerance'] == .001
            assert report['initial_max_error'] <= 1e-4
            assert set(report['density_checks']) == {'baseline', 'ppo'}
            assert all(gate['passed'] for gate in report['density_checks'].values())
            for phase in ('baseline', 'ppo'):
                assert len(report[phase]) == 1
                assert report[phase][0]['instance'] == report['instance']
        pairs = [{'instance': r['instance'], 'seed': r['seed'],
                  'baseline': r['baseline'][0], 'ppo': r['ppo'][0],
                  'initial_max_error': r['initial_max_error'],
                  'predicted_gripper_command_changes': r['modified_gripper_commands']} for r in reports]
        summary = {'interface_checks_passed': True, 'no_weight_updates': True, 'pairs': pairs,
                   'same_interface_for_both': reports[0]['same_interface'] == reports[1]['same_interface'],
                   'baseline_successes': sum(p['baseline']['success'] for p in pairs),
                   'ppo_successes': sum(p['ppo']['success'] for p in pairs), 'episodes_per_checkpoint': 2,
                   'change_counter_note': 'Counts all predicted chunk slots, including an unused tail at termination; not an executed-command count.',
                   'scope': 'Two fixed instances, one seed, existing frozen weights; not a test of general RL benefit.'}
        summary['baseline_mean_score'] = sum(p['baseline']['score'] for p in pairs) / len(pairs)
        summary['ppo_mean_score'] = sum(p['ppo']['score'] for p in pairs) / len(pairs)
        summary['success_to_failure'] = sum(p['baseline']['success'] and not p['ppo']['success'] for p in pairs)
        summary['failure_to_success'] = sum(not p['baseline']['success'] and p['ppo']['success'] for p in pairs)
        summary['performance_regression'] = (summary['ppo_successes'] < summary['baseline_successes']
                                              or summary['ppo_mean_score'] < summary['baseline_mean_score'])
        (HERE / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
        if active_path.exists():
            active = json.loads(active_path.read_text())
            active['interface_validation'].update(status='complete', summary=str(HERE / 'summary.json'))
            active_path.write_text(json.dumps(active, ensure_ascii=False, indent=2) + '\n')
        current['summary'] = summary
    print(json.dumps(current, ensure_ascii=False))


if __name__ == '__main__':
    main()
