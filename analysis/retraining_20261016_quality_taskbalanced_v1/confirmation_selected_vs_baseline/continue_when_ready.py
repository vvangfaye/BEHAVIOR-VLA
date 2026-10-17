"""Monitor existing tuning, publish its complete comparison, then admit confirmation."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
TUNING = ROOT.parent / 'huber50_heldout20'


def load(path):
    return json.loads(path.read_text())


def save(value):
    value['observed_at'] = time.time()
    path = ROOT / 'continuation_status.json'
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)
    print(json.dumps(value), flush=True)


def run(path, *args):
    result = subprocess.run([sys.executable, str(path), *args], capture_output=True,
                            text=True, timeout=180)
    if result.returncode:
        raise RuntimeError('Continuation entry failed: ' + path.name)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    with (ROOT / 'continuation.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # The existing campaign's original finite deadline is never extended.
        deadline = load(TUNING / 'plan.json')['deadline_epoch']
        while time.time() < deadline:
            try:
                run(TUNING / 'observe_live.py')
            except (RuntimeError, subprocess.TimeoutExpired):
                save({'phase': 'observation_unavailable', 'jobs_resubmitted': False,
                      'deadline_epoch': deadline})
                if args.once:
                    return
                time.sleep(60)
                continue
            status = load(TUNING / 'status.json')
            live = load(TUNING / 'current_live.json')
            failed = []
            for path in (TUNING / 'reports').glob('candidate-*.json'):
                row = load(path)
                if row.get('state') == 'failed':
                    failed.append({'id': row.get('id'), 'error_type': row.get('error_type')})
            attention = {'failed_reports': failed, 'collector_stale': live.get('collector_stale', False),
                         'stale_videos': [r['member'] for r in live['rows'] if r.get('video_started')
                                          and r.get('unchanged_video_seconds', 0) > 600],
                         'stale_members': [r['member'] for r in live['rows']
                                           if r.get('unchanged_file_seconds', 0) > 600]}
            if status.get('complete'):
                if status['accepted'] != 20 or attention['collector_stale']:
                    raise ValueError('Completion state and acceptance count disagree')
                # Validates every source, all native UIDs/terminal states and metrics.
                run(TUNING / 'build_report.py')
                run(ROOT / 'admit.py', '--start')
                selection = load(ROOT / 'selection.json')
                launch_path = ROOT / 'controller_launch.json'
                if selection['confirmation_required'] and not launch_path.exists():
                    raise ValueError('Selected model lacks confirmation controller launch')
                save({'phase': 'confirmation_controller_launched' if launch_path.exists()
                      else 'baseline_selected_no_self_comparison',
                      'selected_model': selection['selected_model'],
                      'confirmation_pid': load(launch_path)['pid'] if launch_path.exists() else None,
                      'three_model_report': str(TUNING / 'report.html'),
                      'three_model_csv': str(TUNING / 'comparison.csv')})
                return
            save({'phase': 'waiting_for_complete_tuning', 'accepted': status['accepted'],
                  'attention': attention, 'deadline_epoch': deadline,
                  'jobs_resubmitted': False, 'confirmation_started': False})
            if args.once:
                return
            time.sleep(60)
        save({'phase': 'original_tuning_deadline_reached', 'jobs_resubmitted': False})


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        save({'phase': 'requires_reconciliation', 'error_type': type(exc).__name__,
              'jobs_resubmitted': False})
        raise
