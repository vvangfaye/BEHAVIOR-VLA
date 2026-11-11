"""Bounded foreground observation only; never create, stop, or retry a job."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
HOLDOUT = ROOT / 'heldout20_baseline_vs_expert1k'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--polls', type=int, choices=range(1, 13), default=8)
    parser.add_argument('--training', action='store_true', help='Observe actual updates after cache staging.')
    args = parser.parse_args()
    sys.path.insert(0, str(HOLDOUT))
    import observe_live
    initial_status = json.loads((HOLDOUT / 'status.json').read_text())
    initial = initial_status['accepted']
    initial_complete = initial_status['complete']
    for tick in range(args.polls):
        script = 'observe_training.py' if args.training else 'observe_cache_progress.py'
        result = subprocess.run([sys.executable, str(ROOT / 'expert_huber50_candidate' / script)],
                                capture_output=True, text=True, timeout=80)
        if result.returncode:
            print(json.dumps({'observation_error': 'cache_query', 'jobs_modified': False}), flush=True)
            return
        cache = json.loads(result.stdout)
        if args.training:
            row = cache.get('latest_metric') or {}
            metrics = row.get('metrics', {})
            training = {'iid': cache['native']['iid'], 'state': cache['native']['application_status'],
                        'completed_updates': row.get('completed_updates'),
                        'metrics': {key: metrics[key] for key in ('loss', 'action_loss', 'action_mse',
                            'action_robust_tail_fraction', 'grad_norm', 'learning_rate',
                            'data_wait_seconds', 'step_seconds', 'nonfinite_count') if key in metrics}}
        status = json.loads((HOLDOUT / 'status.json').read_text())
        proof_path = HOLDOUT / 'report_proof.json'
        completed_proof = json.loads(proof_path.read_text()) if proof_path.exists() else {}
        verified_complete = (status['complete'] and completed_proof.get('complete') is True and
            hashlib.sha256((HOLDOUT / 'status.json').read_bytes()).hexdigest() ==
            completed_proof.get('native_completion', {}).get('status_sha256'))
        if verified_complete:
            live = {'rows': []}
        else:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    observe_live.main()
            except Exception as exc:
                print(json.dumps({'observation_error': type(exc).__name__, 'jobs_modified': False}), flush=True)
                return
            status = json.loads((HOLDOUT / 'status.json').read_text())
            live = json.loads((HOLDOUT / 'current_live.json').read_text())
        stale = [{'member': row['member'], 'seconds': row.get('unchanged_file_seconds', 0)}
                 for row in live['rows'] if row.get('unchanged_file_seconds', 0) > 600]
        collector_age = time.time() - status['observed_at']
        collector_stale = collector_age > 180 and not status['complete']
        print(json.dumps({'tick': tick, **({'training': training} if args.training else {'cache_workers': cache['workers']}), 'accepted': status['accepted'],
              'complete': status['complete'], 'live_evaluation_jobs': len(live['rows']),
              'stale': stale, 'collector_status_age_seconds': round(collector_age),
              'collector_stale': collector_stale, 'observed_at': time.time()}), flush=True)
        terminal = {'FAILED', 'FAIL', 'STOPPED', 'KILLED', 'FINISHED', 'SUCCEEDED'}
        phase_event = (training['state'].upper() in terminal or metrics.get('nonfinite_count', 0) != 0
                       if args.training else
                       any(row.get('stage') != 'expert_cache' or row.get('status', '').upper() in terminal
                           for row in cache['workers']))
        if (status['accepted'] != initial or (status['complete'] and not initial_complete)
                or stale or collector_stale or phase_event):
            return
        if tick + 1 < args.polls:
            time.sleep(45)


if __name__ == '__main__':
    main()
