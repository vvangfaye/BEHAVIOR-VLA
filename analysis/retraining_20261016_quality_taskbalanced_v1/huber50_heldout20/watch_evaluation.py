"""Finite read-only observation of existing Huber50 evaluation jobs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--polls', type=int, choices=range(1, 13), default=6)
    args = parser.parse_args()
    initial = None
    for tick in range(args.polls):
        result = subprocess.run([sys.executable, str(ROOT / 'observe_live.py')],
                                capture_output=True, text=True, timeout=90)
        if result.returncode:
            print(json.dumps({'observation_unavailable': True, 'jobs_modified': False}), flush=True)
            return
        live = json.loads((ROOT / 'current_live.json').read_text())
        status = json.loads((ROOT / 'status.json').read_text())
        failed_reports = []
        for index in range(20):
            path = ROOT / 'reports' / f'candidate-{index:02d}.json'
            if path.exists():
                report = json.loads(path.read_text())
                if report.get('state') == 'failed':
                    failed_reports.append({'id': report.get('id'), 'error_type': report.get('error_type')})
        if initial is None:
            initial = status['accepted']
        stale = [{'member': r['member'], 'unchanged_file_seconds': r['unchanged_file_seconds']}
                 for r in live['rows'] if r.get('unchanged_file_seconds', 0) > 600]
        stale_videos = [{'member': r['member'], 'unchanged_video_seconds': r['unchanged_video_seconds']}
                        for r in live['rows'] if r.get('video_started') and r.get('unchanged_video_seconds', 0) > 600]
        seed_receipts = sum(bool(r.get('files', {}).get('seed_ack.json', {}).get('exists')) for r in live['rows'])
        videos_nonempty = sum(any(name.endswith('.mp4') and (f.get('bytes') or 0) > 0
                                  for name, f in r.get('files', {}).items()) for r in live['rows'])
        print(json.dumps({'tick': tick, 'observed_at': live['observed_at'],
                          'accepted': status['accepted'], 'complete': status['complete'],
                          'live_states': dict(Counter(r['state'] for r in live['rows'])),
                          'active_members': [r['member'] for r in live['rows'] if r['member']],
                          'completed_seed_receipts_present': seed_receipts, 'videos_nonempty': videos_nonempty,
                          'collector_stale': live.get('collector_stale'), 'stale_members': stale,
                          'stale_videos': stale_videos, 'failed_reports': failed_reports}), flush=True)
        if (status['accepted'] != initial or status['complete'] or live.get('collector_stale')
                or stale or stale_videos or failed_reports or any(s['state'] == 'FAILED' for s in status['states'])):
            return
        if tick + 1 < args.polls:
            time.sleep(45)


if __name__ == '__main__':
    main()
