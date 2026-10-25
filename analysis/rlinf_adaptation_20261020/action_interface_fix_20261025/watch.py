"""Bounded result collection for this pair only; never submit or restart jobs."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def main():
    limits = []
    for rank in (0, 1):
        receipt = json.loads((HERE / f'attempt02/r{rank}/submission.json').read_text())
        submitted = datetime.fromisoformat(receipt['submitted_at']).timestamp()
        limits.append(submitted + receipt['max_seconds'] + 300)
    previous = None
    while time.time() < max(limits):
        result = subprocess.run([sys.executable, str(HERE / 'observe.py')],
                                capture_output=True, text=True, timeout=120)
        if result.returncode:
            print(json.dumps({'state': 'observation_error', 'at': datetime.now(timezone.utc).isoformat(),
                              'error': result.stderr[-1500:]}), flush=True)
            time.sleep(40)
            continue
        snapshot = json.loads(result.stdout)
        compact = [{'instance': w['instance'], 'state': w['state'],
                    'phase': w.get('status', {}).get('stage'),
                    'last_result': w.get('status', {}).get('last_result'),
                    'report_passed': w.get('report', {}).get('passed')}
                   for w in snapshot['workers']]
        if compact != previous:
            print(json.dumps({'at': snapshot['observed_at'], 'workers': compact,
                              'summary': snapshot.get('summary')}, ensure_ascii=False), flush=True)
            previous = compact
        if snapshot.get('summary') or all(w['state'] in ('SUCCEEDED', 'FAILED', 'SUSPENDED')
                                          for w in snapshot['workers']):
            return
        time.sleep(40)
    print(json.dumps({'state': 'observation_deadline', 'no_jobs_resubmitted': True}), flush=True)


if __name__ == '__main__':
    main()
