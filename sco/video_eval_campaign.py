"""Wait for the verified 2M download, prepare storage, then run the bounded queue."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from video_eval_queue import Controller, write_json
from video_eval_prepare import step
from video_eval_adapter import Adapter


def wait_after_round(root, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end and not (root / 'STOP').exists():
        time.sleep(min(1, max(0, end - time.monotonic())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    with (root / 'campaign.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(root / 'controller_process.json', {'pid': os.getpid(), 'started_at': time.time(),
            'poll_seconds': 45, 'purpose': 'checkpoint2000000_submission1000_with_videos'})
        while not (root / 'STOP').exists():
            try:
                ready = step(root, root / 'preparation_plan.json')
                phase = {'phase': 'preparation', 'state': ready['state'], 'observed_at': time.time()}
            except Exception as error:
                phase = {'phase': 'preparation', 'state': 'error', 'error_type': type(error).__name__,
                         'observed_at': time.time()}
                ready = {}
            write_json(root / 'campaign_status.json', phase)
            print(json.dumps(phase), flush=True)
            if ready.get('state') == 'ready':
                break
            if args.once:
                return
            wait_after_round(root, 45)
        if (root / 'STOP').exists():
            return
        plan = json.loads((root / 'preparation_plan.json').read_text())
        plan['immutable_inputs'] = {k: ready[k] for k in
            ('source_manifest_sha256', 'checkpoint_ready_sha256', 'worker_sha256')}
        path = root / 'plan.json'
        if path.exists():
            assert json.loads(path.read_text()) == plan, 'Existing production plan changed'
        else:
            write_json(path, plan, exclusive=True)
        adapter = Adapter(root, plan)
        status = Controller(root, adapter).run(once=args.once, poll_seconds=45)
        write_json(root / 'campaign_status.json', {'phase': 'evaluation',
            'state': status['completion_state'], 'observed_at': time.time(),
            'verified': status['verified_episodes']})
        if status['finished']:
            from video_eval_finalize import summarize
            summary = summarize(root)
            if summary['all_results_verified']:
                from video_eval_finish import step as finish
                while not (root / 'STOP').exists():
                    try:
                        report = finish(root)
                    except Exception as error:
                        report = {'state': 'finalization_error', 'error_type': type(error).__name__}
                    write_json(root / 'campaign_status.json', dict(report, phase='finalization',
                                                                  observed_at=time.time()))
                    print(json.dumps(report), flush=True)
                    if report['state'] in ('complete', 'finalization_failed', 'stopped'):
                        break
                    wait_after_round(root, 45)


if __name__ == '__main__':
    main()
