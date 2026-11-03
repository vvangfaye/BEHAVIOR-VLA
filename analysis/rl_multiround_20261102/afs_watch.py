"""AFS headroom / controller-error watchdog for rl_multiround_20261102 (user 2026-11-03: "注意afs 2t的空间限制").

Strictly read-only and CLI-free: the account quota query temporarily switches the global SCO zone, so a second
caller could race with the controller. This watchdog only reads the quota reading the controller itself refreshes
at every submission (quota_latest.json, via the storage proof) and the controller log. Every 5 minutes it appends
a line to afs_watch.jsonl and exits (so the operator is notified) when
  * the latest reading shows free space < ALERT_GIB (the campaign refuses new jobs below 200 GiB), or
  * the controller logged >= 3 new controller_error lines since the previous check, or
  * the campaign finished (outcome.json).
"""
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
ALERT_GIB = 230
GIB = 1024**3


def errors():
    try:
        return sum(1 for line in (HERE / 'controller.log').open(errors='replace') if line.startswith('controller_error'))
    except OSError:
        return 0


def main():
    seen = errors()
    started = time.time()
    while time.time() - started < 12 * 3600:
        reading = {'at': time.time()}
        try:
            quota = json.loads((HERE / 'quota_latest.json').read_text())
            row = quota['rows'][0]
            reading.update(observed_at=quota['observed_at'], used_gib=row['used_bytes'] / GIB,
                           free_gib=(row['quota_bytes'] - row['used_bytes']) / GIB)
        except (OSError, ValueError, KeyError) as error:
            reading['error'] = type(error).__name__
        now = errors()
        reading['new_controller_errors'] = now - seen
        seen = now
        with (HERE / 'afs_watch.jsonl').open('a') as log:
            log.write(json.dumps(reading) + '\n')
        print(json.dumps(reading), flush=True)
        if reading.get('free_gib') is not None and reading['free_gib'] < ALERT_GIB:
            print('ALERT afs free space below', ALERT_GIB, 'GiB', flush=True)
            return
        if reading['new_controller_errors'] >= 3:
            print('ALERT repeated controller errors', flush=True)
            return
        if (HERE / 'outcome.json').exists():
            print('campaign finished', flush=True)
            return
        time.sleep(300)


if __name__ == '__main__':
    main()
