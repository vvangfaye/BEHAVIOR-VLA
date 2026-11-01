"""Finish the restore index and verify the existing water controller resumes."""
import fcntl
import json
import os
from pathlib import Path
import signal
import time

from control import HERE, CAMPAIGN, save
from finalize_archive import main as finalize_archive


def main():
    handle = (HERE/'follow_through.lock').open('a')
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    expected = {x['index'] for x in json.loads((HERE/'plan.json').read_text())['files']}
    deadline = time.time() + 6*3600
    while time.time() < deadline:
        try:
            released = {}
            for folder in [HERE] + sorted(HERE.glob('retry_a*')):
                file = folder/'archive_status.json'
                if not file.exists():
                    continue
                for item in json.loads(file.read_text()).get('released', []):
                    assert item['index'] not in released, 'Duplicate source release'
                    released[item['index']] = item
            summary = {'at': time.time(), 'released_files': len(released),
                       'expected_files': len(expected),
                       'released_allocated_bytes': sum(x['allocated_bytes'] for x in released.values())}
            if set(released) == expected:
                if not (HERE/'archive_complete.json').exists():
                    finalize_archive()
                summary['archive_complete'] = True
            water = json.loads((CAMPAIGN/'water/latest_status.json').read_text())
            proof = json.loads((CAMPAIGN/'water/training_storage_clearance.json').read_text())
            workers = [w for w in water.get('workers', []) if w['phase'] != 'baseline']
            summary.update(free_bytes=proof['quota_lower_bound_bytes']-proof['account_allocated_bytes'],
                           storage_passed=proof['passed'], water_state=water.get('state'),
                           water_phase=water.get('next_phase'),
                           water_workers=[{k:w.get(k) for k in ['job','uid','phase','state','progress']} for w in workers])
            save(HERE/'combined_status.json', summary)
            if summary.get('archive_complete') and proof['passed'] and any(w['state']=='RUNNING' for w in workers):
                save(HERE/'water_resumed.json', summary)
                # Only retire this archive's local monitors; the water controller continues.
                for folder in [HERE] + sorted(HERE.glob('retry_a*')):
                    record = folder/'monitor_process.json'
                    if not record.exists():
                        continue
                    pid = json.loads(record.read_text())['pid']
                    proc = Path('/proc')/str(pid)/'cmdline'
                    if proc.exists():
                        command = proc.read_bytes().replace(b'\0', b' ').decode()
                        if str(folder/'control.py') in command and 'monitor' in command:
                            os.kill(pid, signal.SIGTERM)
                return
        except Exception as error:
            save(HERE/'follow_through_error.json', {'at':time.time(),'type':type(error).__name__,
                                                   'message':str(error)[:500]})
        time.sleep(30)
    raise TimeoutError('Archive follow-through deadline')


if __name__ == '__main__':
    for key in list(os.environ):
        if key.lower() in {'http_proxy', 'https_proxy', 'all_proxy'}:
            os.environ.pop(key)
    main()
