"""Read-only observation of the three existing, finite evaluation campaigns."""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_dual_evaluation import module, read, save

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'current_live_eval_check.json'
FOLDERS = ('paired_fixed20_step5000_5x4', 'expert_fixed20_step1000_5x4',
           'paired_fixed20_step5000_recovery_g1', 'paired_fixed20_step5000_stall_recovery_g1',
           'expert_fixed20_step1000_stall_recovery_g1', 'paired_fixed20_step5000_c11_recovery_g2',
           'paired_fixed20_step5000_c05_recovery_g2')


def main():
    previous = read(DEST) if DEST.exists() else {'rows': []}
    old = {(x['campaign'], x['id']): x for x in previous['rows']}
    ctl = module('observation_controller', ROOT / FOLDERS[0] / 'controller.py')
    client = ctl.aoss_client()

    def fetch(key):
        try:
            response = client.get_object(Bucket='260010165', Key=key)
        except client.exceptions.NoSuchKey:
            return None
        try:
            return json.loads(response['Body'].read(4 * 1024**2))
        finally:
            response['Body'].close()

    rows = []
    for folder in FOLDERS:
        if not (ROOT / folder / 'plan.json').exists():
            continue
        plan = read(ROOT / folder / 'plan.json')
        intents = [read(p) for p in sorted((ROOT / folder / 'intents').glob('*.json'))]
        with ThreadPoolExecutor(max_workers=5) as pool:
            states = list(pool.map(ctl.observe, intents))
        for intent, native in zip(intents, states):
            if native['state'] in ctl.TERMINAL:
                continue
            batch = fetch(plan['object_prefix'] + '/reports/' + intent['id'] + '.json') or {}
            member = intent['id'] if intent['id'].startswith('candidate-') else batch.get('current_member')
            heart = fetch(plan['object_prefix'] + '/telemetry/' + member + '/heartbeat.json') if member else None
            row = dict(campaign=folder, id=intent['id'], uid=native['uid'],
                       state=native['state'], current_member=member,
                       heartbeat_age=round(time.time()-heart['observed_at']) if heart else None,
                       child_returncode=heart.get('child_returncode') if heart else None,
                       files=heart.get('episode_files') if heart else None)
            if heart:
                row['member_elapsed_seconds'] = round(time.time()-heart['started_at'])
                row['resource'] = heart.get('resource')
            prior = old.get((folder, intent['id']))
            if prior and prior['current_member'] == member:
                row['files_changed'] = prior['files'] != row['files']
                row['comparison_interval_seconds'] = round(time.time()-previous['observed_at'])
                row['last_file_change_at'] = (time.time() if row['files_changed'] else
                    prior.get('last_file_change_at', previous['observed_at']))
            else:
                row['last_file_change_at'] = time.time()
            row['unchanged_file_seconds'] = round(time.time()-row['last_file_change_at'])
            rows.append(row)
    result = {'observed_at': time.time(), 'rows': rows, 'read_only': True}
    save(DEST, result)
    # Do not print logs, commands, URLs, credentials or process arguments.
    print(json.dumps({'observed_at': result['observed_at'], 'rows': [
        {k: v for k, v in row.items() if k not in ('resource', 'files')} for row in rows]}))


if __name__ == '__main__':
    main()
