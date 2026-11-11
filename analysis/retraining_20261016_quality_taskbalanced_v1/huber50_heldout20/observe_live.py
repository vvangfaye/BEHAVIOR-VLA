"""Read-only current worker and artifact observations for the holdout campaign."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from controller import ROOT, base, observe

TERMINAL, load, save = base.TERMINAL, base.load, base.save
transport, tracked_intents = base.transport, base.tracked_intents


def main():
    plan = load(ROOT / 'plan.json')
    client = transport.aoss_client()
    previous = load(ROOT / 'current_live.json') if (ROOT / 'current_live.json').exists() else {'rows': []}
    prior = {r['id']: r for r in previous['rows']}
    intents = tracked_intents(plan)
    with ThreadPoolExecutor(max_workers=6) as pool:
        states = list(pool.map(observe, intents))

    def read(key):
        try:
            response = client.get_object(Bucket='260010165', Key=plan['object_prefix'] + '/' + key)
        except Exception as exc:
            if getattr(exc, 'response', {}).get('Error', {}).get('Code') in ('NoSuchKey', '404'):
                return None
            raise
        try:
            return json.loads(response['Body'].read(1024 * 1024))
        finally:
            response['Body'].close()

    rows = []
    for intent, state in zip(intents, states):
        if state['state'] in TERMINAL:
            continue
        batch = read('reports/' + intent['id'] + '.json') or {}
        member = batch.get('current_member')
        heart = read('telemetry/' + member + '/heartbeat.json') if member else None
        row = {'id': intent['id'], 'uid': state['uid'], 'state': state['state'], 'member': member}
        if heart:
            row.update(heartbeat_age=round(time.time()-heart['observed_at']),
                       elapsed_seconds=round(time.time()-heart['started_at']),
                       child_returncode=heart['child_returncode'], files=heart['episode_files'])
            old = prior.get(intent['id'], {})
            if old.get('member') != member:
                old = {}
            row['files_changed'] = old.get('files') != row['files']
            row['last_file_change_at'] = time.time() if row['files_changed'] else old.get('last_file_change_at', time.time())
            row['unchanged_file_seconds'] = round(time.time()-row['last_file_change_at'])
            row['video_bytes'] = sum((value.get('bytes') or 0) for name, value in row['files'].items()
                                     if name.endswith('.mp4'))
            row['video_started'] = row['video_bytes'] > 0
            row['video_changed'] = old.get('video_bytes') != row['video_bytes']
            row['last_video_change_at'] = (time.time() if row['video_changed'] else
                                           old.get('last_video_change_at', time.time()))
            row['unchanged_video_seconds'] = round(time.time() - row['last_video_change_at'])
        rows.append(row)
    result = {'observed_at': time.time(), 'rows': rows, 'read_only': True}
    status_path = ROOT / 'status.json'
    if status_path.exists():
        status = load(status_path)
        result['collector_status_age_seconds'] = round(time.time() - status['observed_at'])
        result['collector_stale'] = result['collector_status_age_seconds'] > 180 and not status.get('complete')
        result['accepted'] = status['accepted']
    save(ROOT / 'current_live.json', result)
    # Preserve bounded numeric progress across observations for stall diagnosis.
    history = {'observed_at': result['observed_at'], 'accepted': result.get('accepted'),
               'collector_stale': result.get('collector_stale'),
               'members': [{key: row.get(key) for key in
                            ('id', 'uid', 'state', 'member', 'elapsed_seconds', 'video_bytes',
                             'unchanged_video_seconds', 'child_returncode')} for row in rows]}
    with (ROOT / 'progress_observations.jsonl').open('a') as stream:
        stream.write(json.dumps(history) + '\n')
    print(json.dumps({'observed_at': result['observed_at'],
                      'accepted': result.get('accepted'),
                      'collector_status_age_seconds': result.get('collector_status_age_seconds'),
                      'collector_stale': result.get('collector_stale'),
                      'rows': [{k: v for k, v in r.items() if k != 'files'} for r in rows]}))


if __name__ == '__main__':
    main()
