"""Read submitted native pilot jobs and bounded receipts, without mutations."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def inspect(path):
    intent = json.loads(path.read_text())
    name = intent['job_name']
    process = subprocess.run([str(Path.home() / '.local/bin/sco'), 'acp', 'jobs', 'describe', name,
                              '--workspace-name=share-space', '-o', 'json'],
                             capture_output=True, text=True, timeout=45)
    if process.returncode:
        return {'job_name': name, 'observation_error': True, 'terminal_state_unknown': True}
    job = json.loads(process.stdout)
    if intent.get('uid') and job['uid'] != intent['uid']:
        raise ValueError('Exact job UID differs')
    state = {k: job.get(k) for k in ('uid', 'name', 'state', 'create_time', 'start_time', 'complete_time')}
    (path.parent / 'current_status.json').write_text(json.dumps(state, indent=2) + '\n')
    from video_eval_adapter import aoss_client
    from botocore.exceptions import ClientError
    client = aoss_client()
    outcome = {'available': False}
    try:
        try:
            response = client.get_object(Bucket='260010165',
                Key='behavior1k/action-support-20261021/closed-loop-v1/' + name + '/result.json')
            try:
                raw = response['Body'].read(256 * 1024 + 1)
            finally:
                response['Body'].close()
            if len(raw) > 256 * 1024:
                raise ValueError('Unexpectedly large episode receipt')
            result = json.loads(raw)
            (path.parent / 'result.json').write_bytes(raw)
            outcome = {'available': True, **{k: result.get(k) for k in
                                            ('accepted', 'worker_returncode', 'record', 'error_type')}}
        except ClientError as error:
            if error.response.get('Error', {}).get('Code') not in ('NoSuchKey', 'NotFound', '404'):
                raise
    finally:
        client.close()
    return {'job': state, 'result': outcome}


def main():
    sys.path.insert(0, str(ROOT / 'BEHAVIOR-VLAs/sco'))
    paths = [p for p in sorted((HERE / 'closed_loop_v1').glob('*/submission_intent.json'))
             if json.loads(p.read_text()).get('submission_attempted')]
    with ThreadPoolExecutor(max_workers=4) as executor:
        print(json.dumps(list(executor.map(inspect, paths)), indent=2))


if __name__ == '__main__':
    main()
