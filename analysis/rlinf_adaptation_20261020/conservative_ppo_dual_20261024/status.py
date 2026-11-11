"""Safe status for this exact two-worker run; never expose startup commands."""
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client


def status():
    intent = json.loads((HERE / 'submission_intent.json').read_text())
    client = aoss_client()
    result = {'workers': []}
    try:
        for worker in intent['workers']:
            reply = subprocess.run([str(Path.home() / '.local/bin/sco'), 'acp', 'jobs', 'describe', worker['job_name'],
                                    '--workspace-name=share-space', '-o', 'json'], capture_output=True, text=True, check=True, timeout=45)
            actual = json.loads(reply.stdout)
            assert actual['uid'] == worker['uid'] and actual['ownership']['user_name'] == '260010165'
            row = {k: actual.get(k) for k in ('name', 'uid', 'state', 'start_time', 'complete_time')}
            row.update(rank=worker['rank'], quota_type=worker['quota_type'])
            try:
                raw = client.get_object(Bucket='260010165', Key=f"behavior1k/rlinf-conservative-20261024/dual01/status_rank{worker['rank']}.json")['Body'].read(65537)
                assert len(raw) < 65537
                row['progress'] = json.loads(raw)
            except client.exceptions.NoSuchKey:
                row['progress'] = None
            result['workers'].append(row)
    finally:
        client.close()
    (HERE / 'latest_status.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    print(json.dumps(status(), ensure_ascii=False))
