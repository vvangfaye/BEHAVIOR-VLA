"""Read only the two recorded job identities and their small progress objects."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client


def status():
    rows = []
    client = aoss_client()
    try:
        for rank in (0, 1):
            receipt = json.loads((HERE / f'r{rank}/submission.json').read_text())
            response = subprocess.run([str(Path.home() / '.local/bin/sco'), 'acp', 'jobs', 'describe',
                receipt['job'], '--workspace-name=share-space', '-o', 'json'],
                capture_output=True, text=True, check=True, timeout=45)
            job = json.loads(response.stdout)
            assert job['uid'] == receipt['uid'] and job['ownership']['user_name'] == '260010165'
            row = {'rank': rank, 'name': receipt['job'], 'uid': receipt['uid'], 'state': job['state'],
                   'quota': job['scheduling']['quota_type']}
            try:
                data = client.get_object(Bucket='260010165', Key=receipt['aoss_prefix']+f'status_rank{rank}.json')
                row['progress'] = json.loads(data['Body'].read())
            except client.exceptions.ClientError as error:
                if error.response['Error']['Code'] not in ('NoSuchKey', '404'):
                    raise
            rows.append(row)
    finally:
        client.close()
    result = {'observed_at': datetime.now(timezone.utc).isoformat(), 'workers': rows}
    temporary = HERE / f'latest_status.{os.getpid()}.tmp'
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(HERE / 'latest_status.json')
    active_path = HERE.parent.parent / 'active_run.json'
    if active_path.exists():
        active = json.loads(active_path.read_text())
        if active.get('experiment') == 'sampling32_ppo_20261025':
            active.update(observed_at=result['observed_at'], workers=rows,
                          status='complete' if all(r['state']=='SUCCEEDED' for r in rows) else 'running')
            if any(r['state'] in ('FAILED','SUSPENDED','TERMINATED','CANCELED','STOPPED') for r in rows):
                active['status']='interrupted'
            tmp=active_path.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(active,ensure_ascii=False,indent=2)+'\n');tmp.replace(active_path)
    return result


if __name__ == '__main__':
    print(json.dumps(status(), ensure_ascii=False))
