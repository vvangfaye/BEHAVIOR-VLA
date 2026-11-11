"""Read only the exact two submitted roles and bounded private result objects."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from submit import cli, write

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True,
                        choices=('meta','maml','fixed_crossplay','rl_bc','before_after'))
    parser.add_argument('--run-id',default='')
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT/'BEHAVIOR-VLAs/sco'))
    from video_eval_adapter import aoss_client
    from botocore.exceptions import ClientError
    client = aoss_client()
    try:
        for role in ('renderer', 'trainer'):
            work = HERE/'split_trial_submissions'/args.method/role
            if args.run_id:work=HERE/'split_trial_submissions'/args.method/args.run_id/role
            if not (work/'submission.json').exists():
                print(json.dumps(dict(role=role,state='NOT_SUBMITTED')),flush=True)
                continue
            intent = json.loads((work/'submission.json').read_text())
            response = cli('acp', 'jobs', 'describe', intent['job_name'],
                           '--workspace-name=share-space', '-o', 'json')
            if response.returncode:
                raise RuntimeError('Read failed; no terminal state inferred for '+role)
            job = json.loads(response.stdout)
            assert job['uid'] == intent['uid']
            status = {k:job.get(k) for k in ('name','uid','state','start_time','complete_time')}
            status.update(role=role, observed_at=datetime.now(timezone.utc).isoformat(),
                          resource_pool=job.get('resource_pool',{}).get('name'))
            write(work/'job_status.json', status)
            print(json.dumps(status), flush=True)
            for name in ('progress.json', 'result.json'):
                try:
                    item = client.get_object(Bucket=intent['bucket'], Key=intent['prefix']+name)
                except ClientError as error:
                    if error.response['Error']['Code'] in ('NoSuchKey','404','NotFound'):
                        continue
                    raise
                try:
                    raw = item['Body'].read(512*1024+1)
                finally:
                    item['Body'].close()
                if len(raw) > 512*1024:
                    raise ValueError('Result object exceeds read limit')
                value = json.loads(raw)
                write(work/name, value)
                keys=('stage','iteration','slot','task','instance','episode','complete',
                      'training_episodes','rendered_episodes','worker_returncode',
                      'elapsed_seconds','error_type','error','gpu_description')
                print(json.dumps(dict(role=role, artifact=name,
                                      **{k:value[k] for k in keys if k in value})),flush=True)
    finally:
        client.close()


if __name__=='__main__':
    main()
