"""Read the exact active pilot job and bounded result; never submit/restart."""
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    directory = HERE / 'gpu_probe_job'
    intent = json.loads((directory / 'submission_intent.json').read_text())
    response = subprocess.run([str(Path.home() / '.local/bin/sco'), 'acp', 'jobs', 'describe',
                               intent['job_name'], '--workspace-name=share-space', '-o', 'json'],
                              capture_output=True, text=True, timeout=45)
    if response.returncode:
        raise RuntimeError(f'Job observation failed ({response.returncode}); terminal state is unknown')
    raw = json.loads(response.stdout)
    assert raw['uid'] == intent['uid']
    current = {k: raw.get(k) for k in ['uid', 'name', 'state', 'create_time', 'start_time', 'complete_time']}
    (directory / 'current_status.json').write_text(json.dumps(current, indent=2) + '\n')
    sys.path.insert(0, str(ROOT / 'BEHAVIOR-VLAs/sco'))
    from video_eval_adapter import aoss_client
    from botocore.exceptions import ClientError
    client = aoss_client()
    result_status = {'available': False}
    try:
        key = json.loads((directory / 'artifacts.json').read_text())['result_key']
        try:
            result = client.get_object(Bucket='260010165', Key=key)
            try:
                body = result['Body'].read(128 * 1024 + 1)
            finally:
                result['Body'].close()
            assert len(body) <= 128 * 1024
            value = json.loads(body)
            (directory / 'result.json').write_bytes(body)
            result_status = {'available': True, **{k: value.get(k) for k in
                              ['passed', 'worker_returncode', 'gpu', 'completed_updates', 'action_expert_attention_max_update']}}
            if not value.get('passed'):
                (directory / 'failure.log').write_text(value.get('log_tail', ''))
        except ClientError as error:
            code = error.response.get('Error', {}).get('Code')
            if code not in ('NoSuchKey', 'NotFound', '404'):
                raise
    finally:
        client.close()
    image = json.loads((HERE / 'vla_pilot/image_progress.json').read_text())
    submissions = []
    for path in (HERE / 'vla_pilot').glob('submission_*/submission_intent.json'):
        item = json.loads(path.read_text())
        if item.get('submission_attempted'):
            submissions.append({k: item.get(k) for k in ('job_name', 'uid', 'state')})
    staging = {'complete': False, 'verified_objects': 0}
    stage_receipt = HERE / 'vla_pilot/data_stage_receipt.json'
    stage_progress = HERE / 'vla_pilot/data_stage_progress.json'
    if stage_receipt.exists():
        saved = json.loads(stage_receipt.read_text())
        staging = {'complete': saved['complete'], 'verified_objects': len(saved['objects'])}
    elif stage_progress.exists():
        try:
            saved = json.loads(stage_progress.read_text())
            staging['verified_objects'] = len(saved)
        except json.JSONDecodeError:
            staging['reading_progress'] = True
    print(json.dumps({'gpu_job': current, 'gpu_result': result_status,
                      'images': {'status': image['status'], 'completed': image.get('completed_images', 0),
                                 'total': 16344, 'completed_video_objects': len(image['completed_video_objects']),
                                 'total_video_objects': 23, 'bytes_reserved': image['transfer_bytes_reserved']},
                      'data_staging': staging, 'formal_training_submissions': submissions}, indent=2))


if __name__ == '__main__':
    main()
