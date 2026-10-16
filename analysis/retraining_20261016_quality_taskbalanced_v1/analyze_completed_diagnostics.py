"""Collect bounded exit diagnostics for accepted episodes; never submit or alter jobs."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from observe_five_batches import CAMPAIGN, controller


def summarize(report, diagnostic):
    labels = re.findall(r'Generated subtask: ([^\r\n]+)', diagnostic.get('logs', {}).get('policy', ''))
    longest = current = 0
    previous = None
    for label in labels:
        current = current + 1 if label == previous else 1
        longest = max(longest, current)
        previous = label
    return {'id': report['id'], 'record': report['record'],
            'worker_exit_code': diagnostic.get('child_returncode'),
            'policy_predictions_in_tail': len(labels),
            'most_common_predictions': Counter(labels).most_common(5),
            'longest_repeated_prediction_run_in_tail': longest,
            'evidence_scope': 'last 32 KiB of each worker log; not complete rollout',
            'cause_established': False}


def main():
    client = controller.aoss_client()
    plan = controller.load(CAMPAIGN/'plan.json')
    folder = CAMPAIGN/'failure_analysis'
    folder.mkdir(exist_ok=True)
    rows = []
    for path in sorted((CAMPAIGN/'reports').glob('candidate-*.json')):
        report = json.loads(path.read_text())
        if report.get('state') != 'accepted':
            continue
        key = plan['object_prefix']+'/telemetry/'+report['id']+'/diagnostics.json'
        response = client.get_object(Bucket='260010165', Key=key)
        try:
            data = response['Body'].read(1024*1024+1)
        finally:
            response['Body'].close()
        if len(data)>1024*1024:
            raise ValueError('Unexpected diagnostic size')
        diagnostic = json.loads(data)
        if diagnostic.get('evaluation_id') != report['id'] or diagnostic.get('checkpoint_step') != 1000:
            raise ValueError('Diagnostic identity differs from accepted episode')
        if report['checkpoint_manifest_sha256'] != plan['expected_checkpoint_manifest_sha256']:
            raise ValueError('Unexpected model identity')
        (folder/(report['id']+'.diagnostics.json')).write_text(json.dumps(diagnostic,indent=2)+'\n')
        row = summarize(report,diagnostic)
        row['diagnostics_sha256']=hashlib.sha256(data).hexdigest()
        rows.append(row)
    output = {'episodes':rows,'cloud_read_only':True,'model_improvement_proven':False,
              'warning':'Repeated text does not alone establish action failure or incorrect labels.'}
    (folder/'completed_summary.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output))


if __name__=='__main__':
    main()
