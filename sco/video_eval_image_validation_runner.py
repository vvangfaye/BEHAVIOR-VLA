"""Execute two finite serving-image probes; preserve evidence outside image paths."""
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request


def run(payload):
    root = Path(payload['output'])
    root.mkdir(parents=True, exist_ok=False)
    scripts = root / 'probe_code'
    scripts.mkdir()
    for name, source in payload['sources'].items():
        raw = base64.b64decode(source['data'])
        assert hashlib.sha256(raw).hexdigest() == source['sha256']
        (scripts / name).write_bytes(raw)
    report = dict(kind='serving_image_baseline_and_budget_v1', image_reference=payload['image_reference'],
                  source_image_id=payload['image_id'], image_config_digest=payload['config_digest'],
                  physical_24gb_verified=False, source_manifest_sha256=payload['source_manifest_sha256'],
                  checkpoint_ready_sha256=payload['checkpoint_ready_sha256'], probes={})
    result = 1
    try:
        for label, extra in [('baseline', []), ('budget_23000mib', ['--memory-budget-mib', '23000'])]:
            command = [sys.executable, str(scripts / 'video_eval_image_gpu_probe.py'),
                       '--output', str(root / label), '--image-reference', payload['image_reference'],
                       '--image-id', payload['image_id'], '--source-manifest-sha256', payload['source_manifest_sha256'],
                       '--checkpoint-ready-sha256', payload['checkpoint_ready_sha256'], '--requests', '32'] + extra
            with (root / (label + '.log')).open('w') as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=3500)
            probe = root / label / 'probe.json'
            report['probes'][label] = dict(exit_code=process.returncode,
                report=json.loads(probe.read_text()) if probe.exists() else None,
                log_tail=(root / (label + '.log')).read_text(errors='replace')[-4000:])
            if process.returncode:
                break
        result = int(len(report['probes']) != 2 or any(x['exit_code'] for x in report['probes'].values()))
    except Exception as error:
        report['error_type'] = type(error).__name__
    finally:
        report['exit_code'] = result
        data = json.dumps(report, indent=2).encode()
        (root / 'report.json').write_bytes(data)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(urllib.request.Request(payload['report_put'], data=data, method='PUT',
                                               headers={'Content-Type': 'application/json'}), timeout=90) as response:
            assert response.status in (200, 201)
        print(json.dumps({'validation_exit_code': result, 'report_sha256': hashlib.sha256(data).hexdigest(),
                          'report_bytes': len(data), 'physical_24gb_verified': False}), flush=True)
    return result
