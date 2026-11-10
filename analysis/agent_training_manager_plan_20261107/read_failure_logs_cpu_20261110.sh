#!/bin/bash
set -euo pipefail
python3 - <<'PYREAD' > /tmp/b1k_failure_audit.jsonl
from pathlib import Path
import json
root=Path('/mnt/afs/260010165/rlinf_runs_20261104')
for tag in ['pc_fixed_s3_resume11','pc_agent_s1_resume10r2']:
    directory=root/tag
    print(json.dumps({'tag':tag,'exists':directory.exists()}))
    for relative in ['bootstrap.log','termination.log','resource_health.log','train/console.log','stages_status.json']:
        path=directory/relative
        if not path.is_file():
            continue
        with path.open('rb') as handle:
            handle.seek(max(0,path.stat().st_size-65536))
            text=handle.read().decode(errors='replace')
        print(json.dumps({'tag':tag,'file':relative,'bytes':path.stat().st_size,'tail':'\n'.join(text.splitlines()[-160:])}))
PYREAD
curl -fsS --max-time 60 -X PUT --upload-file /tmp/b1k_failure_audit.jsonl "$B1K_PUT_LOG" >/dev/null
python3 - <<'PYSTATUS' > /tmp/b1k_audit_status.json
import os,json,time
print(json.dumps({'tag':os.environ['B1K_TAG'],'state':'finished_rc_0','time':time.time()}))
PYSTATUS
curl -fsS --max-time 60 -X PUT --upload-file /tmp/b1k_audit_status.json "$B1K_PUT_STATUS" >/dev/null
