"""Read existing failed-run logs only; no model, simulator, or GPU use."""
from collections import deque
import hashlib
import json
from pathlib import Path
import re
import urllib.request


def collect(path):
    if not path.exists():
        return {'exists': False}
    pattern = re.compile(r'fatal|segmentation|segfault|aborted|assertion|traceback|cuda error|out of memory|failed to allocate|exception', re.I)
    context = deque(maxlen=35)
    selected = []
    following = 0
    digest = hashlib.sha256()
    lines = 0
    with path.open('rb') as stream:
        for raw in stream:
            digest.update(raw)
            lines += 1
            line = raw.decode(errors='replace')[:2500].rstrip()
            if pattern.search(line):
                if not following:
                    selected.extend(list(context))
                following = 65
            if following:
                selected.append((lines, line))
                following -= 1
            context.append((lines, line))
    # Preserve the last critical windows if startup emits many benign errors.
    return dict(exists=True, bytes=path.stat().st_size, sha256=digest.hexdigest(),
                lines=lines, error_context=selected[-320:], tail=list(context)[-15:])


def run(url):
    root=Path('/mnt/afs/260010165/behavior_1k/update_composition_20261022/trial_h100/meta/r02')
    episode=root/'episodes/iteration_000_episode_0'
    paths=[episode/'simulator.log',root/'renderer/worker.log']
    paths += sorted((episode/'appdata/local/logs/Kit/OmniGibson').glob('*/*.log'))[-2:]
    result={'scope':'Existing failed-run logs only; no simulation or training',
            'logs':{str(p.relative_to(root)):collect(p) for p in paths}}
    data=json.dumps(result).encode()
    request=urllib.request.Request(url,data=data,method='PUT')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request,timeout=45) as response:
        assert response.status in (200,201,204)
    print(json.dumps({'uploaded_result_bytes':len(data),'files':len(paths)}),flush=True)
