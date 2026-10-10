"""Bounded personal AFS discovery; never search the tenant filesystem."""
import json, os, shutil
from pathlib import Path
root = Path('/mnt/afs/260010165')
assert root.is_dir(), 'personal AFS root unavailable'
out = root / 'behavior_1k' / 'migration_20261007'
out.mkdir(parents=True, exist_ok=True)
probe = out / 'write_probe'
probe.write_text('afs write verification\n')
assert probe.read_text() == 'afs write verification\n'
probe.unlink()
report = {'personal_root': str(root), 'uid': os.getuid(), 'gid': os.getgid(), 'writable': True,
          'disk_usage': dict(zip(['total','used','free'], shutil.disk_usage(root))), 'directories': {}}
for name in ['code', 'runtime', 'datasets', 'checkpoints', 'behavior_1k']:
    p = root / name
    report['directories'][name] = sorted(x.name for x in p.iterdir()) if p.is_dir() else None
report['scope'] = 'only direct children of selected personal directories; no environment or credential contents'
(out / 'storage_preflight.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
