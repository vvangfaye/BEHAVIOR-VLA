"""Build an isolated SERF source tree; never edit either original checkout."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'repos/SERF-VLA'
TARGET = HERE / 'runtime51/repo'
IGNORE = shutil.ignore_patterns('.git', '__pycache__', '*.pyc', '*.egg-info', '.venv')


def main():
    if TARGET.exists():
        raise RuntimeError('Existing runtime51 is immutable; inspect it instead of rebuilding over changes')
    for rel in ['src', 'scripts', 'external/openpi', 'external/co-tracker',
                'BEHAVIOR-1K/OmniGibson', 'BEHAVIOR-1K/bddl', 'BEHAVIOR-1K/joylo']:
        shutil.copytree(SOURCE / rel, TARGET / rel, ignore=IGNORE)
    # Apply the compatible portions of the official 5.1 migration first;
    # residual API changes are ported explicitly and recorded separately.
    result = subprocess.run(['patch', '--batch', '--forward', '--fuzz=2', '-p1',
                             '-i', str(HERE/'official_isaac51.patch')],
                            cwd=TARGET/'BEHAVIOR-1K', capture_output=True, text=True)
    (HERE/'official_patch_application.log').write_text(result.stdout+result.stderr)
    report = {'returncode':result.returncode,
              'rejects':[str(p.relative_to(TARGET)) for p in TARGET.rglob('*.rej')]}
    (HERE/'runtime51_patch_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
