"""Run the CPU suite against an AST-minified copy of the runtime (what the learner bundle executes)."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from deployment import minify_source

with tempfile.TemporaryDirectory() as root:
    target = Path(root) / 'runtime'
    for path in (HERE.parent / 'runtime').rglob('*.py'):
        if '__pycache__' in path.parts:
            continue
        out = target / path.relative_to(HERE.parent / 'runtime')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(minify_source(path.read_text()))
    tests = Path(root) / 'tests'
    shutil.copytree(HERE, tests, ignore=shutil.ignore_patterns('__pycache__', '*.json'))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(target), os.environ['UPSTREAM']]))
    result = subprocess.run([sys.executable, '-m', 'unittest', 'test_fast_learner', 'test_lr_scan', 'test_prefix_cache', '-v'],
                            cwd=tests, env=env, capture_output=True, text=True)
    print(result.stderr[-1500:])
    (HERE / 'minified_tests_result.json').write_text(json.dumps({'returncode': result.returncode,
        'passed': result.returncode == 0, 'tail': result.stderr[-600:]}, indent=2) + '\n')
    sys.exit(result.returncode)
