"""Exercise shell interpreter/source selection without model or GPU execution."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("pilot", [True, False])
def test_training_preflight_uses_opt_in_runtime_without_changing_legacy_activation(tmp_path, pilot):
    code = tmp_path / "new-code"
    openpi = code / "openpi"
    openpi.mkdir(parents=True)
    runtime = tmp_path / "existing-pi05"
    (runtime / "bin").mkdir(parents=True)
    if not pilot:
        (openpi / '.venv').symlink_to(runtime, target_is_directory=True)
    # Model the restored conda activation hook, including its PYTHONPATH reset.
    (runtime / "bin/activate").write_text(
        "unset PYTHONPATH\nexport PATH=" + shlex.quote(str(runtime / "bin")) + ':"$PATH"\n')
    interpreter = runtime / "bin/python"
    interpreter.write_text(f"#!{sys.executable}\n" + r'''
import json, os, pathlib, sys
pathlib.Path(os.environ['SELECTION_RECEIPT']).write_text(json.dumps({
    'argv': sys.argv, 'pythonpath': os.environ.get('PYTHONPATH'),
    'path_head': os.environ['PATH'].split(':')[0], 'cwd': os.getcwd(),
}))
args = sys.argv
pathlib.Path(args[args.index('--write-episodes-file') + 1]).write_text('0\n1\n')
pathlib.Path(args[args.index('--report-file') + 1]).write_text('{}\n')
''')
    interpreter.chmod(0o700)
    dataset = tmp_path / "mixed-data"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text('{}\n')
    receipt = tmp_path / "selection.json"
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('TRAIN_', 'VAL_', 'EXPECTED_', 'FRAMESAMP_', 'MEMORY_', 'OPENPI_', 'SUBTASK_', 'DEBUGPY_'))}
    env.update(CODE_ROOT=str(REPO), OPENPI_DIR=str(openpi),
               PI05_ENV=str(runtime if pilot else tmp_path / 'ignored-external-env'),
               CODEX_TRAINING_RUNTIME='1' if pilot else '0',
               DATASET_ROOT=str(dataset), PREFLIGHT_OUTPUT_DIR=str(tmp_path / "preflight"),
               NUM_GPUS='4', CUDA_DEVICES='0,1,2,3', BATCH_SIZE='4', EXP_NAME='isolated-preflight',
               TRAIN_EPOCHS='', EPISODES_FILE='', PYTHONPATH='/stale/runtime/code',
               SELECTION_RECEIPT=str(receipt))
    completed = subprocess.run(['/bin/bash', str(REPO / 'scripts/train_pi05_subtask_b1k.sh'), 'check'],
                               env=env, capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
    selected = json.loads(receipt.read_text())
    assert selected['path_head'] == str(runtime / 'bin')
    assert selected['cwd'] == str(openpi)
    assert selected['pythonpath'] == (':'.join([str(openpi / 'src'), str(openpi / 'packages/openpi-client/src')])
                                      if pilot else None)
    assert selected['argv'][1] == 'scripts/b1k/check_subtask_data.py'
    assert (openpi / '.venv').exists() is not pilot


@pytest.mark.parametrize('resume,decay,budget,expected', [
    ('1','1688978','3377956','1688978'),
    ('1','20000','40000','20000'),
    ('0','','20000','20000'),
])
def test_nebula_budget_does_not_stretch_explicit_lr_decay(resume,decay,budget,expected):
    text=(REPO/'nebula/train_pi05_multihost.sh').read_text()
    block=text[text.index('if [[ -n "${LR_DECAY_STEPS:-}" ]]'):text.index('TRAIN_EXPECTED_TASKS="$EXPECTED_TASKS"')]
    program='fail() { echo "$*" >&2; exit 2; }\n'+block+'\nprintf "%s" "$LR_DECAY_STEPS_VALUE"\n'
    env=dict(os.environ,RESUME=resume,LR_DECAY_STEPS=decay,TRAIN_STEPS=budget)
    r=subprocess.run(['/bin/bash','-c',program],env=env,capture_output=True,text=True,check=True)
    assert r.stdout==expected


def test_nebula_resume_rejects_unspecified_previous_lr_decay():
    text=(REPO/'nebula/train_pi05_multihost.sh').read_text()
    block=text[text.index('if [[ -n "${LR_DECAY_STEPS:-}" ]]'):text.index('TRAIN_EXPECTED_TASKS="$EXPECTED_TASKS"')]
    env=dict(os.environ,RESUME='1',TRAIN_STEPS='3377956');env.pop('LR_DECAY_STEPS',None)
    r=subprocess.run(['/bin/bash','-c','fail() { echo "$*" >&2; exit 2; };\n'+block],env=env,capture_output=True,text=True)
    assert r.returncode==2 and 'explicit LR_DECAY_STEPS' in r.stderr
