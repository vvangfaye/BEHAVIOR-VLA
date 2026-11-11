"""Use the already installed local simulator without changing shared files."""
import hashlib
import json
import os
from pathlib import Path


def source_preflight(plan, full_baseline=False):
    for raw, expected in plan['native_source_sha256'].items():
        path = Path(raw)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Native source changed: ' + raw)
    metadata = Path(plan['dataset_root']) / '2026-challenge-task-instances/metadata/task.jsonl'
    raw = metadata.read_bytes()
    tasks = {row['task_name']: row for line in raw.decode().splitlines() if line.strip()
             for row in [json.loads(line)]}
    return {'task_metadata_sha256': hashlib.sha256(raw).hexdigest(),
            'native_task_budgets': {task: int(1.5 * tasks[task]['length']) for task, _ in plan['pairs']}}


def build_env(plan, cache, vk, egl, library_path, *, process):
    env = dict(os.environ)
    for key in list(env):
        if (key.startswith(('PI05_', 'SUBTASK_', 'TEACHER_', 'EXTERNAL_SUBTASK', 'PROVIDED_SUBTASK'))
                or key in {'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'PYTHONPATH', 'PYTHONHOME',
                           'PYTHONSTARTUP', 'VIRTUAL_ENV', 'MAX_STEPS', 'OMNIGIBSON_NO_OMNIVERSE'}):
            env.pop(key)
    env.update(PYTHONNOUSERSITE='1', PYTHONHASHSEED='0', CUDA_VISIBLE_DEVICES='0',
               OMNI_KIT_ACCEPT_EULA='YES', OMNIGIBSON_HEADLESS='1', OMNIGIBSON_GPU_ID='0',
               OMNIGIBSON_DISABLE_TORCH_COMPILE='1',
               OMNIGIBSON_DATA_PATH=plan['dataset_root'],
               OPENPI_DATA_HOME=plan['openpi_cache'],
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
               VK_ICD_FILENAMES=str(vk), __EGL_VENDOR_LIBRARY_FILENAMES=str(egl),
               HF_HUB_OFFLINE='1')
    code = Path(plan['code'])
    if process == 'policy':
        env['PYTHONPATH'] = ':'.join([str(code / 'openpi/src'), str(code / 'openpi/packages/openpi-client/src')])
    elif process == 'evaluator':
        env.update(OMNIGIBSON_APPDATA_PATH=str(cache / 'appdata'), XDG_CACHE_HOME=str(cache / 'cache'),
                   XDG_CONFIG_HOME=str(cache / 'config'), WARP_CACHE_PATH=str(cache / 'warp'),
                   TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'), TRITON_CACHE_DIR=str(cache / 'triton'),
                   TMPDIR=str(cache),
                   LD_LIBRARY_PATH=str(Path(plan['eval_python']).parent.parent / 'lib') + ':/usr/lib/x86_64-linux-gnu')
        # The existing environment's pinned .pth selects the verified frozen
        # evaluator. Do not edit it or replace it with the mutable worktree.
        env['PYTHONPATH'] = ''
    else:
        raise ValueError('Unknown process environment')
    return env
