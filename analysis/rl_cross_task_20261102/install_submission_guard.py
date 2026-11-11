"""Serialize submissions across the water and two cross-task controllers."""
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    folders = [HERE.parent / 'rl_combined_20261102', HERE / 'trash', HERE / 'modem']
    for root in folders:
        path = root / 'deployment.py'
        source = path.read_text()
        if 'def _submit_unlocked(' in source:
            continue
        combined = root.name == 'rl_combined_20261102'
        signature = "role, case=None, attempt=1" + ("" if combined else ", variant='balanced_gamma1_ppo'")
        call = 'role, case, attempt' + ('' if combined else ', variant')
        assert 'def submit(' + signature + '):' in source
        source = source.replace('def submit(' + signature + '):', 'def _submit_unlocked(' + signature + '):')
        source += '''

def submit(''' + signature + '''):
    import fcntl
    import controller as scheduling
    # Host-side lock only; never occupies a GPU or changes an existing writer.
    with (PROJECT / 'analysis/rl_gpu_submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        active = scheduling.active_gpu_count([])
        required = 4 if role == 'train' else 1
        if active + required > campaign()['max_total_live_gpus']:
            raise RuntimeError('Waiting for the shared 32-GPU admission limit')
        return _submit_unlocked(''' + call + ''')
'''
        path.write_text(source)
        compile(source, str(path), 'exec')
        print('Shared submission guard installed:', root.name)


if __name__ == '__main__':
    main()
