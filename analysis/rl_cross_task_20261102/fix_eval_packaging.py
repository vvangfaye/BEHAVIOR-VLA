"""Keep training immutable; use the original inline contract for single-GPU eval."""
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    for root in [HERE.parent / 'rl_combined_20261102', HERE / 'trash', HERE / 'modem']:
        path = root / 'deployment.py'
        source = path.read_text()
        if 'bootstrap_source =' in source:
            continue
        needle = "contents['driver.py'] = contents['distributed_driver.py' if role == 'train' else 'comparison_driver.py']"
        assert source.count(needle) == 1
        source = source.replace(needle, needle + '''
    bootstrap_source = (HERE / 'runtime' / ('bootstrap.py' if role == 'train' else 'single_bootstrap.py')).read_text()
    if role == 'eval':
        # The proven one-GPU bootstrap consumes an inline bundle. Exclude unused
        # four-GPU utilities so its immutable command remains below 64 KiB.
        names = ['common.py','training.py','experiment.py','simulator.py','helpers.py','eval_utils.py',
                 'resource_watch.py','goals.py','storage_guard.py','capacity_admission.py',
                 'variant_update.py','driver.py']
        names += ['b1k_rlinf/' + n + '.py' for n in ['__init__','action_interface','factory','io','model','precision','tokenizer']]
        contents = {name: contents[name] for name in names}
''')
        needle = "bootstrap = contents['bootstrap.py' if role == 'train' else 'single_bootstrap.py'].decode()"
        assert source.count(needle) == 1
        source = source.replace(needle, '''if role == 'eval':
            startup['bundle_inline'] = base64.b64encode(bundle).decode()
            startup.pop('bundle_get')
        bootstrap = bootstrap_source''')
        compile(source, str(path), 'exec')
        path.write_text(source)
        print('Correct inline evaluation packaging:', root.name)
    path = HERE / 'modem/controller.py'
    source = path.read_text()
    needle = 'def may_retry(row):\n'
    assert source.count(needle) == 1
    source = source.replace(needle, needle + '''    marker = HERE / 'jobs' / row['id'] / 'packaging_recovery.json'
    if marker.exists():
        recovery = json.loads(marker.read_text())
        if recovery.get('old_uid') == row['uid'] and recovery.get('no_environment_controls') and recovery.get('old_worker_exited'):
            return True
''')
    needle = "tasks.extend(('eval', case, 'control') for case in config['eval_cases'])"
    assert source.count(needle) == 1
    source = source.replace(needle, '''first = config['eval_cases'][0]
                        first_key = f'i{first[0]}_s{first[1]}'
                        # This fixed case counts in the final result regardless of score.
                        canary_valid = (HERE / 'evidence/control' / first_key / 'result.json').exists()
                        control_cases = config['eval_cases'] if canary_valid else [first]
                        tasks.extend(('eval', case, 'control') for case in control_cases)''')
    compile(source, str(path), 'exec')
    path.write_text(source)


if __name__ == '__main__':
    main()
