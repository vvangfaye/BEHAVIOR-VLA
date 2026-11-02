"""Deduplicate job state transitions without weakening pagination checks."""
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    for root in [HERE.parent / 'rl_combined_20261102', HERE / 'trash', HERE / 'modem']:
        path = root / 'controller.py'
        source = path.read_text()
        if 'seen_in_this_state = set()' in source:
            continue
        assert source.count('for state in states:\n        for page') == 1
        source = source.replace('for state in states:\n        for page',
                                'for state in states:\n        seen_in_this_state = set()\n        for page')
        source = source.replace("assert job['uid'] not in jobs, 'Repeated page; do not admit extra work'",
                                "assert job['uid'] not in seen_in_this_state, 'Repeated page within state; no admission'\n                seen_in_this_state.add(job['uid'])")
        source = source.replace("jobs[job['uid']] = {'state': state, 'gpus': count}",
                                "# A job may move from CREATING to RUNNING between state queries.\n                # Count that UID once, conservatively using its largest observed request.\n                count = max(count, jobs.get(job['uid'], {}).get('gpus', 0))\n                jobs[job['uid']] = {'state': state, 'gpus': count}")
        compile(source, str(path), 'exec')
        path.write_text(source)
        print('State-transition-safe admission:', root.name)


if __name__ == '__main__':
    main()
