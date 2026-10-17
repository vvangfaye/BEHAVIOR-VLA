"""Admit the one prepared recovery only after the original worker is absent."""
import json
from pathlib import Path
import shlex

import controller

ROOT = Path(__file__).resolve().parent
RECOVERY = ROOT / 'recovery_g1'


def main():
    manifest = controller.load(RECOVERY / 'manifest.json')
    plan = controller.load(ROOT / 'plan.json')
    controller.verify_local(plan)
    if controller.sha((ROOT / 'plan.json').read_bytes()) != manifest['original_plan_sha256']:
        raise ValueError('Original plan changed')
    if not controller.load(RECOVERY / 'validation.json')['passed']:
        raise ValueError('Recovery was not validated')
    staged = controller.load(RECOVERY / 'staging_accepted.json')
    if staged['path'] != manifest['startup_path'] or staged['sha256'] != manifest['startup_sha256'] or not staged['readback_verified']:
        raise ValueError('Recovery staging proof differs')
    stopped = controller.load(RECOVERY / 'old_writer_stopped.json')
    if stopped['old_uid'] != manifest['old_uid'] or stopped.get('worker_absent') is not True:
        raise ValueError('Old writer not proven absent')
    old = controller.load(ROOT / 'intents' / (manifest['old_id'] + '.json'))
    state = controller.observe(old)
    if state['uid'] != manifest['old_uid'] or state['state'] not in controller.TERMINAL:
        raise ValueError('Old native job is not terminal')
    for identity, expected in manifest['preserved_reports'].items():
        if controller.sha((ROOT / 'reports' / (identity + '.json')).read_bytes()) != expected:
            raise ValueError('An accepted result changed')
    target = ROOT / 'reports/candidate-01.json'
    if target.exists() and controller.load(target).get('state') == 'accepted':
        raise ValueError('Target already completed; no recovery is needed')
    path = ROOT / 'intents' / (manifest['id'] + '.json')
    if path.exists():
        raise ValueError('Existing recovery intent; reconcile instead of resubmitting')
    states = [controller.observe(value) for value in controller.tracked_intents(plan)]
    if sum(s['state'] not in controller.TERMINAL for s in states) >= 10:
        raise ValueError('Ten-GPU concurrency cap reached')
    loader = ('from pathlib import Path;import hashlib;p=Path(' + repr(manifest['startup_path']) +
              ');b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==' + repr(manifest['startup_sha256']) +
              ';exec(compile(b,str(p),"exec"))')
    command = shlex.join(['timeout', '--signal=TERM', '--kill-after=60', str(manifest['job_walltime_seconds']),
                         plan['runtime'] + '/envs/b1k-eval/bin/python', '-c', loader])
    intent = {'id': manifest['id'], 'name': manifest['name'], 'uid': None, 'gpu': 1,
              'image': plan['image'], 'members': manifest['members'],
              'startup_sha256': manifest['startup_sha256'], 'generation': 1,
              'old_uid': manifest['old_uid'], 'creation_attempted': False}
    controller.transport.new(path, intent)
    controller.native_create(intent, command)
    print(json.dumps({'id': intent['id'], 'name': intent['name'], 'uid': intent['uid'], 'state': intent['state']}))


if __name__ == '__main__':
    main()
