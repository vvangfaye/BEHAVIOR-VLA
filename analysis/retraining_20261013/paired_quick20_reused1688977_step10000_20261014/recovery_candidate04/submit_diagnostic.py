"""One exact CPU-only evidence reader, with intent saved before creation."""
import importlib.util
import os
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT.parent
spec = importlib.util.spec_from_file_location('original_controller', CAMPAIGN / 'controller.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def main():
    plan = c.load(CAMPAIGN / 'plan.json')
    original = c.load(CAMPAIGN / 'intents/candidate-04.json')
    old_observation = c.observe(original)
    assert old_observation['state'] == 'FAILED'
    workers = subprocess.run([c.CLI, 'acp', 'jobs', 'get-workers', original['name'],
                              '--workspace-name=share-space'], env=c.environment(),
                             capture_output=True, text=True, check=True, timeout=30).stdout
    assert 'Deleted' in workers and 'Running' not in workers and 'Pending' not in workers
    c.save(ROOT / 'old_writer_terminal.json', dict(observation=old_observation, workers=workers))
    client = c.aoss_client()
    name = 'b1k-q20k10-1014-diag04-g0'
    key = plan['object_prefix'] + '/recovery-candidate04/diagnostic-g0.json'
    report_url = client.generate_presigned_url('put_object', Params={'Bucket': '260010165', 'Key': key}, ExpiresIn=172800)
    source = (ROOT / 'diagnostic_worker.py').read_text()
    startup = source + '\nimport urllib.request\nresult=main()\n'
    startup += "raw=json.dumps(result,ensure_ascii=False).encode()\nopener=urllib.request.build_opener(urllib.request.ProxyHandler({}))\n"
    startup += 'with opener.open(urllib.request.Request(' + repr(report_url) + ",data=raw,method='PUT'),timeout=60) as response: pass\n"
    startup += "print(json.dumps({'state':'diagnostic_uploaded','files':len(result['files'])}),flush=True)\n"
    private = ROOT / 'private'
    private.mkdir(exist_ok=True)
    startup_path = private / 'diagnostic.startup.py'
    intent_path = ROOT / 'diagnostic_intent.json'
    if intent_path.exists():
        existing = c.load(intent_path)
        obs = c.observe(existing)
        c.save(ROOT / 'diagnostic_observation.json', obs)
        print({'existing_uid': obs['uid'], 'state': obs['state']})
        return
    startup_path.write_text(startup)
    os.chmod(startup_path, 0o600)
    intent = dict(id='diagnostic-g0', name=name, mode='diagnostic_read_only', gpu=0,
                  image=plan['image'], created_intent_at=time.time(),
                  startup_sha256=c.sha_bytes(startup.encode()), uid=None, attempts=1,
                  report_key=key, creation_attempted=False, maximum_runtime_seconds=600,
                  authorized_original_uid=original['uid'], training_modified=False)
    c.new(intent_path, intent)
    command = 'timeout --signal=TERM --kill-after=30 600 ' + shlex.quote(plan['runtime'] + '/envs/b1k-eval/bin/python') + ' -c ' + shlex.quote(startup)
    args = [c.CLI, 'acp', 'jobs', 'create', '--workspace-name=share-space',
            '--aec2-name=share-cluster', '--training-framework=pytorch',
            '--worker-spec=n6ls.iu.i40.8c32g', '--worker-nodes=1', '--quota-type=spot',
            '--retry-times=0', '--name='+name, '--job-name='+name,
            '--container-image-url='+plan['image'],
            '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs', '--command='+command]
    intent.update(creation_attempted=True, creation_started_at=time.time(), state='creation_unknown')
    c.save(intent_path, intent)
    try:
        cp = subprocess.run(args, env=c.environment(), capture_output=True, text=True, timeout=90)
        intent['cli_exit_code'] = cp.returncode
    except subprocess.TimeoutExpired:
        intent['cli_outcome'] = 'timeout_unknown'
    c.save(intent_path, intent)
    obs = c.observe(intent)
    intent.update(uid=obs['uid'], state=obs['state'], verified_at=obs['observed_at'])
    c.save(intent_path, intent)
    c.save(ROOT / 'diagnostic_observation.json', obs)
    print({'uid': obs['uid'], 'state': obs['state'], 'gpu': 0})


if __name__ == '__main__':
    main()
