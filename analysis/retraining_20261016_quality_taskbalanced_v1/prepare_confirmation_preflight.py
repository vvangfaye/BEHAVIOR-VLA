"""One bounded CPU metadata probe for the previously reserved confirmation panel."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'heldout20_confirmation_preflight'


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_exclusion(panel):
    source = ROOT / panel['declared_heldout_source']['path']
    if sha(source) != panel['declared_heldout_source']['sha256']:
        raise ValueError('Declared training exclusions changed')
    declared = load(source)
    chosen = set(map(tuple, panel['pairs']))
    if {task for task, _ in chosen} & {task for task, _ in declared['pairs']}:
        raise ValueError('Confirmation tasks overlap the tuning panel')
    for proof in declared['source_proofs']:
        path = ROOT / proof['path']
        if sha(path) != proof['sha256']:
            raise ValueError('Training index changed')
        index = load(path)
        if (not chosen.issubset(set(map(tuple, index['held_out_instances'])))
                or any((row['task_name'], row['task_instance_id']) in chosen for row in index['episodes'])):
            raise ValueError('Confirmation pair is not held out of indexed training')
    for proof in panel['historical_identity_sources']:
        path = ROOT / proof['path']
        if sha(path) != proof['sha256']:
            raise ValueError('Historical identity snapshot changed')
        if any((row['task'], row['instance_id']) in chosen for row in load(path)['records']):
            raise ValueError('Confirmation pair already evaluated in the exclusion history')
    return {'training_index_sha256_rechecked': len(declared['source_proofs']),
            'indexed_training_pair_overlap': 0, 'tuning_task_overlap': 0,
            'historical_evaluation_pair_overlap': 0, 'baseline_pretraining_exclusion_proven': False}


def main():
    panel_path = ROOT / 'heldout20_confirmation_panel.json'
    panel = load(panel_path)
    if (len(panel['pairs']) != 20 or len(set(map(tuple, panel['pairs']))) != 20
            or panel['selection_uses_evaluation_outcomes'] or panel['candidate_checkpoint_selected']):
        raise ValueError('Reserved confirmation panel identity changed')
    spec = importlib.util.spec_from_file_location('confirmation_preflight_transport',
                                                 ROOT / 'heldout20_baseline_vs_expert1k/controller.py')
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    DEST.mkdir(exist_ok=True)
    (DEST / 'intents').mkdir(exist_ok=True)
    (DEST / 'private').mkdir(exist_ok=True, mode=0o700)
    base.ROOT = DEST
    base.transport.ROOT = DEST
    config = load(ROOT / 'heldout20_preflight/config.json')
    config['pairs'] = panel['pairs']
    config_path = DEST / 'config.json'
    if config_path.exists() and load(config_path) != config:
        raise ValueError('Existing preflight configuration changed')
    if not config_path.exists():
        base.transport.new(config_path, config)
    plan = load(ROOT / 'heldout20_baseline_vs_expert1k/plan.json')
    key = 'behavior1k/confirmation-preflight-20261017/result.json'
    client = base.transport.aoss_client()
    intent_path = DEST / 'intents/preflight.json'
    if not intent_path.exists():
        url = base.signed(client, key)
        probe_source = (ROOT / 'heldout20_preflight/probe.py').read_text()
        source = ('import contextlib,io,json,urllib.request\n'
                  + 'namespace={}\nexec(compile(' + repr(probe_source) + ',"probe.py","exec"),namespace)\n'
                  + 'buffer=io.StringIO()\nwith contextlib.redirect_stdout(buffer):\n namespace["probe"](' + repr(config) + ')\n'
                  + 'raw=buffer.getvalue().strip()\nassert raw.startswith("HOLDOUT_PREFLIGHT_JSON=")\n'
                  + 'value=json.loads(raw.split("=",1)[1])\n'
                  + 'value.update(panel_sha256=' + repr(sha(panel_path)) + ',probe_source_sha256=' + repr(hashlib.sha256(probe_source.encode()).hexdigest()) + ')\n'
                  + 'body=json.dumps(value).encode()\nopener=urllib.request.build_opener(urllib.request.ProxyHandler({}))\n'
                  + 'with opener.open(urllib.request.Request(' + repr(url) + ',data=body,method="PUT"),timeout=90):pass\n'
                  + 'print("confirmation metadata probe saved",flush=True)\n')
        path = DEST / 'private/startup.py'
        path.write_text(source)
        path.chmod(0o600)
        command = 'timeout --signal=TERM --kill-after=15 240 python3 -u -c ' + shlex.quote(source)
        if len(command.encode()) >= 100000:
            raise ValueError('Preflight startup is too large')
        intent = {'id': 'preflight', 'name': 'b1k-confirm20-1017-preflight', 'uid': None,
                  'gpu': 0, 'image': plan['image'], 'creation_attempted': False,
                  'panel_sha256': sha(panel_path), 'source_sha256': sha(path),
                  'config_sha256': sha(config_path), 'maximum_runtime_seconds': 240}
        base.transport.new(intent_path, intent)
        base.native_create(intent, command, cpu=True)
    intent = load(intent_path)
    if (intent['panel_sha256'] != sha(panel_path) or intent['config_sha256'] != sha(config_path)
            or intent['source_sha256'] != sha(DEST / 'private/startup.py')):
        raise ValueError('Preflight evidence changed')
    state = base.observe(intent)
    base.save(DEST / 'native_observation.json', state)
    try:
        response = client.get_object(Bucket='260010165', Key=key)
    except Exception as exc:
        if getattr(exc, 'response', {}).get('Error', {}).get('Code') not in ('NoSuchKey', '404'):
            raise
        if state['state'] in base.TERMINAL:
            raise ValueError('Preflight ended without metadata report; reconcile existing job')
        print(json.dumps({'state': state['state'], 'uid': state['uid'], 'report_received': False}))
        return
    try:
        data = json.loads(response['Body'].read(1024 * 1024))
    finally:
        response['Body'].close()
    modified = response['LastModified'].timestamp()
    if modified + 2 < intent['creation_started_at']:
        raise ValueError('Metadata object predates this exact submission')
    if (data['panel_sha256'] != sha(panel_path)
            or data['probe_source_sha256'] != sha(ROOT / 'heldout20_preflight/probe.py')
            or data['task_metadata_sha256'] != load(ROOT / 'heldout20_preflight/result.json')['task_metadata_sha256']
            or not data['all_instance_states_found']
            or not data['train_resolution_exact'] or data['verified_sources'] != config['critical_files']
            or [(r['task'], r['instance_id']) for r in data['rows']] != list(map(tuple, panel['pairs']))
            or any(len(r['state_files']) != 1 or not r['state_files'][0]['has_robot_poses']
                   or r['native_task_budget'] <= 0 for r in data['rows'])):
        raise ValueError('Confirmation scene or source metadata differs')
    base.save(DEST / 'result.json', data)
    exclusion = verify_exclusion(panel) if state['state'] == 'SUCCEEDED' else None
    base.save(DEST / 'acceptance.json', {'accepted': state['state'] == 'SUCCEEDED',
              'native': state, 'panel_sha256': sha(panel_path), 'result_sha256': sha(DEST / 'result.json'),
              'result_object_last_modified': modified, 'result_object_etag': response['ETag'],
              'exclusion': exclusion,
              'confirmation_evaluation_started': False})
    print(json.dumps({'state': state['state'], 'uid': state['uid'], 'verified_instances': len(data['rows']),
                      'accepted': state['state'] == 'SUCCEEDED'}))


if __name__ == '__main__':
    main()
