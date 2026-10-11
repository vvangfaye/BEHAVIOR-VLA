"""Prepare one complete public-test audit, then execute it on an AFS CPU worker.

prepare: rehash all 2000 local originals BEFORE any SCO query; pin the existing
accepted evidence, every recovery generation, and exact terminal worker identities.
execute: read-only audit all 401 AFS roots, then invoke the two pinned streaming
helpers. This module never submits or stops a cloud job. A caller may submit ONE
CPU job only with the prepared bundle; CPU resource selection/transport remain
explicit caller responsibilities. There are no per-shard audits or dataset writes.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import time

CHECKPOINT = 1688977
TOTAL = 2000
OWNER = '260010165'
QUEUE_SHA = '0e1c0a0ad0daae695119766731adc83981c98c6cacc4f726a7df3c27a0e0419f'
PINS = {
    'streaming_finalize_public_test.py': 'fe0b68583c6067c3637a0162d01abdd63ce15d5f08bfa02221d90c187db399d8',
    'streaming_package_public_test.py': '86301bc6b68c37c7c809b292630dd5c3edd3c23297aa7e90d5825835a0080cc9',
}
IDENTITY = ('assignment_root', 'run_id', 'source_manifest_sha256', 'driver_sha256',
            'checkpoint', 'code', 'task', 'instance_ids', 'expected_rollouts')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def within(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(root in path.parents and not Path(relative).is_absolute(), 'Unsafe evidence path')
    return path


def load_evidence(root, relative, captured):
    raw = within(root, relative).read_bytes()
    captured[relative] = dict(sha256=digest(raw), bytes=len(raw), content=json.loads(raw))
    return captured[relative]['content']


def validate_recovery_chain(root, current, captured=None):
    """Validate ALL generations, even when the latest partial already has this UID."""
    captured = {} if captured is None else captured
    node = current
    chain = []
    seen = set()
    require(type(node['generation']) is int and 0 <= node['generation'] <= 3, 'Invalid generation')
    while node['generation']:
        require(node['uid'] not in seen, 'Cyclic recovery chain')
        seen.add(node['uid'])
        receipt = load_evidence(root, node['recovery_receipt'], captured)
        require(receipt['state'] == 'submitted' and receipt['uid'] == node['uid'], 'Recovery UID mismatch')
        require(receipt['generation'] == node['generation'], 'Recovery generation mismatch')
        require(receipt['name'] == node['name'], 'Recovery name mismatch')
        require(receipt['old_writer_terminal'] is True, 'Old writer not terminal')
        require(receipt['old_worker_phase'] in ('Failed', 'Deleted'), 'Old worker not stopped')
        old = receipt['old_submission']
        require(old['generation'] + 1 == node['generation'], 'Noncontiguous recovery chain')
        require(old['uid'] != node['uid'], 'Recovery reused old UID')
        for field in IDENTITY:
            require(old[field] == node[field] == receipt[field], 'Recovery changed ' + field)
        if receipt.get('queue_untouched'):
            # The only historical no-attempt recovery predates pinned reconciliation
            # proofs. Bind BOTH its actual failure receipt and zero-attempt readback;
            # do not silently accept missing proof for other generations.
            require(node.get('index') == 48 and node['generation'] == 1,
                    'Unrecognized legacy recovery without reconciliation proof')
            platform = load_evidence(root, 'production_a048_failure_platform.json', captured)
            proof = load_evidence(root, 'production_a048_failure_afs_readback.json', captured)
            require(platform['uid'] == old['uid'] and platform['name'] == old['name']
                    and platform['state'] == 'FAILED', 'Legacy terminal identity mismatch')
            require(proof['result_json_count'] == 0 and proof['canonical_results'] == [],
                    'Legacy recovery had completed artifacts')
            require(proof['status.json']['state'] == 'failed', 'Legacy source did not fail')
            require(all(r['state'] == 'pending' and r['attempts'] == 0 for r in proof['queue']),
                    'Legacy recovery reset attempts')
            artifacts = {}
        else:
            rel = receipt['queue_reconciliation_proof']
            proof = load_evidence(root, rel, captured)
            require(captured[rel]['sha256'] == receipt['queue_reconciliation_proof_sha256'],
                    'Recovery proof SHA mismatch')
            require(proof['state'] == 'terminal_worker_queue_reconciled'
                    and proof['old_uid'] == old['uid']
                    and proof['old_generation'] == old['generation'], 'Recovery proof identity mismatch')
            require(proof['old_writer_terminal'] is True and proof['completed_artifacts_preserved'] is True,
                    'Recovery did not preserve stopped writer artifacts')
            require(proof['queue_source_sha256'] == QUEUE_SHA, 'Queue source changed')
            require(receipt['attempt_counts_preserved'] is True, 'Attempt preservation missing')
            require(proof['metadata'] == dict(run_id=node['run_id'], mode='public_test', num_rollouts=1,
                    require_videos=False, checkpoint_step=CHECKPOINT,
                    source_manifest_sha256=node['source_manifest_sha256']), 'Recovery queue metadata mismatch')
            before = {r['instance_id']: r for r in proof['before_queue']}
            require(len(before) == len(proof['before_queue']) == len(node['instance_ids']), 'Duplicate prior queue row')
            for row in proof['queue']:
                prior = before[row['instance_id']]
                require(row['attempts'] == prior['attempts'] and row['max_attempts'] == prior['max_attempts'] == 3,
                        'Recovery reset attempt budget')
                require(row['state'] in ('pending', 'succeeded'), 'Recovery retained active lease')
                if prior['state'] != 'running':
                    require(row['state'] == prior['state'], 'Recovery changed completed/pending state')
            for name, value in proof['artifacts_before'].items():
                require(proof['artifacts_after'].get(name) == value, 'Recovery changed original artifact')
            artifacts = proof['artifacts_after']
        require(proof['assignment_root'] == old['assignment_root'], 'Recovery root mismatch')
        require(proof['status.json']['source_manifest_sha256'] == old['source_manifest_sha256'],
                'Recovery source mismatch')
        rows = proof['queue']
        require(len(rows) == len(node['instance_ids']) and
                {(r['task'], r['instance_id']) for r in rows} == {(node['task'], i) for i in node['instance_ids']},
                'Recovery queue scope mismatch')
        chain.append(dict(old_submission=old, receipt=receipt, proof=proof, artifacts=artifacts))
        node = old
    require(node['state'] == 'submitted' and node['generation'] == 0, 'Missing initial submission')
    return chain


def observe_terminal(receipt, cli):
    """Read-only exact identity query; an unknown worker format fails closed."""
    env = os.environ.copy()
    env.update(SCO_CONFIG='/private/tmp/behavior-sco-config', SCO_HOME='/private/tmp/behavior-sco-runtime',
               SCO_DATA_HOME='/private/tmp/behavior-sco-data')
    prefix = [cli, 'acp', 'jobs']
    job = json.loads(subprocess.run(prefix + ['describe', receipt['name'], '--workspace-name=share-space',
        '-o', 'json'], env=env, capture_output=True, text=True, check=True, timeout=45).stdout)
    workers = subprocess.run(prefix + ['get-workers', receipt['name'], '--workspace-name=share-space'],
        env=env, capture_output=True, text=True, check=True, timeout=45).stdout
    phases = re.findall(r'\b(?:Running|Pending|Starting|Creating|ContainerCreating|Unknown|Succeeded|Completed|Failed|Deleted)\b', workers)
    return dict(name=job['name'], uid=job['uid'], owner=job['ownership']['user_name'], state=job['state'],
        worker_uid_present=receipt['uid'] in workers, worker_phases=phases,
        workers_sha256=digest(workers.encode()), observed_at=time.time())


def validate_terminal(receipt, observed, expected_state):
    require(observed['uid'] == receipt['uid'] and observed['name'] == receipt['name']
            and observed['owner'] == OWNER, 'Terminal observation identity mismatch')
    require(observed['state'] == expected_state, 'Writer has not reached required terminal state')
    allowed = {'Succeeded', 'Completed', 'Deleted'} if expected_state == 'SUCCEEDED' else {'Failed', 'Deleted'}
    require(observed['worker_uid_present'] is True and observed['worker_phases']
            and set(observed['worker_phases']) <= allowed, 'Writer worker is not proven stopped')


def prepare(root, destination, archive, audit_report, *, observer=None, count_fn=None):
    """Return a sealed execution bundle; no writes or cloud mutations occur here."""
    root = Path(root).resolve()
    if count_fn is None:
        from count_public_test_originals import count_originals
        count_fn = count_originals
    count = count_fn(root)
    # This MUST precede observer construction/use or any network operation.
    require(count['actual_completed_rollouts'] == TOTAL, 'Full scope incomplete; no cloud queries or submission allowed')
    require(count['accepted_rollouts'] == 80, 'Preserve the existing 80 accepted results')
    captured = {}
    prepared = load_evidence(root, 'unified_final_audit_inputs_prepared.json', captured)
    plan = load_evidence(root, 'production_plan.json', captured)
    ready = load_evidence(root, 'preparation_ready.json', captured)
    pilot = load_evidence(root, 'pilot_accepted.json', captured)
    from submit_public_test_shard import build_payload
    source_sha = file_digest(root / 'pilot_driver_source.py')
    require(source_sha == pilot['driver_sha256'], 'Frozen driver changed')
    require(prepared['scope'] == ready['scope'] and prepared['code'] == ready['code']
            and prepared['source_manifest_sha256'] == ready['source_manifest_sha256'], 'Prepared scope changed')
    expected = {(t, i, 0) for t in ready['scope']['tasks'] for i in range(301, 321)}
    require(len(expected) == TOTAL and len(plan['shards']) == 400, 'Invalid public scope')
    assignments = prepared['assignments']
    require(len(assignments) == 401, 'Expected pilot and 400 assignments')
    receipts, originals, required, chains = [], [], [], {}
    # Keep the bytes admitted by count_originals, rather than trusting a later
    # file hash if a concurrent process changes a local original.
    admitted_sha = {(r['task'], r['instance_id'], 0): r['sha256'] for r in count['partial_rows']}
    admitted_sha[('turning_on_radio', 301, 0)] = pilot['checked_artifacts']['json']['sha256']
    for accepted_path in (root / 'production_audits').glob('a*_accepted.json'):
        a = json.loads(accepted_path.read_bytes())
        report_path = root / 'production_audits' / ('a%03d_report.json' % a['index'])
        raw = report_path.read_bytes()
        require(digest(raw) == a['report_sha256'], 'Accepted report changed')
        for row in json.loads(raw)['results']:
            admitted_sha[(row['task'],row['instance_id'],row['rollout_id'])] = row['json']['sha256']
    require(len(admitted_sha) == TOTAL, 'Original SHA inventory incomplete')
    for assignment in assignments:
        label = 'pilot' if assignment['pilot'] else Path(assignment['root']).name
        relative = 'pilot_submission.json' if label == 'pilot' else 'production_submissions/' + label + '.json'
        current = load_evidence(root, relative, captured)
        if label == 'pilot':
            require(current['uid'] == pilot['pilot_uid'] and current['generation'] == 0, 'Pilot identity changed')
        else:
            index = int(label[1:])
            predicted = build_payload(ready, plan, pilot, index, current['name'], source_sha)
            for field in IDENTITY:
                require(current[field] == predicted[field], 'Current receipt differs from frozen plan: ' + field)
        require(current['assignment_root'] == assignment['root'] and current['task'] == assignment['task']
                and current['instance_ids'] == assignment['instance_ids'], 'Prepared assignment binding mismatch')
        chain = validate_recovery_chain(root, current, captured)
        chains[label] = chain
        required.append((current, 'SUCCEEDED'))
        required.extend((link['old_submission'], 'FAILED') for link in chain)
        receipts.append(current)
        for instance in current['instance_ids']:
            name = '%s_%s_0.json' % (current['task'], instance)
            local = root / ('pilot_artifacts' if label == 'pilot' else 'production_artifacts/' + label) / name
            sha = file_digest(local)
            require(sha == admitted_sha[(current['task'], instance, 0)], 'Local original changed after admission')
            originals.append(dict(task=current['task'], instance_id=instance, rollout_id=0,
                label=label, sha256=sha, bytes=local.stat().st_size))
    require(len(originals) == TOTAL and {(r['task'], r['instance_id'], r['rollout_id']) for r in originals} == expected,
            'Original result scope mismatch')
    accepted = {}
    for path in sorted((root / 'production_audits').glob('a*_accepted.json')):
        row = load_evidence(root, path.relative_to(root).as_posix(), captured)
        label = 'a%03d' % row['index']
        report = root / 'production_audits' / (label + '_report.json')
        require(file_digest(report) == row['report_sha256'], 'Accepted report changed')
        accepted[label] = dict(receipt_sha256=file_digest(path), report_sha256=file_digest(report),
            rollout_count=row['rollout_count'], production_uid=row['production_uid'])
    require(sum(r['rollout_count'] for r in accepted.values()) + 1 == 80, 'Accepted evidence was lost')
    outputs = dict(destination=str(destination), archive=str(archive), audit_report=str(audit_report))
    validate_outputs(outputs, assignments, prepared['code'])
    # Only a fresh function call can supply terminal evidence, never a cached CLI snapshot.
    observer = observer or (lambda receipt: observe_terminal(receipt, '/private/tmp/behavior-sco-runtime/bin/sco'))
    started = time.time()
    with ThreadPoolExecutor(max_workers=6) as pool:
        observations = list(pool.map(lambda pair: observer(pair[0]), required))
    for (receipt, state), observed in zip(required, observations):
        validate_terminal(receipt, observed, state)
        require(started <= observed['observed_at'] <= time.time(), 'Cached terminal observation rejected')
    for relative, record in captured.items():
        require(file_digest(within(root, relative)) == record['sha256'], 'Evidence changed during preparation')
    for label, record in accepted.items():
        require(file_digest(root / 'production_audits' / (label + '_report.json')) == record['report_sha256'],
                'Accepted report changed during preparation')
    sources = {}
    for name, sha in PINS.items():
        raw = (root / 'finalization_sources' / name).read_bytes()
        require(digest(raw) == sha, 'Pinned finalization helper changed')
        sources[name] = base64.b64encode(raw).decode()
    payload = dict(scope=prepared['scope'], code=prepared['code'], source_manifest_sha256=prepared['source_manifest_sha256'],
        assignments=assignments, destination=str(destination))
    body = dict(schema=1, state='ready_for_one_unified_audit', checkpoint_step=CHECKPOINT,
        originals=originals, accepted=accepted, pilot_accepted=pilot, receipts=receipts, recovery_chains=chains,
        terminal_observations=observations, evidence_pins={k:v['sha256'] for k,v in captured.items()},
        payload=payload, sources=sources, outputs=outputs, prepared_at=time.time(),
        original_count=TOTAL, accepted_count=80)
    return dict(bundle=body, sha256=digest(canonical(body)))


def validate_outputs(outputs, assignments, code):
    paths = [Path(outputs[k]).expanduser() for k in ('destination', 'archive', 'audit_report')]
    require(all(p.is_absolute() for p in paths), 'Explicit absolute output paths required')
    paths = [p.resolve() for p in paths]
    require(len(set(paths)) == 3, 'Explicit distinct output paths required')
    destination, archive, audit = paths
    fields = destination.name.split('.')
    require(len(fields) == 5 and fields[:2] == ['standard', 'public'] and all(fields[2:]),
            'Destination must specify standard.public.team.affiliation.date')
    for path in paths:
        for protected in [Path(code).resolve()] + [Path(a['root']).resolve() for a in assignments]:
            require(path != protected and protected not in path.parents and path not in protected.parents,
                    'Output overlaps immutable input')
    require(destination not in archive.parents and destination not in audit.parents, 'Reports/archive must be outside final directory')


def inspect_afs(body):
    """One read-only audit over the full run; no per-shard acceptance is written."""
    originals = {(r['task'], r['instance_id'], r['rollout_id']): r for r in body['originals']}
    require(len(originals) == len(body['originals']) == TOTAL, 'Audit requires exactly 2000 unique originals')
    receipts = {r['assignment_root']: r for r in body['receipts']}
    seen, reports = set(), []
    for assignment in body['payload']['assignments']:
        root = Path(assignment['root']).resolve()
        receipt = receipts[str(root)]
        label = 'pilot' if assignment['pilot'] else root.name
        marker_path = root / ('_PILOT_SUCCESS' if assignment['pilot'] else '_ASSIGNED_SUCCESS')
        marker, manifest = json.loads(marker_path.read_bytes()), json.loads((root / 'RESULT_MANIFEST.json').read_bytes())
        require(marker['state'] == 'complete' and marker['checkpoint_step'] == CHECKPOINT,
                'Missing successful completion marker')
        require(marker['source_manifest_sha256'] == receipt['source_manifest_sha256'] and
                marker['task'] == receipt['task'] and marker['instance_ids'] == receipt['instance_ids'],
                'Completion marker binding mismatch')
        with sqlite3.connect((root / 'queue.sqlite3').as_uri() + '?mode=ro', uri=True) as db:
            db.row_factory = sqlite3.Row
            metadata = {r['key']: json.loads(r['value']) for r in db.execute('SELECT key,value FROM metadata')}
            jobs = [dict(r) for r in db.execute('SELECT * FROM jobs')]
            attempts = [dict(r) for r in db.execute('SELECT a.*,j.task,j.instance_id FROM attempts a JOIN jobs j ON a.job_id=j.id')]
        require(metadata == manifest['metadata'], 'Queue and manifest metadata differ')
        require(metadata['run_id'] == receipt['run_id'] and metadata['checkpoint_step'] == CHECKPOINT
                and metadata['source_manifest_sha256'] == receipt['source_manifest_sha256'], 'Final queue identity mismatch')
        require(len(jobs) == len(receipt['instance_ids']) and all(j['state'] == 'succeeded' for j in jobs),
                'Queue is incomplete or still has a writer')
        require({(j['task'],j['instance_id']) for j in jobs} == {(receipt['task'],i) for i in receipt['instance_ids']},
                'Final queue scope changed')
        require(all(a['state'] in ('failed','succeeded') and a['ended_at'] is not None for a in attempts),
                'Unfinished attempt remains')
        for job in jobs:
            used = [a for a in attempts if a['job_id'] == job['id']]
            require(job['max_attempts'] == 3 and 1 <= job['attempts'] <= 3 and
                    {a['attempt'] for a in used} == set(range(1,job['attempts']+1)), 'Attempt history/budget missing')
        for link in body['recovery_chains'][label]:
            for relative, expected in link['artifacts'].items():
                path = within(root, relative)
                require(path.stat().st_size == expected['bytes'] and file_digest(path) == expected['sha256'],
                        'Preserved recovery artifact changed')
            prior = {r['instance_id']: r for r in link['proof']['queue']}
            for job in jobs:
                require(job['attempts'] >= prior[job['instance_id']]['attempts'], 'Attempt count decreased after recovery')
            actual_attempts = {(a['instance_id'],a['attempt']):a for a in attempts}
            for old in link['proof'].get('after_attempts', []):
                actual = actual_attempts[(old['instance_id'],old['attempt'])]
                require(all(actual[k] == old[k] for k in ('worker','lease_token','attempt_dir','started_at','ended_at','state')),
                        'Historical attempt identity changed')
        require(manifest['rollout_count'] == len(receipt['instance_ids']) == len(manifest['rollouts']), 'Manifest incomplete')
        for row in manifest['rollouts']:
            key = (row['task'],row['instance_id'],row['rollout_id'])
            require(key in originals and key not in seen, 'Duplicate or unexpected AFS result')
            expected = originals[key]
            require(expected['label'] == label and row['json']['sha256'] == expected['sha256']
                    and row['json']['size'] == expected['bytes'], 'AFS result differs from verified original')
            path = within(root / 'results', row['json']['path'])
            require(path.stat().st_size == expected['bytes'] and file_digest(path) == expected['sha256'], 'AFS bytes changed')
            seen.add(key)
        reports.append(dict(root=str(root), uid=receipt['uid'], generation=receipt['generation'],
            marker_sha256=file_digest(marker_path), manifest_sha256=file_digest(root/'RESULT_MANIFEST.json')))
    require(len(seen) == TOTAL, 'Unified audit incomplete')
    return dict(state='unified_audit_verified', checkpoint_step=CHECKPOINT, rollouts=TOTAL,
                accepted_preserved=80, assignments=reports)


def execute(envelope):
    """Run only on the one separately submitted AFS CPU worker, after preparation."""
    body = envelope['bundle']
    require(digest(canonical(body)) == envelope['sha256'], 'Prepared bundle changed')
    require(body['state'] == 'ready_for_one_unified_audit' and body['original_count'] == TOTAL
            and body['accepted_count'] == 80, 'Bundle was not admitted')
    validate_outputs(body['outputs'], body['payload']['assignments'], body['payload']['code'])
    outputs = {k:Path(v) for k,v in body['outputs'].items()}
    require(all(not p.exists() for p in outputs.values()), 'Preserve existing final outputs; inspect prior execution')
    namespaces = {}
    for name, sha in PINS.items():
        raw = base64.b64decode(body['sources'][name], validate=True)
        require(digest(raw) == sha, 'Pinned source mismatch')
        namespace = {'__name__': 'unified_finalization_pinned_helper'}
        exec(compile(raw, name, 'exec'), namespace)
        namespaces[name] = namespace
    audit_path = outputs['audit_report']
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    # A permanent exclusive intent prevents duplicate CPU writers/automatic retries.
    intent = Path(str(audit_path) + '.intent.json')
    with intent.open('x') as stream:
        json.dump(dict(state='executing', bundle_sha256=envelope['sha256'], outputs=body['outputs']), stream)
    audit = inspect_afs(body)
    audit['bundle_sha256'] = envelope['sha256']
    with audit_path.open('x') as stream:
        json.dump(audit, stream, indent=2)
    final = namespaces['streaming_finalize_public_test.py']['run'](body['payload'])
    require(final['state'] == 'complete' and final['public_test_rollouts'] == TOTAL, 'Finalizer did not complete')
    package = namespaces['streaming_package_public_test.py']['package'](outputs['destination'], outputs['archive'])
    require(package['roundtrip_verified'] is True and package['public_test_rollouts'] == TOTAL, 'Package not verified')
    return dict(state='finalized_and_packaged', unified_audit_sha256=file_digest(audit_path),
                bundle_sha256=envelope['sha256'], package=package)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--evidence', type=Path, required=True)
    for name in ('destination', 'archive', 'audit-report', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p = sub.add_parser('execute')
    p.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        envelope = prepare(args.evidence, args.destination, args.archive, args.audit_report)
        with args.output.open('x') as stream:
            json.dump(envelope, stream, indent=2)
        print(json.dumps(dict(state=envelope['bundle']['state'], bundle=str(args.output), sha256=envelope['sha256'],
                              cloud_submitted=False, rollouts=TOTAL)))
    else:
        print(json.dumps(execute(json.loads(args.bundle.read_text())), indent=2))


if __name__ == '__main__':
    main()
