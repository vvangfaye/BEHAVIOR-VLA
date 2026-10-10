"""Disjoint SCO node assignments with the original full-panel final validator."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
import eval_b1k_subtask_dynamic as runner
from pi05_eval_queue import EvaluationQueue, atomic_write_json
from subtask_sample_panel import load_panel, preflight_panel


def task_assignments(panel, nodes):
    """Keep each task's four frozen instances together and balance control work."""
    jobs = runner._panel_jobs(panel)
    tasks = sorted({job['task'] for job in jobs})
    if type(nodes) is not int or not 1 <= nodes <= len(tasks):
        raise ValueError('Node count must be between one and the task count')
    grouped = {task: [job for job in jobs if job['task'] == task] for task in tasks}
    samples = {s['sample_id']: s for s in panel['samples']}
    weights = {task: sum(2048 * job['rollout_count'] + max(
        samples[s]['start'] for s in job['sample_ids']) for job in grouped[task]) for task in tasks}
    assigned = [[] for _ in range(nodes)]
    loads = [0] * nodes
    for task in sorted(tasks, key=lambda task: (-weights[task], task)):
        node = min(range(nodes), key=lambda i: (loads[i], len(assigned[i]), i))
        assigned[node].append(task)
        loads[node] += weights[task]
    return [{'index': i, 'tasks': sorted(group),
             'episode_keys': [[job['task'], job['instance_id']] for task in sorted(group)
                              for job in grouped[task]],
             'sample_ids': [s for task in sorted(group) for job in grouped[task]
                            for s in job['sample_ids']],
             'estimated_control_and_replay_work': loads[i]}
            for i, group in enumerate(assigned)]


def verify_assignments(document, panel):
    if document.get('panel_fingerprint') != panel['fingerprint']:
        raise ValueError('Assignments changed the full panel identity')
    expected = task_assignments(panel, document['node_count'])
    if document.get('assignments') != expected:
        raise ValueError('Assignments are missing, overlapping or changed')
    if document.get('fingerprint') != runner.dynamic._fingerprint(
            {k: v for k, v in document.items() if k != 'fingerprint'}):
        raise ValueError('Assignment fingerprint mismatch')
    return expected


def gate(path, checkpoint, panel):
    proof = json.loads(Path(path).read_text())
    ready = json.loads((Path(checkpoint) / '_READY').read_text())
    summary = proof.get('summary', {})
    if (proof.get('state') != 'complete' or summary.get('complete') is not True or
            summary.get('eligible_rollouts', 0) < 1 or
            summary.get('panel_fingerprint') != panel['fingerprint'] or
            proof.get('checkpoint_step') != ready['step'] or
            proof.get('checkpoint_manifest_sha256') != ready['manifest_sha256']):
        raise ValueError('A matching one-GPU real rollout gate is required')


def prepare(args):
    root = args.run_root.resolve()
    if root.exists():
        raise ValueError('Preparation requires a fresh run root')
    panel = load_panel(args.panel)
    if panel['selection'].get('task_scope') != 'full100':
        raise ValueError('SCO assignments must retain the full100 parent panel')
    gate(args.validation_success, args.checkpoint, panel)
    assignments = task_assignments(panel, args.nodes)
    raw = preflight_panel(panel, args.raw_data_root)
    if raw['valid'] is not True:
        raise ValueError('Full400 raw state preflight failed')
    spec = argparse.Namespace(**{**vars(args), 'gpus':list(range(args.gpus_per_node)),
        'max_attempts':3, 'max_restarts_per_slot':8, 'sample_max_steps':2048,
        'write_video':True, 'subtask_prediction':True,
        'runtime_root':root / 'global-unused-runtime'})
    manifest = runner.create_manifest(spec, panel, raw)
    runner.validate_manifest(manifest, panel)
    root.mkdir(parents=True)
    for name in ['results', 'attempts', 'logs', 'status', 'shards']:
        (root / name).mkdir()
    atomic_write_json(root / 'panel.json', panel)
    atomic_write_json(root / 'manifest.json', manifest)
    payload = {'schema_version': 1, 'run_id': args.run_id,
               'panel_fingerprint': panel['fingerprint'], 'node_count': args.nodes,
               'assignments': assignments, 'gpus_per_node': args.gpus_per_node}
    payload['fingerprint'] = runner.dynamic._fingerprint(payload)
    verify_assignments(payload, panel)
    atomic_write_json(root / 'assignments.json', payload)
    queue = EvaluationQueue(manifest['queue_db'])
    queue.initialize(((j['task'], j['instance_id']) for j in manifest['jobs']),
                     max_attempts=3, metadata=runner._queue_metadata(manifest))
    from subtask_source_exclusion import prepublish_source_exclusions
    prepublish_source_exclusions(manifest, queue, panel)
    for assignment in assignments:
        shard_root = root / 'shards' / str(assignment['index'])
        shard_root.mkdir()
        for name in ['logs', 'status', 'attempts']:
            (shard_root / name).mkdir()
        shard = copy.deepcopy(manifest)
        shard.update(queue_db=str(shard_root / 'queue.sqlite3'),
                     attempt_root=str(shard_root / 'attempts'),
                     runtime_root=str(shard_root / 'runtime'))
        shard['child_environment'].update(
            VK_ICD_FILENAMES=str(shard_root / 'graphics/nvidia_icd.json'),
            __EGL_VENDOR_LIBRARY_FILENAMES=str(shard_root / 'graphics/10_nvidia.json'))
        shard.pop('fingerprint')
        shard['fingerprint'] = runner.dynamic._fingerprint(shard)
        runner.validate_manifest(shard, panel)
        atomic_write_json(shard_root / 'manifest.json', shard)
        assigned_queue = EvaluationQueue(shard['queue_db'])
        metadata = runner._queue_metadata(shard)
        metadata.update(assignment_index=assignment['index'], assignment_fingerprint=payload['fingerprint'],
                        assigned_episode_jobs=len(assignment['episode_keys']),
                        assigned_segment_rollouts=len(assignment['sample_ids']))
        assigned_queue.initialize((tuple(key) for key in assignment['episode_keys']),
                                  max_attempts=3, metadata=metadata)
        # Reconcile CPU-proven source exclusions without any model invocation.
        completed = {(j['task'], j['instance_id']) for j in queue.snapshot(readonly=True)['jobs']
                     if j['state'] == 'succeeded'}
        for task, instance in assignment['episode_keys']:
            if (task, instance) in completed:
                lease = assigned_queue.claim(task, args.run_id + ':source-preflight:generation-0',
                                              shard['attempt_root'], instance_id=instance)
                if not runner.reconcile_lease(assigned_queue, shard, lease):
                    raise ValueError('Frozen source exclusion failed reconciliation')
    atomic_write_json(root / '_PREPARED', {'assignments_fingerprint': payload['fingerprint'],
        'tasks': 100, 'episode_jobs': len(panel['episodes']), 'segment_rollouts': len(panel['samples']),
        'prepared_at': time.time()})


def load_assignment(root, index):
    root = Path(root).resolve()
    panel = load_panel(root / 'panel.json')
    document = json.loads((root / 'assignments.json').read_text())
    assignments = verify_assignments(document, panel)
    if type(index) is not int or not 0 <= index < len(assignments):
        raise ValueError('Invalid assignment index')
    shard = root / 'shards' / str(index)
    manifest = runner.validate_manifest(json.loads((shard / 'manifest.json').read_text()), panel)
    queue = EvaluationQueue(manifest['queue_db'])
    snapshot = queue.snapshot(readonly=True)
    expected = {tuple(key) for key in assignments[index]['episode_keys']}
    if {(j['task'], j['instance_id']) for j in snapshot['jobs']} != expected:
        raise ValueError('Queue membership changed its task assignment')
    metadata = queue.metadata(readonly=True)
    if metadata.get('assignment_fingerprint') != document['fingerprint'] or metadata.get('assignment_index') != index:
        raise ValueError('Queue belongs to a different assignment')
    return panel, assignments[index], manifest, queue


def finish_assignment(manifest, queue):
    root = Path(manifest['queue_db']).parent
    _, assignment, _, _ = load_assignment(root.parent.parent, int(root.name))
    specs = {(j['task'], j['instance_id']): j['rollout_specs'] for j in manifest['jobs']
             if [j['task'], j['instance_id']] in assignment['episode_keys']}
    result = queue.create_result_manifest(manifest['results_root'], root / 'result_manifest.json',
                                          model='pi05', rollout_specs_by_job=specs)
    atomic_write_json(root / '_ASSIGNED_SUCCESS', {'panel_fingerprint': manifest['panel_fingerprint'],
        'assignment_index': assignment['index'], 'episode_jobs': len(assignment['episode_keys']),
        'sample_ids': assignment['sample_ids'], 'complete_full100': False,
        'result_manifest_sha256': runner.dynamic._fingerprint(result), 'completed_at': time.time()})


def run_assignment(args):
    panel, assignment, manifest, queue = load_assignment(args.run_root, args.index)
    root = Path(manifest['queue_db']).parent
    with runner.dynamic.exclusive_run_lock(root):
        specs = runner.worker_specs(manifest['gpus'], manifest['base_port'], tuple(assignment['tasks']))
        runner.dynamic.verify_preflight(runner._base_preflight_manifest(manifest),
                                        resume=(root / '_STARTED').exists(), specs=specs)
        # Each node verifies its own exact files against the one full CPU preflight.
        assigned = set(assignment['sample_ids'])
        for row in manifest['raw_preflight']['episodes']:
            if not assigned.intersection(row['sample_ids']):
                continue
            digest = hashlib.sha256()
            with Path(row['path']).open('rb') as source:
                for block in iter(lambda: source.read(8 * 1024**2), b''):
                    digest.update(block)
            if digest.hexdigest() != row['sha256']:
                raise ValueError('Assigned raw file changed after full-panel preflight')
        Path(manifest['runtime_root']).mkdir(parents=True, exist_ok=True)
        (root / '_STARTED').touch()
        runner.requeue_orphaned_leases(queue, manifest)
        runner.supervise(manifest, queue, tasks=tuple(assignment['tasks']),
                         completion_handler=finish_assignment)


def aggregate(args):
    root = args.run_root.resolve()
    panel = load_panel(root / 'panel.json')
    document = json.loads((root / 'assignments.json').read_text())
    assignments = verify_assignments(document, panel)
    manifest = runner.validate_manifest(json.loads((root / 'manifest.json').read_text()), panel)
    with runner.dynamic.exclusive_run_lock(root):
        for assignment in assignments:
            _, _, shard, queue = load_assignment(root, assignment['index'])
            if runner.dynamic.supervisor_is_active(Path(shard['queue_db']).parent):
                raise ValueError('An assignment supervisor is still active')
            finish_assignment(shard, queue)  # Re-read every assigned JSON/video and its hashes.
        queue = EvaluationQueue(manifest['queue_db'])
        for job in queue.snapshot(readonly=True)['jobs']:
            if job['state'] == 'succeeded':
                continue
            lease = queue.claim(job['task'], manifest['run_id'] + ':aggregate:generation-0',
                                manifest['attempt_root'], instance_id=job['instance_id'])
            if lease is None or not runner.reconcile_lease(queue, manifest, lease):
                raise ValueError('Global episode artifacts are missing or invalid')
        runner.finalize_results(manifest, queue)
        sys.path.insert(0, str(REPO / 'nebula'))
        from subtask_eval import write_reports
        summary = json.loads((root / 'summary.json').read_text())
        write_reports(summary, root)
        atomic_write_json(root / '_REPORTED', {'complete': summary['complete'],
            'panel_fingerprint': panel['fingerprint'], 'reported_at': time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare_parser = commands.add_parser('prepare')
    for name in ['run-root','panel','raw-data-root','checkpoint','eval-env','pi05-env',
                 'omnigibson-data-path','validation-success']:
        prepare_parser.add_argument('--' + name, type=Path, required=True)
    prepare_parser.add_argument('--run-id', required=True)
    prepare_parser.add_argument('--repo-id', required=True)
    prepare_parser.add_argument('--nodes', type=int, required=True)
    prepare_parser.add_argument('--gpus-per-node', type=int, choices=range(1,9), required=True)
    prepare_parser.add_argument('--base-port', type=int, default=8810)
    run_parser = commands.add_parser('run')
    run_parser.add_argument('--run-root', type=Path, required=True)
    run_parser.add_argument('--index', type=int, required=True)
    aggregate_parser = commands.add_parser('aggregate')
    aggregate_parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()
    {'prepare':prepare, 'run':run_assignment, 'aggregate':aggregate}[args.command](args)


if __name__ == '__main__':
    main()
