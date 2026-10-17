"""Bounded CPU-only inspection; prints metadata, never file contents or credentials."""
import ast
import hashlib
import json
from pathlib import Path


def probe(config):
    runtime = Path(config['runtime'])
    code = Path(config['code'])
    dataset = runtime / 'omnigibson-data/2026-challenge-task-instances'
    metadata = dataset / 'metadata/task.jsonl'
    tasks = {r['task_name']: r for line in metadata.read_text().splitlines()
             if line.strip() for r in [json.loads(line)]}
    sources = {}
    for relative, expected in config['critical_files'].items():
        actual = hashlib.sha256((code / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Frozen source mismatch: ' + relative)
        sources[relative] = actual
    evaluator = code / 'BEHAVIOR-1K/OmniGibson/omnigibson/eval/evaluator.py'
    tree = ast.parse(evaluator.read_text())
    resolver = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'resolve_instance_ids')
    resolver_source = ast.get_source_segment(evaluator.read_text(), resolver)
    namespace = {'EVAL_MODES': ('train', 'public_test', 'hidden_test')}
    exec(compile(ast.Module(body=[resolver], type_ignores=[]), str(evaluator), 'exec'), namespace)
    direct_resolution = all(namespace['resolve_instance_ids'](task, [instance], mode='train') == [instance]
                            for task, instance in config['pairs'])
    rows = []
    for task, instance in config['pairs']:
        task_metadata = tasks[task]
        candidates = sorted((dataset / 'scenes').glob(
            f'*/json/*_task_{task}_instances/*_task_{task}_*_{instance}_template-tro_state.json'))
        if len(candidates) > 10:
            raise ValueError('Unexpected unbounded instance matches')
        files = []
        for path in candidates:
            raw = path.read_bytes()
            state = json.loads(raw)
            files.append({'relative_path': str(path.relative_to(dataset)),
                          'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                          'has_robot_poses': 'robot_poses' in state})
        rows.append({'task': task, 'instance_id': instance,
                     'native_task_budget': int(1.5 * task_metadata['length']),
                     'metadata_keys': sorted(task_metadata), 'state_files': files})
    result = {'kind': 'heldout20_cpu_asset_preflight', 'rows': rows,
              'task_metadata_sha256': hashlib.sha256(metadata.read_bytes()).hexdigest(),
              'verified_sources': sources, 'train_resolution_exact': direct_resolution,
              'resolver_sha256': hashlib.sha256(resolver_source.encode()).hexdigest(),
              'all_instance_states_found': all(len(r['state_files']) == 1 for r in rows)}
    print('HOLDOUT_PREFLIGHT_JSON=' + json.dumps(result, separators=(',', ':')), flush=True)
