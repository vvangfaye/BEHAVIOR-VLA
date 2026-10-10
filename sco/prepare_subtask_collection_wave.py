"""Freeze a finite collection wave using an already verified source snapshot.

Run with the archived evaluator Python on AFS. The input is the pilot's saved
preparation_inputs.json plus an explicit new output root and finite selection.
This does CPU preparation only; it does not submit GPU resources.
"""
import hashlib
import json
from pathlib import Path
import sys
import time


def run(payload):
    code = Path(payload['code_root'])
    marker = (code/'_CODE_SNAPSHOT_READY').read_bytes()
    assert hashlib.sha256(marker).hexdigest() == payload['source_manifest_sha256']
    for name, expected in json.loads(marker)['overlay_files_sha256'].items():
        assert hashlib.sha256((code/name).read_bytes()).hexdigest() == expected
    output = Path(payload['output_root'])
    output.mkdir(parents=True, exist_ok=False)
    inputs = json.loads(Path(payload['pilot_inputs']).read_text())
    assert inputs['code_root'] == str(code)
    for key in ('max_groups','starts_per_group','repeats','seed','held_out_instances'):
        inputs[key] = payload[key]
    inputs.update(output_root=str(output),pilot_tasks=[])
    assert inputs['max_groups']*inputs['starts_per_group']*inputs['repeats'] <= 30
    (output/'preparation_inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
    sys.path[:0] = [str(code/'sco'),str(code/'scripts')]
    from prepare_subtask_collection import prepare_inputs
    from subtask_sample_panel import load_panel
    ready = prepare_inputs(inputs)
    panel = load_panel(output/'parent_panel.json')
    samples = {s['sample_id']:s for s in panel['samples']}
    plan = ready['plan']
    shards = []
    for task in sorted({samples[t['sample_id']]['task'] for t in plan['trials']}):
        trials = [t for t in plan['trials'] if samples[t['sample_id']]['task']==task]
        shards.append(dict(index=len(shards),task=task,trial_ids=[t['trial_id'] for t in trials],
            expected_trials=len(trials),maximum_control_steps=len(trials)*plan['max_steps']))
    assert len({tid for s in shards for tid in s['trial_ids']}) == plan['expected_trials']
    results = output/'results'; results.mkdir()
    contract = dict(run_id=payload['wave_name'],kind=plan['kind'],official_benchmark=False,
        plan_fingerprint=plan['fingerprint'],parent_panel_fingerprint=panel['fingerprint'],
        checkpoint_identity=plan['checkpoint_identity'],max_steps=plan['max_steps'])
    (results/'collection_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    ready.update(state='complete',output_root=str(output),code_root=str(code),
        source_manifest_sha256=payload['source_manifest_sha256'],shards=shards,
        raw_data_root=inputs['raw_data_root'],results_root=str(results),
        wave_name=payload['wave_name'],completed_at=time.time(),
        storage_budget_bytes=2*plan['maximum_output_bytes']+16*1024**3)
    (output/'prepared.json').write_text(json.dumps(ready,indent=2)+'\n')
    return ready
