"""Small standard-resource probes for native, history, and additional train starts."""
import hashlib
import json
from pathlib import Path
import sys


def run(inputs,code_root,source_identity,output):
    code=Path(code_root);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    assert hashlib.sha256((code/'_CODE_SNAPSHOT_READY').read_bytes()).hexdigest()==source_identity
    sys.path[:0]=[str(code/'scripts'),str(code/'sco')]
    import h5py
    import numpy as np
    import eval_b1k_subtask_samples as runner
    from full_subtask_collection import freeze,targets,NATIVE,HISTORY
    from subtask_sample_panel import load_panel
    from recorded_native_state import decode_start,validate_baseline_result,preview_source_role_bindings
    from audit_history_assignment_artifacts import validate_history_result
    panel=load_panel(inputs['panel_path']);extra=load_panel(inputs['additional_panel_path'])
    groups=targets(inputs['metrics'],inputs['acceptance'])
    by_task={a['tasks'][0]:a for a in inputs['metrics']['assignments']}
    held={tuple(x) for x in inputs['held_out_instances']};ready=[]
    for kind,task,label in [('native','cook_a_brisket','place on frying pan to bar'),
                            ('history','assembling_gift_baskets','place in bow to wicker basket'),
                            ('additional','sorting_books_on_shelf','place on notebook to bookcase')]:
        group=next(g for g in groups if (g['task'],g['subtask'])==(task,label))
        selected_panel=extra if kind=='additional' else panel
        candidates=[s for s in selected_panel['samples'] if (s['task'],s['subtask'])==(task,label) and (s['task'],s['task_instance_id']) not in held]
        candidates.sort(key=lambda s:(s['start'],s['sample_id']))
        if kind=='native':candidates=[s for s in candidates if s['sample_id']=='cook_a_brisket-instance-125-episode-10286-segment-16-interval-0']
        proofs={};samples=[]
        for sample in candidates:
            sid=sample['sample_id'];raw_root=inputs['additional_raw_root'] if kind=='additional' else inputs['raw_data_root']
            cache=runner.load_episode_cache(runner.raw_hdf5_path(raw_root,sample),[sample],allow_unverified_grasp_state=kind=='history')
            raw=dict(cache['sample_evidence'][sid],sha256=cache['sha256'])
            if kind=='additional':
                decoded=decode_start(cache,sample);source=extra['source_evidence'][sid];assert source['raw_sha256']==cache['sha256']
                proof=dict(source_evidence_kind='additional_training_start_preflight',raw_sha256=cache['sha256'],
                    selected_state_sha256=raw['selected_state_sha256'],state_protocol=NATIVE,
                    literal_source_role_binding=preview_source_role_bindings(cache,decoded,sample['predicate']),public_training_source=source)
                proof['source_preflight_sha256']=hashlib.sha256(json.dumps(proof,sort_keys=True).encode()).hexdigest()
            else:
                a=by_task[task];root=Path(a.get('producing_run_root') or inputs['supplementary_run_roots'][str(a['index'])]);manifest=json.loads((root/'manifest.json').read_text())
                path=Path(manifest['results_root'])/'pi05'/task/'json'/(runner._artifact_stem(sample)+'.json')
                body=path.read_bytes();result=json.loads(body)
                if not result['metric_eligible']:continue
                if kind=='history':
                    with h5py.File(cache['path'],'r',locking=False) as file:actions=np.asarray(file['data'][cache['details']['demo_group']]['action'][:sample['start']],dtype=np.float32)
                    prefix=hashlib.sha256(actions.tobytes()).hexdigest();validate_history_result(result,sample,panel['fingerprint'],raw,prefix)
                else:
                    decode_start(cache,sample);validate_baseline_result(result,sample,raw,2048);prefix=None
                proof=dict(raw_sha256=cache['sha256'],selected_state_sha256=raw['selected_state_sha256'],baseline_result_sha256=hashlib.sha256(body).hexdigest(),
                    baseline_metric_eligible=True,state_protocol=HISTORY if kind=='history' else NATIVE,prefix_actions_sha256=prefix,raw_sample_evidence=raw)
            proofs[sid]=proof;samples.append(sample)
            if kind!='additional':break
        assert samples
        evidence=(json.dumps(dict(kind=kind,samples=proofs),sort_keys=True)+'\n').encode()
        directory=output/kind;directory.mkdir();(directory/'source_evidence.json').write_bytes(evidence)
        plan=freeze(selected_panel,group,samples,proofs,inputs['checkpoint_identity'],inputs['held_out_instances'],
            accepted_input_sha256=inputs['accepted_input_sha256'],evidence_manifest_sha256=hashlib.sha256(evidence).hexdigest(),
            repeats=1 if kind=='additional' else 2,seed=1010,max_steps=2048)
        (directory/'parent_panel.json').write_text(json.dumps(selected_panel,indent=2)+'\n')
        (directory/'collection_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
        entry=dict(state='complete',kind=kind,code_root=str(code),source_manifest_sha256=source_identity,output_root=str(directory),raw_data_root=raw_root,plan=plan)
        (directory/'prepared.json').write_text(json.dumps(entry,indent=2)+'\n');ready.append(entry)
    report=dict(state='complete',gates=ready,source_manifest_sha256=source_identity)
    (output/'gates.json').write_text(json.dumps(report,indent=2)+'\n');return report
