"""Audit frozen SERF assets and existing rollouts without running a policy."""
from pathlib import Path
import ast
import hashlib
import json
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT/'analysis/serf_reproduction_20261025'
MT = ROOT/'analysis/serf_multitask_20261029'
E20 = ROOT/'analysis/serf_eval20_20261026'
AUTHOR = OLD/'repos/SERF-VLA'


def write(name, value):
    (HERE/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def check_file(path, item):
    if not path.is_file():
        return dict(path=str(path.relative_to(ROOT)), passed=False, error='missing')
    size = path.stat().st_size
    sha = digest(path)
    passed = size == item['size']
    if item.get('sha256'):
        passed &= sha == item['sha256']
    elif item.get('git_blob_sha1'):
        h = hashlib.sha1(f'blob {size}\0'.encode())
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024**2), b''):
                h.update(block)
        passed &= h.hexdigest() == item['git_blob_sha1']
    else:
        raise ValueError('No authoritative digest: '+str(path))
    return dict(path=str(path.relative_to(ROOT)), passed=bool(passed),
                size=size, sha256=sha)


def weights():
    inventory = json.loads((OLD/'public_asset_manifest.json').read_text())['model']['data']
    all_entries = []
    for base, manifest in [(OLD, OLD/'download_manifest.json'),
                           (MT, MT/'checkpoint_manifest.json')]:
        items = [x for x in json.loads(manifest.read_text())['items']
                 if x['destination'].startswith('checkpoints/')]
        entries = [check_file(base/'assets'/item['destination'], item) for item in items]
        all_entries += entries
        print('weights_checked', base.name, len(entries), all(x['passed'] for x in entries), flush=True)
    expected = {x['rfilename'] for x in inventory['siblings'] if x['rfilename'].startswith('pi_')}
    actual = {x['path'].split('/assets/checkpoints/',1)[1] for x in all_entries}
    assert expected == actual, (expected-actual, actual-expected)
    result = dict(model_revision=inventory['sha'], total_files=len(all_entries),
                  bytes=sum(x['size'] for x in all_entries),
                  all_author_files_present=expected==actual,
                  passed=all(x['passed'] for x in all_entries), files=all_entries)
    write('weights.json', result)


def source():
    summaries = {}
    for base in [E20, MT]:
        manifest = json.loads((base/'source_manifest.json').read_text())
        receipts = json.loads((base/'source_upload.json').read_text())
        mismatch = [rel for rel,meta in manifest.items()
                    if digest(base/'runtime'/rel) != meta['sha256']]
        copied = []
        learning = AUTHOR/'src/serf_b1k/learning'
        runtime_learning = base/'runtime/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning'
        for f in learning.rglob('*'):
            if f.is_file() and '__pycache__' not in f.parts:
                rel=f.relative_to(learning)
                if digest(f) != digest(runtime_learning/rel):copied.append(str(rel))
        changed_src=[]
        for f in (base/'runtime/repo/src').rglob('*.py'):
            rel=f.relative_to(base/'runtime/repo')
            original=AUTHOR/rel
            if original.is_file() and digest(original)!=digest(f):
                # Moving an import inside its function is a dependency isolation change.
                def core(path):
                    t=ast.parse(path.read_text())
                    class RemoveImports(ast.NodeTransformer):
                        def visit_Import(self,node):return None
                        def visit_ImportFrom(self,node):return None
                    return ast.dump(RemoveImports().visit(t),include_attributes=False)
                changed_src.append(dict(path=str(rel),unchanged_nonimport_ast=core(original)==core(f)))
        summaries[base.name] = dict(frozen_source_sha256=receipts['source_sha256'],
            frozen_files_verified=len(manifest), local_manifest_mismatches=mismatch,
            archive_sha256_matches=digest(base/'source.tar.gz')==receipts['source_sha256'],
            learning_changed_from_author=copied,policy_source_changes=changed_src)
    write('source_identity.json',summaries)


def rollouts():
    initial_rows=[];tracking_rows=[]
    for rank in [0,1]:
        root=E20/'retrieved'/f'worker{rank}'
        for image_dir in sorted(root.glob('*_image')):
            instance=int(image_dir.name.split('_')[0]);serf_dir=root/f'{instance}_serf'
            a=json.loads((image_dir/'initial.json').read_text());b=json.loads((serf_dir/'initial.json').read_text())
            init_row=dict(instance=instance,shared_initial_sha_matches=a['initial_state_sha256']==b['initial_state_sha256'],
                max_state_reload_delta=max(a['max_initial_state_delta'],b['max_initial_state_delta']),
                image_camera_shapes={k:v['shape'] for k,v in a['cameras'].items()},
                serf_camera_shapes={k:v['shape'] for k,v in b['cameras'].items()},
                action_dim=[a['action_dim'],b['action_dim']], proprio_dim=[a['proprio_dim'],b['proprio_dim']],
                physics_dt=[a['physics_dt'],b['physics_dt']], rendering_dt=[a['rendering_dt'],b['rendering_dt']])
            initial_rows.append(init_row)
            samples=[json.loads(line) for line in (serf_dir/'tracking_audit.jsonl').open()]
            visible=[];hidden=[];wrong=[];boundaries=[];intrinsics=[];target_ids=set()
            for sample in samples:
                if sample['geometry_changed']:boundaries.append(sample['step'])
                intrinsics+=list(sample['intrinsics_relative_error'].values())
                for name,t in sample['targets'].items():
                    target_ids.add(t['train_id'])
                    for online in t['online_ids']:
                        mapped=sample['mapping'].get(str(online))
                        if mapped is not None and mapped!=t['train_id']:
                            wrong.append(dict(step=sample['step'],name=name,online_id=online,mapped=mapped,expected=t['train_id']))
                    if t['movement_m']>=0.05:
                        (visible if t['visible_pixels']>=32 else hidden).append(t['centroid_error_m'])
            def quant(values):
                return dict(n=len(values),median=float(np.median(values)),p95=float(np.percentile(values,95))) if values else dict(n=0)
            tracking_rows.append(dict(instance=instance,samples=len(samples),
                last_step=samples[-1]['step'],tracker_iterations=samples[-1]['tracker_iterations'],
                refreshed_boundaries=len(boundaries),off_boundary_steps=[x for x in boundaries if x%20],
                max_intrinsics_relative_error=max(intrinsics,default=None),
                observed_id_mismatches=len(wrong),mismatch_examples=wrong[:5],
                moved_visible_centroid_error_m=quant(visible),moved_occluded_centroid_error_m=quant(hidden)))
    write('existing_rollout_audit.json',dict(initial_states=initial_rows,tracking=tracking_rows))
    print('rollouts_checked',len(initial_rows),'id_mismatches',sum(x['observed_id_mismatches'] for x in tracking_rows),flush=True)


def main():
    started=time.time()
    source();rollouts();weights()
    write('audit_checks_completed.json',dict(completed_at=time.time(),elapsed_seconds=time.time()-started,
        scope='Read-only source, local asset hashes and existing twenty paired task22 rollouts; no new GPU episodes.'))


if __name__=='__main__':main()
