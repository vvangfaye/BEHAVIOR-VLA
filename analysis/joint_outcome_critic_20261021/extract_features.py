"""Frozen original-checkpoint SigLIP features from archived observations only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import types
import argparse

import numpy as np
import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--follow-seconds', type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(8)
    modules = ROOT/'analysis/rlinf_adaptation_20261020/upstream/rlinf/models/embodiment/openpi/modules'
    package = types.ModuleType('frozen_vision'); package.__path__ = [str(modules)]
    sys.modules['frozen_vision'] = package
    from frozen_vision.siglip import SigLIPViT
    checkpoint = ROOT/'analysis/rlinf_adaptation_20261020/checkpoint1688977_openpi/model.safetensors'
    model = SigLIPViT(num_classes=2048, dtype_mm='float32').eval()
    with safe_open(checkpoint, framework='pt') as src:
        weights = {k.removeprefix('img.'): src.get_tensor(k) for k in src.keys() if k.startswith('img.')}
    model.load_state_dict(weights, strict=True)
    del weights
    for p in model.parameters(): p.requires_grad_(False)
    camera_keys = ['observation.rgb.zed_link_camera_0', 'observation.rgb.left_realsense_link_camera_0',
                   'observation.rgb.right_realsense_link_camera_0']
    dest = HERE/'features'; dest.mkdir(exist_ok=True)
    records = []
    started = time.monotonic()
    target_count = len(json.loads((HERE/'read_receipt.json').read_text())['files'])
    seen = set()
    def available():
        while True:
            for p in sorted((HERE/'frames').glob('*.provenance.json')):
                if p.name not in seen:
                    seen.add(p.name)
                    yield json.loads(p.read_text())
            if len(seen) == target_count:
                break
            if not args.follow_seconds:
                break
            if time.monotonic()-started > args.follow_seconds:
                raise TimeoutError('Frame collection incomplete; existing features preserved')
            time.sleep(2)
    for episode in available():
        source = HERE/episode['npz']
        expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        assert expected_hash == episode['npz_sha256']
        out = dest/(source.stem+'.npz')
        meta = out.with_suffix('.json')
        if out.exists() and meta.exists():
            record = json.loads(meta.read_text())
            assert record['source_sha256'] == expected_hash
            assert record['features_sha256'] == hashlib.sha256(out.read_bytes()).hexdigest()
            records.append(record); continue
        with np.load(source, allow_pickle=False) as data:
            rgb = np.stack([data[k] for k in camera_keys], axis=1)
            assert rgb.shape == (3, 3, 224, 224, 3)
            image = torch.from_numpy(rgb.reshape(9,224,224,3)).float()/127.5-1
            state = data['observation.state'].copy()
        with torch.inference_mode():
            # Batch one limits CPU memory without changing the frozen encoder.
            feats = torch.cat([model(x[None])[0].mean(dim=1) for x in image]).numpy().reshape(3,3,2048)
        assert np.isfinite(feats).all()
        np.savez_compressed(out, vision=feats, state=state, frames=np.array([0,1500,3000]))
        record = {'task': episode['task'], 'instance': episode['instance'], 'source_sha256': expected_hash,
                  'features': str(out.relative_to(HERE)), 'features_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
                  'shape': list(feats.shape)}
        meta.write_text(json.dumps(record, indent=2)+'\n')
        records.append(record)
        print(json.dumps({'completed': len(records), 'total': target_count,
                          'seconds': round(time.monotonic()-started,2)}), flush=True)
    manifest = {'records': records, 'encoder': 'original1688977 frozen SigLIP, converted weights, strict load',
                'checkpoint_contract_sha256': hashlib.sha256(checkpoint.with_name('b1k_checkpoint.json').read_bytes()).hexdigest(),
                'pooling': 'mean spatial tokens separately per camera and timestamp',
                'inputs_excluded': ['actions','terminal predicates','future frames','episode length','subtask labels'],
                'frames': [0,1500,3000], 'seconds': time.monotonic()-started,
                'complete': len(records) == target_count}
    name = 'feature_manifest.json' if manifest['complete'] else 'feature_manifest_partial.json'
    (HERE/name).write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    main()
