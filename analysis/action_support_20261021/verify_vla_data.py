"""Check the actual cache-to-model action and camera contracts on a real row."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import types

import numpy as np
import torch

from vla_runtime import ROOT
from b1k_rlinf.io import CheckpointIO, IMAGE_KEYS, resize_image
from b1k_rlinf.model import JointPi0RL
from b1k_rlinf.tokenizer import JointTokenizer
from vla_data import VlaPilotData, digest


def main():
    here = Path(__file__).resolve().parent
    cache = ROOT / 'analysis/training_boundary_audit_20261019/pilot_three_tasks_cache_v1'
    record = json.loads((cache / 'cache_manifest.json').read_text())['records'][0]
    path = cache / record['npz_path']
    assert digest(path) == record['npz_sha256']
    with np.load(path) as z:
        source = {k: z[k].copy() for k in z.files}
    base = SimpleNamespace(
        checkpoint_io=CheckpointIO(ROOT / 'analysis/rlinf_adaptation_20261020/checkpoint1688977_openpi/norm_stats.json', 'checkpoint61'),
        joint_tokenizer=JointTokenizer(ROOT / 'analysis/retraining_20261015_comet_rft60_40/launch_bundle/inputs/paligemma_tokenizer.model'),
        device=torch.device('cpu'))
    base._token_observation = types.MethodType(JointPi0RL._token_observation, base)
    camera_keys = ('observation.rgb.zed_link_camera_0', 'observation.rgb.left_realsense_link_camera_0',
                   'observation.rgb.right_realsense_link_camera_0')
    rgb = np.stack([resize_image(source[k]) for k in camera_keys])[None]
    with tempfile.TemporaryDirectory(prefix='b1k-vla-data-check-') as directory:
        directory = Path(directory)
        np.save(directory / 'rgb.npy', rgb)
        np.savez(directory / 'raw_targets.npz', state=source['observation.state'][None],
                 actions=source['action'][None], active=(np.ptp(source['action'], axis=0) > 0)[None],
                 episode=source['episode_index'][None], frame=source['frame_index'][None], fold=np.array([2]))
        plan = {'raw_targets': {'path': 'raw_targets.npz', 'sha256': digest(directory / 'raw_targets.npz')},
                'samples': [{'episode_index': int(source['episode_index']), 'frame_index': int(source['frame_index']),
                             'prompt': str(source['prompt']), 'subtask': str(source['subtask'])}]}
        (directory / 'data_plan.json').write_text(json.dumps(plan))
        receipt = {'complete': True, 'image_file': 'rgb.npy', 'image_sha256': digest(directory / 'rgb.npy'),
                   'data_plan_sha256': digest(directory / 'data_plan.json'), 'shape': list(rgb.shape)}
        (directory / 'image_receipt.json').write_text(json.dumps(receipt))
        data = VlaPilotData(directory, base)
        obs, target, active = data.batch([0])
        decoded = base.checkpoint_io.decode(target, obs.state, 32)
        expected = torch.from_numpy(source['action'][None])
        torch.testing.assert_close(decoded, expected, atol=3e-6, rtol=3e-6)
        for channel, name in enumerate(IMAGE_KEYS):
            torch.testing.assert_close(obs.images[name], torch.from_numpy(rgb[:, channel]).float() / 255 * 2 - 1,
                                       atol=0, rtol=0)
        np.testing.assert_array_equal(active.numpy(), (np.ptp(source['action'], axis=0) > 0)[None])
        result = {'passed': True, 'real_sample_sha256': record['npz_sha256'],
                  'raw_action_roundtrip_max_error': float((decoded - expected).abs().max()),
                  'camera_order_verified': list(IMAGE_KEYS), 'target_shape': list(target.shape),
                  'raw_activity_labels_preserved': True, 'padded_model_channels_zero': bool((target[..., 23:] == 0).all()),
                  'scope': 'Real-row data contract only; temporary one-row fixture is not a training dataset'}
    (here / 'vla_data_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
