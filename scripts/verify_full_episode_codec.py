"""Real codec/reader check on actual captured frames, including failed episodes.

This small dataset is a format probe, never a behavior-cloning training source.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from full_episode_trajectory import file_sha256, inspect_full_episode_trajectory
from subtask_trajectory import CAMERAS, array


def verify(path, output):
    from lerobot.datasets import LeRobotDataset
    path, output = Path(path), Path(output)
    info = inspect_full_episode_trajectory(path, require_success=False)
    count = min(64, info['steps'])
    features = {'action': dict(dtype='float32', shape=(23,), names=['action']),
                'observation.state': dict(dtype='float32', shape=tuple(info['state_shape']), names=['state'])}
    features.update({key: dict(dtype='video', shape=tuple(shape), names=['height', 'width', 'channels'])
                     for key, shape in info['camera_shapes'].items()})
    repo_id = 'native_full_episode_format_probe'
    dataset = LeRobotDataset.create(repo_id=repo_id, root=output, fps=info['metadata']['fps'],
                                    features=features, robot_type='r1pro', use_videos=True)
    try:
        with h5py.File(path, 'r') as raw:
            for step in range(count):
                dataset.add_frame({**{key: raw[key][step] for key in features}, 'task': info['metadata']['task_prompt']})
        dataset.save_episode()
    finally:
        dataset.finalize()
    reader = LeRobotDataset(repo_id=repo_id, root=output, download_videos=False, video_backend='pyav')
    assert len(reader) == count
    errors = []
    with h5py.File(path, 'r') as raw:
        for step in range(count):
            decoded = reader[step]
            for key in ('action', 'observation.state'):
                np.testing.assert_allclose(array(decoded[key]), raw[key][step], rtol=0, atol=1e-6)
            for key in CAMERAS:
                image = array(decoded[key])
                if image.shape[0] == 3:
                    image = image.transpose(1, 2, 0)
                if image.dtype != np.uint8:
                    image = np.rint(image * 255).clip(0, 255).astype(np.uint8)
                assert image.shape == raw[key][step].shape
                error = float(np.abs(image.astype(float) - raw[key][step]).mean())
                assert error < 20, 'Decoded camera differs from the real captured source'
                errors.append(error)
    report = dict(kind='native_full_episode_real_codec_probe', real_lerobot_reader=True,
        native_frame_action_alignment_verified=True, decoded_frames=count,
        source_steps=info['steps'], source_trajectory_sha256=file_sha256(path),
        source_content_sha256=info['content_sha256'], source_success=info['result']['success'],
        maximum_image_mae=max(errors), fps=info['metadata']['fps'], ready_for_training=False,
        format_probe_only=True, official_benchmark=False)
    (output / 'format_probe_report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.trajectory, args.output)))
