"""Read exported data through the actual OpenPI loader in the policy environment.

This is a CPU data-format check. It does not tokenize prompts or train a model.
The evaluator and policy environments stay separate and are never modified.
"""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def run(payload):
    root=Path(payload['dataset_root'])
    assert (root/'_VERIFIED_EXPORT').is_file()
    from openpi.training import config as config_lib,data_loader
    from openpi import transforms
    from lerobot.datasets import LeRobotDataset
    config_lib.ModelTransformFactory=lambda:lambda model:transforms.Group()
    config=config_lib.get_config('pi05_b1k')
    model=dataclasses.replace(config.model,train_subtask_prediction=True,
        max_token_len=512,subtask_action_conditioning='task_and_subtask')
    info=json.loads((root/'meta/info.json').read_text())
    rows=[r for path in sorted((root/'meta/episodes').glob('chunk-*/*.parquet'))
          for r in pq.read_table(path).to_pylist()]
    episodes=[r['episode_index'] for r in rows]
    factory=dataclasses.replace(config.data,repo_id=payload['repo_id'],dataset_root=str(root),
        episodes_index=episodes,video_backend='pyav')
    data_config=factory.create(root,model)
    actual=data_loader.create_b1k_dataset(data_config,model.action_horizon)
    source=LeRobotDataset(repo_id=payload['repo_id'],root=root,download_videos=False,
        video_backend='pyav',delta_timestamps={'action':[i/info['fps'] for i in range(32)]})
    assert len(actual)==len(source)==sum(r['length'] for r in rows)
    labels=pq.read_table(root/'meta/subtask_segments.parquet').to_pylist()
    tasks={r['task_index']:r['task'] for r in map(json.loads,(root/'meta/tasks.jsonl').read_text().splitlines())}
    offset=0;checked=0
    for episode in rows:
        n=episode['length']
        for frame in sorted({0,n//2,n-1}):
            sample,reference=actual[offset+frame],source[offset+frame]
            assert sample['action'].shape==(32,23)
            np.testing.assert_array_equal(sample['action'],reference['action'])
            np.testing.assert_array_equal(sample['observation.state'],reference['observation.state'])
            assert sample['prompt']==tasks[int(reference['task_index'])]
            segment=[s for s in labels if s['episode_index']==episode['episode_index'] and s['start']<=frame<s['end']]
            assert len(segment)==1 and sample['subtask']==segment[0]['subtask']
            for key in info['features']:
                if key.startswith('observation.rgb.'):
                    np.testing.assert_array_equal(sample[key],reference[key])
            checked+=1
        offset+=n
    return dict(complete=True,actual_openpi_training_reader=True,decoded_samples=checked,
        frames=len(actual),episodes=len(rows),action_horizon=32,tokenization_verified=False,
        gpu_training_verified=False,repo_id=payload['repo_id'],dataset_root=str(root))
