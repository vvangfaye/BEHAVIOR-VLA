"""Keep LeRobot PyAV timestamp comparisons in float64 without relaxing tolerance."""
import hashlib
import json
from pathlib import Path
import sys


def apply(root):
    path=Path(root)/'src/lerobot/datasets/video_utils.py'
    text=path.read_text()
    start=text.index('def decode_video_frames_pyav(')
    end=text.find('\ndef ',start+5)
    if end<0:end=len(text)
    block=text[start:end]
    pairs=[('query_ts = torch.tensor(timestamps)','query_ts = torch.tensor(timestamps, dtype=torch.float64)'),
           ('loaded_ts_t = torch.tensor(loaded_ts)','loaded_ts_t = torch.tensor(loaded_ts, dtype=torch.float64)')]
    before=hashlib.sha256(text.encode()).hexdigest()
    for old,new in pairs:
        if new in block:continue
        if block.count(old)!=1:raise ValueError('Unexpected LeRobot timestamp implementation; inspect before patching')
        block=block.replace(old,new)
    result=text[:start]+block+text[end:]
    if result!=text:
        backup=path.with_suffix('.py.before_timestamp_fix')
        if not backup.exists():backup.write_text(text)
        path.write_text(result)
    return {'kind':'lerobot_pyav_float64_timestamp_comparison','tolerance_changed':False,
            'image_or_action_data_changed':False,'source_sha256_before':before,
            'source_sha256_after':hashlib.sha256(result.encode()).hexdigest()}


if __name__=='__main__':print(json.dumps(apply(Path(sys.argv[1])),indent=2))
