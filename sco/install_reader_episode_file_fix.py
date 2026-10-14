"""Prune parquet files by canonical episode metadata before identical row filtering."""
import hashlib,json
from pathlib import Path
import sys

OLD='''        hf_dataset = load_nested_dataset(self.root / "data", features=features, episodes=self.episodes)
'''
NEW='''        if self.episodes is not None:
            import pyarrow.dataset as pa_ds
            from .io_utils import SuppressProgressBars
            paths = sorted({self.root / self._meta.get_data_file_path(ep) for ep in self.episodes})
            filters = pa_ds.field("episode_index").isin(self.episodes)
            with SuppressProgressBars():
                hf_dataset = datasets.Dataset.from_parquet([str(path) for path in paths], filters=filters, features=features)
        else:
            hf_dataset = load_nested_dataset(self.root / "data", features=features, episodes=self.episodes)
'''

def apply(root):
    path=Path(root)/'src/lerobot/datasets/dataset_reader.py';text=path.read_text();before=hashlib.sha256(text.encode()).hexdigest()
    if NEW not in text:
        if text.count(OLD)!=1:raise ValueError('Unexpected LeRobot reader source; inspect before patching')
        text=text.replace(OLD,NEW);path.write_text(text)
    return dict(kind='lerobot_exact_episode_parquet_selection',episode_row_filter_unchanged=True,training_samples_unchanged=True,source_sha256_before=before,source_sha256_after=hashlib.sha256(text.encode()).hexdigest())

if __name__=='__main__':print(json.dumps(apply(Path(sys.argv[1])),indent=2))
