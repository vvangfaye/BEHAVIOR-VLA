"""Read only three frozen expert parquet projections; no video or cloud jobs."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pyarrow.parquet as pq

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'analysis/training_boundary_audit_20261019'))
from prepare_probe_cache import ObjectReader,TransferBudget,open_source_bucket


def main():
    source=ROOT/'analysis/training_boundary_audit_20261019/pilot_three_tasks_cache_v1/cache_plan.json'
    old=json.loads(source.read_text())
    names=['data/chunk-053/file-005.parquet','data/chunk-065/file-002.parquet','data/chunk-079/file-000.parquet']
    objects=[r for r in old['objects'] if r['relative_path'] in names]
    assert len(objects)==3
    plan={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'bucket':old['source_bucket'],
          'objects':objects,'columns':['action','observation.state','episode_index','frame_index','task_index'],
          'max_bytes':256*1024**2,'max_seconds':600,'new_jobs':0,'new_videos':0}
    path=HERE/'training_column_plan.json'
    if path.exists(): assert json.loads(path.read_text())==plan
    else:path.write_text(json.dumps(plan,indent=2)+'\n')
    budget=TransferBudget(plan['max_bytes'],plan['max_seconds'])
    bucket=open_source_bucket(Path.home()/'.ossutilconfig',plan['bucket'])
    out=HERE/'training_columns';out.mkdir(exist_ok=True)
    records=[]
    for i,r in enumerate(objects):
        dest=out/f'part_{i}.npz'
        if dest.exists():
            records.append(json.loads(dest.with_suffix('.json').read_text()));continue
        table=pq.read_table(ObjectReader(bucket,r,budget),columns=plan['columns'],pre_buffer=False,use_threads=False)
        d=table.to_pydict()
        arrays={k:np.asarray(v,dtype=np.float32 if k in ['action','observation.state'] else np.int64) for k,v in d.items()}
        assert arrays['action'].shape==(len(table),23) and arrays['observation.state'].shape==(len(table),61)
        np.savez_compressed(dest,**arrays)
        record={'object':r,'file':str(dest.relative_to(HERE)),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
                'rows':len(table),'episodes':np.unique(arrays['episode_index']).tolist()}
        dest.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n');records.append(record)
        print(json.dumps({'part':i,'rows':len(table),'episodes':len(record['episodes']),'read_bytes':budget.used_bytes}),flush=True)
    (HERE/'training_column_receipt.json').write_text(json.dumps({'records':records,'bytes':budget.used_bytes,'requests':budget.requests},indent=2)+'\n')


if __name__=='__main__':
    try:main()
    except Exception as e:
        print(json.dumps({'error_type':type(e).__name__}),file=sys.stderr);sys.exit(1)
