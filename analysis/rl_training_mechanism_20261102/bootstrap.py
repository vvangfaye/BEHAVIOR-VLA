import os,subprocess,tempfile
from pathlib import Path

def run(config):
 root=Path(tempfile.mkdtemp(prefix='b1k-mechanism-'))
 source=config.pop('analysis_source');p=root/'analyze.py';p.write_text(source+'\nrun('+repr(config)+')\n')
 env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1'}
 subprocess.run(['/mnt/afs/260010165/behavior_1k/runtime_20261007_1306/envs/b1k-pi05/bin/python','-u',str(p)],env=env,check=True,timeout=3500)
