import hashlib,json,os,subprocess,sys,tarfile,time,urllib.request
from pathlib import Path
cfg=json.loads(Path(__file__).with_name('bootstrap_input.json').read_text())
root=Path('/var/tmp/wangfaye/two-source-cpu-bootstrap')/str(os.getpid());root.mkdir(parents=True,exist_ok=False)
output=Path(cfg['output']);output.mkdir(parents=True,exist_ok=True)
try:
 archive=root/'runtime.tar.gz'
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 digest=hashlib.sha256();size=0
 with opener.open(cfg['get'],timeout=120) as response,archive.open('wb') as stream:
  while True:
   block=response.read(1024**2)
   if not block:break
   size+=len(block);assert size<=cfg['bytes'];stream.write(block);digest.update(block)
 assert size==cfg['bytes'] and digest.hexdigest()==cfg['sha256']
 bundle=root/'bundle';bundle.mkdir()
 with tarfile.open(archive) as tar:
  for member in tar.getmembers():assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
  tar.extractall(bundle)
 print('Frozen CPU runtime downloaded and SHA verified; no GPU or optimizer',flush=True)
 result=subprocess.run([sys.executable,str(bundle/'four_source_cpu_probe.py')],cwd=bundle,timeout=10800)
 raise SystemExit(result.returncode)
except Exception as error:
 (output/'bootstrap_transport_failure.json').write_text(json.dumps(dict(error_type=type(error).__name__,observed_at=time.time(),training=False)))
 raise
