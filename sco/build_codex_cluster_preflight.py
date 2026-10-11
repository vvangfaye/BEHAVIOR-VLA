"""Freeze a finite cluster image/proxy preflight; no authentication is bundled."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'eval_results/codex_pilot_20261011/cluster_preflight_v2.py'
sources = {name:(ROOT/'sco'/f'{name}.py').read_text() for name in
           ('codex_cluster_proxy', 'codex_auth_handoff')}
driver = '''import hashlib,json,os,shutil,subprocess,sys,types,urllib.request,urllib.error
from pathlib import Path
os.environ['PATH']='/opt/conda/bin:/usr/local/bin:/usr/bin:/bin'
root=Path('/mnt/afs/260010165/behavior_1k/codex_pilot_20261011')
report={'kind':'codex_cluster_preflight_v2','state':'started'}
control=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def publish():
 with control.open(urllib.request.Request(os.environ['CODEX_PILOT_REPORT_PUT'],data=json.dumps(report).encode(),method='PUT'),timeout=30):pass
publish()
for name,source in SOURCES.items():
 module=types.ModuleType(name);sys.modules[name]=module;exec(compile(source,name,'exec'),module.__dict__)
from codex_cluster_proxy import MihomoProxy
from codex_auth_handoff import keygen
report['binaries']={}
for name in ['codex','mihomo','node','openssl']:
 binary=shutil.which(name)
 item={'path':binary}
 if binary:
  version=subprocess.run([binary,'-v' if name=='mihomo' else 'version' if name=='openssl' else '--version'],capture_output=True,text=True,timeout=15)
  item.update(returncode=version.returncode,version=version.stdout[:150].strip())
 report['binaries'][name]=item
report['stage_paths']={name:(root/name).exists() for name in ['bundle-v2','code-v2','original-v2']}
report['state']='binaries_checked';publish()
home=root/'private-codex-home-v1'
report['auth_recipient']=keygen(home)
report['codex_home']=str(home)
report['state']='auth_recipient_ready';publish()
with MihomoProxy(config_dir=Path('/mnt/afs/260010165/container-env/config/mihomo'),
                 run_dir=Path('/var/tmp/codex-preflight-v2'),mihomo_binary='/usr/local/bin/mihomo') as proxy:
 report['proxy']=proxy.receipt.copy();report['state']='proxy_ready';publish()
 env=proxy.environment(os.environ,codex_home=home)
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':env['HTTPS_PROXY']}))
 report['network']={}
 for label,url in [('openai_auth','https://auth.openai.com/.well-known/openid-configuration'),('codex','https://chatgpt.com/backend-api/codex/models?client_version=0.162.1')]:
  try:
   with opener.open(url,timeout=25) as response:report['network'][label]={'http_status':response.status}
  except urllib.error.HTTPError as error:report['network'][label]={'http_status':error.code}
  except Exception as error:report['network'][label]={'error_type':type(error).__name__}
 report['state']='network_checked';publish()
report['proxy_closed']=True;report['state']='preflight_complete';publish()
print(json.dumps({'state':report['state'],'network':report['network']}),flush=True)
'''
driver=driver.replace('SOURCES',repr(sources))
compile(driver,str(OUT),'exec')
with OUT.open('x') as stream:
    stream.write(driver)
print(json.dumps({'source':str(OUT),'sha256':hashlib.sha256(driver.encode()).hexdigest()}))
