import sys,json
from pathlib import Path
sys.path.insert(0,'BEHAVIOR-VLAs/sco');from video_eval_adapter import aoss_client
c=aoss_client();out={}
for p in ['behavior1k/afs-cleanup/','behavior1k/checkpoints/','behavior1k/paired-quick20-reused1688977-step5000-20261014/','behavior1k/retraining_20261013/','behavior1k/expert-fixed20-step1000-20261016/']:
 r=c.list_objects_v2(Bucket='260010165',Prefix=p,Delimiter='/');out[p]={'prefixes':[x['Prefix'] for x in r.get('CommonPrefixes',[])],'objects':[{'key':x['Key'],'size':x['Size']} for x in r.get('Contents',[])],'truncated':r.get('IsTruncated')}
Path('analysis/afs_audit_20261023/backups.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));c.close()
