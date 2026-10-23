"""Keep the exact same-allocation continuation sources in the existing private bucket."""
import hashlib
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client

body=(HERE/'bundle_final.tar.gz').read_bytes()
digest=hashlib.sha256(body).hexdigest()
key='behavior1k/rlinf-adapter-20261023/b1k-rlinf-1023-r07/gpu-continuation-'+digest+'.tar.gz'
client=aoss_client()
try:
    client.put_object(Bucket='260010165',Key=key,Body=body,Metadata={'sha256':digest})
    returned=client.get_object(Bucket='260010165',Key=key)['Body'].read(len(body)+1)
    assert returned==body
finally:
    client.close()
receipt={'bucket':'260010165','key':key,'bytes':len(body),'sha256':digest,'readback_verified':True}
(HERE/'source_archive_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
