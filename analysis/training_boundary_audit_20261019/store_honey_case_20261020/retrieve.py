"""Retrieve only the four manifest-listed objects, verifying every SHA256."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client


def main():
    plan=json.loads((HERE/'plan.json').read_text())
    client=aoss_client()
    try:
        prefix=plan['destination_prefix']
        bucket=plan['destination_bucket']
        body=client.get_object(Bucket=bucket,Key=prefix+'receipt.json')['Body']
        try: raw=body.read(32769)
        finally: body.close()
        assert len(raw)<=32768
        receipt=json.loads(raw)
        expected=[{k:row[k] for k in ('relative_path','bytes','sha256')} for row in plan['files']]
        assert receipt['complete'] and receipt['source_sha256_verified'] and receipt['files']==expected
        verified=[]
        for row in expected:
            path=HERE/'media'/row['relative_path']
            path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():
                obj=client.get_object(Bucket=bucket,Key=prefix+row['relative_path'])
                assert obj['ContentLength']==row['bytes']
                try: content=obj['Body'].read(row['bytes']+1)
                finally: obj['Body'].close()
                assert len(content)==row['bytes'] and hashlib.sha256(content).hexdigest()==row['sha256']
                with path.open('xb') as stream: stream.write(content)
            assert path.stat().st_size==row['bytes']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
            verified.append(row)
        (HERE/'retrieval_receipt.json').write_text(json.dumps({'source_receipt':receipt,'local_files_verified':verified},indent=2)+'\n')
        print(json.dumps({'local_files_verified':len(verified),'bytes':sum(row['bytes'] for row in verified)}))
    finally: client.close()


if __name__=='__main__':
    try: main()
    except Exception as error:
        print(json.dumps({'error_type':type(error).__name__}))
        raise SystemExit(1)
