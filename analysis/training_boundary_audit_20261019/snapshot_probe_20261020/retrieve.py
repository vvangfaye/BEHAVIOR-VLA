"""Retrieve the exact probe result without changing any cluster resource."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
KEY='behavior1k/snapshot-restore-audit-20261020/b1k-restore-probe-1020-p01/result.tar.gz'
LIMIT=256*1024**2


def main():
    sys.path.insert(0,str(PROJECT/'BEHAVIOR-VLAs/sco'))
    from video_eval_adapter import aoss_client
    client=aoss_client()
    try:
        try:head=client.head_object(Bucket='260010165',Key=KEY)
        except client.exceptions.ClientError as error:
            if error.response.get('ResponseMetadata',{}).get('HTTPStatusCode')==404:
                print(json.dumps({'state':'result_not_uploaded','not_a_job_failure':True}));return
            raise
        size=head['ContentLength']
        if not 0<size<LIMIT:raise ValueError('Unexpected result size')
        response=client.get_object(Bucket='260010165',Key=KEY)
        stream=response['Body']
        try:body=stream.read(size+1)
        finally:stream.close()
        if len(body)!=size:raise ValueError('Incomplete result transfer')
    finally:client.close()
    archive=HERE/'result.tar.gz'
    if archive.exists():
        if archive.read_bytes()!=body:raise ValueError('Published result changed')
    else:archive.write_bytes(body)
    root=HERE/'retrieved'
    if root.exists():raise ValueError('Result directory already exists; inspect the accepted receipt')
    with tarfile.open(archive,'r:gz') as tar:
        names=set();total=0
        for member in tar:
            path=PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or member.name in names:
                raise ValueError('Unsafe or repeated result path')
            names.add(member.name)
            if member.isdir():continue
            if not member.isfile() or member.size<0:raise ValueError('Nonregular result member')
            if member.name not in ('driver_status.json','runtime_receipt.json','worker.log') and not member.name.startswith('results/'):
                raise ValueError('Unexpected result member')
            total+=member.size
            if total>LIMIT:raise ValueError('Expanded result exceeds bound')
        root.mkdir()
        for member in tar.getmembers():
            target=root/member.name
            if member.isdir():target.mkdir(parents=True,exist_ok=True);continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as source,target.open('xb') as out:
                while block:=source.read(1024**2):out.write(block)
    status=json.loads((root/'driver_status.json').read_text())
    receipt={'state':'retrieved','key':KEY,'bytes':size,'sha256':hashlib.sha256(body).hexdigest(),
             'worker_state':status['state'],'physical_validation_complete':False}
    if status['state']=='complete':
        result=json.loads((root/'results/result.json').read_text())
        assert result['model_calls']==0 and result['training'] is False
        assert result['historical_rollout_fidelity_claimed'] is False
        receipt.update(physical_validation_complete=True,
                       restore_checks_passed=result['all_checks_passed'],
                       held_object_present=result['held_object_present'])
    (HERE/'retrieval_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
