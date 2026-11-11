"""Read known SERF job outputs; never submit or restart any job."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--artifacts',action='store_true');args=parser.parse_args()
    c=aoss_client();out=HERE/'retrieved';out.mkdir(exist_ok=True)
    reports={}
    try:
        entries=[('init01','behavior1k/serf-reproduction-20261025/init01/')]
        if (HERE/'init02_submission.json').is_file():
            entries.append(('init02','behavior1k/serf-reproduction-20261025/init02/'))
        for attempt in ['eval01','eval02','eval03','eval04','eval05']:
            if (HERE/(attempt+'_submission.json')).is_file():
                entries += [(f'{attempt}_worker{x["rank"]}',f'behavior1k/serf-reproduction-20261025/{attempt}/worker{x["rank"]}/') for x in json.loads((HERE/(attempt+'_submission.json')).read_text())['workers']]
        for name,prefix in entries:
            try:
                data=c.get_object(Bucket='260010165',Key=prefix+'status.json')['Body'].read()
            except c.exceptions.ClientError as e:
                if e.response['Error']['Code'] in {'NoSuchKey','404'}:
                    reports[name]={'state':'not_published'};continue
                raise
            report=json.loads(data);reports[name]=report
            (out/(name+'_status.json')).write_bytes(data)
            if name.startswith('init'):(HERE/(name+'_result.json')).write_bytes(data)
            if args.artifacts:
                try:
                    response=c.get_object(Bucket='260010165',Key=prefix+'artifacts.tar.gz')
                except c.exceptions.ClientError as e:
                    if e.response['Error']['Code'] in {'NoSuchKey','404'}:continue
                    raise
                path=out/(name+'.tar.gz');h=hashlib.sha256()
                with path.open('wb') as f:
                    for b in iter(lambda:response['Body'].read(8*1024**2),b''):f.write(b);h.update(b)
                target=out/name;target.mkdir(exist_ok=True)
                with tarfile.open(path) as tar:
                    for item in tar:
                        rel=Path(item.name)
                        assert item.isfile() and not rel.is_absolute() and '..' not in rel.parts
                        dest=target/rel;dest.parent.mkdir(parents=True,exist_ok=True)
                        with tar.extractfile(item) as source,dest.open('wb') as f:
                            for b in iter(lambda:source.read(8*1024**2),b''):f.write(b)
                (out/(name+'_download.json')).write_text(json.dumps({'sha256':h.hexdigest(),'bytes':path.stat().st_size}))
        print(json.dumps(reports,indent=2))
    finally:c.close()


if __name__=='__main__':main()
