"""Offline-sign one frozen AOSS object into a private file; never print a URL."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import tomllib

import boto3
from botocore.config import Config


def main(sources,key,output):
    manifest=json.loads(sources.read_text())
    if manifest['authorization']['training_allowed'] is not False:
        raise ValueError('Not a data-only authorization')
    record=next((r for r in manifest['source_archives'] if r['key']==key),None)
    if record is None:raise ValueError('Object is outside the frozen data-preparation scope')
    config=tomllib.loads((Path.home()/'.config/sco/profiles/default.toml').read_text())
    client=boto3.client('s3',endpoint_url='https://aoss.cn-sh-01b.sensecoreapi-oss.cn',region_name='cn-sh-01b',
        aws_access_key_id=config['access_key_id'],aws_secret_access_key=config['access_key_secret'],
        config=Config(signature_version='s3v4',s3={'addressing_style':'path'}))
    url=client.generate_presigned_url('get_object',Params={'Bucket':'260010165','Key':record['key']},ExpiresIn=604800)
    output.parent.mkdir(parents=True,exist_ok=True)
    fd,path=tempfile.mkstemp(prefix='.signed-',dir=output.parent)
    try:
        os.fchmod(fd,0o600)
        with os.fdopen(fd,'w') as stream:json.dump({'key':key,'signed_get_url':url},stream)
        os.replace(path,output)
    finally:
        if os.path.exists(path):os.unlink(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sources',type=Path,required=True);p.add_argument('--key',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.sources,a.key,a.output)
