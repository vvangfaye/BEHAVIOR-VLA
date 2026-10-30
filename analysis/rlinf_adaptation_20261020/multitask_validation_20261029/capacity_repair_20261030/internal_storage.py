"""Same configured AOSS account, cluster-internal endpoint, no credential output."""
import ast,os,re
from pathlib import Path

def internal_aoss_client():
    import boto3
    from botocore.config import Config
    text=(Path(os.environ.get('SCO_CONFIG',str(Path.home()/'.config/sco')))/'profiles/default.toml').read_text()
    values={k:ast.literal_eval(v) for k,v in re.findall(r'^(access_key_id|access_key_secret)\s*=\s*([\"\'].*[\"\'])\s*$',text,re.M)}
    return boto3.client('s3',endpoint_url='https://aoss-internal.cn-sh-01b.sensecoreapi-oss.cn',
        region_name='cn-sh-01b',aws_access_key_id=values['access_key_id'],aws_secret_access_key=values['access_key_secret'],
        config=Config(signature_version='s3v4',s3={'addressing_style':'path'},proxies={},connect_timeout=8,read_timeout=30))
