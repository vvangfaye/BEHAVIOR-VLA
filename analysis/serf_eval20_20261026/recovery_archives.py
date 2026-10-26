"""Find completed episodes through a shard's complete recovery chain."""
import io
import json
import tarfile


def completed_archives(client, bucket, receipts, cases, sign):
    prefixes=list(dict.fromkeys(r['result_key'].rsplit('/',1)[0]+'/' for r in receipts))
    completed={}
    archives=[]
    for instance,condition in cases:
        case=f'{instance}_{condition}'
        for prefix in prefixes:
            key=prefix+'episodes/'+case+'.tar.gz'
            try:
                client.head_object(Bucket=bucket,Key=key)
            except client.exceptions.ClientError as exc:
                if exc.response['Error']['Code'] in {'404','NoSuchKey'}:
                    continue
                raise
            data=client.get_object(Bucket=bucket,Key=key)['Body'].read()
            with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
                result=json.load(archive.extractfile(case+'/result.json'))
            assert result['complete'] and result['instance']==instance and result['condition']==condition
            completed[case]=result
            archives.append({'key':key,'url':sign(key)})
            break
    return completed,archives
