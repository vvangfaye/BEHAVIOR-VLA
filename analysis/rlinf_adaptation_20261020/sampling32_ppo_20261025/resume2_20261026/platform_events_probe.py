"""Read own stopped jobs' platform events using the existing SCO identity."""
import base64
import hashlib
import hmac
import json
import os
import ssl
from pathlib import Path
import tomllib
from email.utils import formatdate
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
HOST = 'aec2.cn-sh-01.sensecoreapi.cn'
SUBSCRIPTION = '0197ee17-b6eb-7846-b2b4-a77c5f509b92'


def get(path, credentials):
    date = formatdate(usegmt=True)
    message = f'date: {date}\nhost: {HOST}\n@request-target: get {path}'
    signature = base64.b64encode(hmac.new(credentials['access_key_secret'].encode(),
        message.encode(), hashlib.sha256).digest()).decode()
    authorization = (f'hmac accesskey="{credentials["access_key_id"]}", '
        f'algorithm="hmac-sha256", headers="date host @request-target", signature="{signature}"')
    request = urllib.request.Request('https://'+HOST+path,
        headers={'Date':date,'Host':HOST,'Authorization':authorization,'Accept':'application/json'})
    try:
        context = ssl.create_default_context(cafile='/etc/ssl/certs/ca-certificates.crt')
        with urllib.request.urlopen(request,timeout=30,context=context) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        try: body=json.loads(error.read())
        except Exception: body={'message':'Non-JSON response'}
        return error.code, body


def main():
    credentials=tomllib.loads((Path.home()/'.config/sco/profiles/default.toml').read_text())
    result=[]
    for rank in (1,0):
        receipt=json.loads((HERE/f'r{rank}/submission.json').read_text())
        base=(f'/compute/acp/data/v2/subscriptions/{SUBSCRIPTION}/resourceGroups/default/'
              f'zones/cn-sh-01z/workspaces/share-space/trainingJobs/{receipt["job"]}')
        status,job=get(base,credentials)
        if status!=200:
            print(json.dumps({'rank':rank,'identity_http_status':status,'error':job}));break
        assert job['uid']==receipt['uid'] and job['ownership']['user_name']=='260010165'
        row={'rank':rank,'name':receipt['job'],'uid':receipt['uid'],'state':job['state'],'requests':[]}
        for suffix in ('/events?page_size=100','/workers/pt-'+receipt['uid']+'-worker-0/events?page_size=100'):
            status,body=get(base+suffix,credentials)
            row['requests'].append({'path_suffix':suffix,'status':status,'body':body})
            print(json.dumps({'rank':rank,'path':suffix,'status':status,'body':body},ensure_ascii=False))
        result.append(row)
    path=HERE/'platform_events.json'
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    with os.fdopen(fd,'w') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)


if __name__=='__main__':main()
