"""Read exact old job/pod events using the existing CLI identity and endpoint."""
import base64
import concurrent.futures
from email.utils import formatdate
import hashlib
import hmac
import json
from pathlib import Path
import re
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
NAME = 'b1k-q20k10-1014-candidate-04'
UID = '96a9f2f9f1c5470d850f886c37c1c7c4'
SUB = '0197ee17-b6eb-7846-b2b4-a77c5f509b92'
PROFILE = tomllib.loads(Path('/private/tmp/behavior-sco-config/profiles/default.toml').read_text())
META = Path('/private/tmp/behavior-sco-data/meta/meta.yaml').read_text()
ENDPOINT = re.search(r'^  acp: (https://[^\s]+)$', META, re.M)[1]
assert ENDPOINT == 'https://aec2.cn-sh-01.sensecoreapi.cn'


def get(path):
    # Signature scheme verified against current official API synopsis.
    date = formatdate(usegmt=True)
    host = urllib.parse.urlparse(ENDPOINT).netloc
    text = f'date: {date}\nhost: {host}\n@request-target: get {path}'
    signature = base64.b64encode(hmac.new(PROFILE['access_key_secret'].encode(),
                                          text.encode(), hashlib.sha256).digest()).decode()
    auth = (f'hmac accesskey="{PROFILE["access_key_id"]}", algorithm="hmac-sha256", '
            f'headers="date host @request-target", signature="{signature}"')
    request = urllib.request.Request(ENDPOINT + path, headers={
        'Date': date, 'X-Date': date, 'Authorization': auth, 'Host': host,
        'Accept': 'application/json'}, method='GET')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=25) as response:
            return {'status': response.status, 'data': json.loads(response.read(8*1024**2))}
    except urllib.error.HTTPError as error:
        raw = error.read(4096)
        try:
            body = json.loads(raw)
            detail = {k: body.get(k) for k in ('code', 'message')}
        except ValueError:
            detail = {'response_not_json': True}
        return {'status': error.code, 'error': detail}
    except OSError as error:
        return {'error_type': type(error).__name__}


def main():
    old = f'/ait/data/v1/subscriptions/{SUB}/resourceGroups/default/regions/cn-sh-01/workspaces/share-space/trainingJobs/{NAME}'
    paths = {'job_events': old+'/events',
             'pod_events': old+f'/pods/pt-{UID}-worker-0/events',
             'worker_detail': old+f'/workers/pt-{UID}-worker-0'}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = dict(zip(paths, pool.map(get, paths.values())))
    result = dict(observed_at=time.time(), job_name=NAME, uid=UID, read_only=True,
                  endpoint=ENDPOINT, interface_paths_source='installed official SCO binary descriptors',
                  signature_source='https://www.sensecore.cn/help/docs/ApiDoc/synopsis', observations=results)
    (ROOT/'platform_events.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
