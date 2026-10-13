"""Verify direct, policy-compliant AOSS access before resuming data transfers."""
import json
from pathlib import Path
import sys
import time
import urllib.request

from prepare_retraining_data import Preparation, save


def main(cache, output):
    prep = Preparation(cache, output)
    row = prep.inputs['source_archives'][0]
    request = urllib.request.Request(row['signed_get_url'], headers={
        'If-Match': row['etag'], 'Range': 'bytes=0-4095'})
    with prep.opener.open(request, timeout=30) as response:
        if (response.status != 206 or response.headers.get('ETag') != row['etag']
                or response.headers.get('Content-Range') != f'bytes 0-4095/{row["bytes"]}'
                or len(response.read()) != 4096):
            raise ValueError('Direct source identity/readback differs')
    result = {'direct_source_read_verified':True, 'proxy_used':False,
              'source_etag':row['etag'],'verified_at':time.time(),'training_executed':False}
    save(output/'source_connectivity.json',result)
    blocker=output/'source_transfer_blocker.json'
    if blocker.exists(): blocker.rename(output/f'source_transfer_blocker.resolved.{int(time.time())}.json')
    print(json.dumps(result))


if __name__=='__main__': main(Path(sys.argv[1]),Path(sys.argv[2]))
