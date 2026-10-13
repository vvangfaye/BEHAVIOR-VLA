"""Accept one already-authorized archive delivered over SSH; no AOSS network I/O."""
import argparse
import hashlib
import json
from pathlib import Path

from prepare_retraining_data import Preparation, sha


def main(cache, output, key, prepare_only=False):
    prep=Preparation(cache,output)
    row=next(r for r in prep.inputs['source_archives'] if r['key']==key)
    identity=hashlib.sha256(key.encode()).hexdigest()[:20]
    receipt=output/'receipts'/(identity+'.json')
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if old.get('accepted') and old.get('index_target') and old['source_sha256']==row['sha256']:
            print(json.dumps({'already_accepted':True,'id':identity,'episodes':old['acceptance']['episodes']}));return
    work=cache/'work'/identity;work.mkdir(parents=True,exist_ok=True)
    if prepare_only:
        print(json.dumps({'already_accepted':False,'id':identity,'source_bytes':row['bytes']}));return
    delivered=work/'source.tar.relay'
    source=work/'source.tar'
    if not source.exists():
        if delivered.stat().st_size!=row['bytes'] or sha(delivered)!=row['sha256']:
            raise ValueError('Relayed archive does not match frozen source identity')
        delivered.replace(source)
    if source.stat().st_size!=row['bytes'] or sha(source)!=row['sha256']:
        raise ValueError('Existing source archive identity differs')
    accepted=prep.process(row)
    print(json.dumps({'accepted':accepted['accepted'],'id':identity,'episodes':accepted['acceptance']['episodes']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--key',required=True);p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();main(a.cache,a.output,a.key,a.prepare_only)
