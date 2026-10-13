"""A finite data-only continuation after explicit network preflight succeeds."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from check_retraining_connectivity import main as preflight
from finalize_retraining_data import finalize
from prepare_retraining_data import Preparation, save


def main(cache, output):
    if (output/'source_transfer_blocker.json').exists():
        raise RuntimeError('Wait for the domain allowlist; run direct connectivity preflight first')
    preflight(cache, output)
    prep=Preparation(cache,output)
    if (output/'STOP').exists(): raise RuntimeError('Data preparation STOP is present')
    failures=[]
    # Existing compact, verified subtask exports first; full native episodes next.
    for row in prep.inputs['source_archives']:
        if (output/'STOP').exists():
            prep.status('stopped_by_user'); return
        try: prep.process(row)
        except Exception as error:
            message=re.sub(r'https?://[^\s\"\']+','[redacted URL]',str(error))
            failure={'source_key':row['key'],'id':row['id'],'type':type(error).__name__,
                     'message':message[:1500],'time':time.time(),'training_executed':False}
            failures.append(failure); save(output/'failures'/f'{len(failures):04d}.json',failure)
            if type(error).__name__=='HTTPError' or 'Insufficient bounded scratch' in message:
                prep.status('blocked'); break
    prep.status('data_processing_finished' if not failures else 'data_processing_has_failures')
    prep.lock.close()
    print(json.dumps(finalize(cache,output)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.cache,a.output)
