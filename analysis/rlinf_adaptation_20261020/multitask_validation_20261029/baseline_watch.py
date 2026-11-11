"""Summarize the two bounded baseline panels without allocating resources."""
from datetime import datetime,timezone
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent

def main():
    while True:
        tasks={}
        for slug in ['trash','water']:
            path=ROOT/slug/'latest_status.json'
            d=json.loads(path.read_text()) if path.exists() else {}
            workers={}
            for w in d.get('workers',[]):
                if w['phase']!='baseline':continue
                if w['rank'] not in workers or w['attempt']>workers[w['rank']]['attempt']:workers[w['rank']]=w
            tasks[slug]={'state':d.get('state','starting'),'source_observed_at':d.get('observed_at'),
                         'completed_baseline_cases':sum((w.get('progress') or {}).get('completed_evaluations',0) for w in workers.values()),
                         'expected_baseline_cases':24,'workers':[{k:w.get(k) for k in ['job','uid','state','rank','attempt']} for w in workers.values()]}
        summary={'observed_at':datetime.now(timezone.utc).isoformat(),'state':'baseline_validation',
                 'training_started':False,'training_requires_capacity_clearance':True,'tasks':tasks}
        p=ROOT/'status.json';tmp=p.with_suffix('.watch.tmp');tmp.write_text(json.dumps(summary,indent=2)+'\n');tmp.replace(p)
        if all(t['state']=='baseline_complete_waiting_for_training_capacity' for t in tasks.values()):return
        time.sleep(30)

if __name__=='__main__':main()
