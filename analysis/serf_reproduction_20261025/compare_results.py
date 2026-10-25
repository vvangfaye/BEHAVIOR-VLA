"""Compare only completed, configuration-bound paired SERF episodes."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    records={}
    for path in args.root.rglob('result.json'):
        item=json.loads(path.read_text())
        if item.get('condition') not in {'image','serf'} or item.get('instance') not in {73,282,256}:
            continue
        if item.get('complete') is not True:
            continue
        key=(item['instance'],item['condition'])
        if key in records:
            raise ValueError(f'Multiple completed results for {key}; resolve attempt lineage before comparing')
        assert isinstance(item['success'],bool) and 0<=item['score']<=1
        initial=path.parent/'initial.json'
        assert initial.is_file(),f'Missing initial state evidence: {path}'
        item['initial']=json.loads(initial.read_text());item['source']=str(path)
        records[key]=item
    rows=[]
    for instance in [73,282,256]:
        a,b=records.get((instance,'image')),records.get((instance,'serf'))
        if a and b:
            assert a['initial']['initial_state_sha256']==b['initial']['initial_state_sha256']
            assert a['initial']['max_initial_state_delta']<1e-4 and b['initial']['max_initial_state_delta']<1e-4
        rows.append({'instance':instance,'image':a,'serf':b,'score_delta':b['score']-a['score'] if a and b else None})
    matched=[r for r in rows if r['score_delta'] is not None]
    result={'required_episodes':6,'completed_episodes':len(records),'complete':len(records)==6,'rows':rows,
        'paired_instances':len(matched),'success_failure_flips':sum(r['image']['success']!=r['serf']['success'] for r in matched)}
    if len(records)==6:
        result['image_mean_score']=sum(r['image']['score'] for r in rows)/3
        result['serf_mean_score']=sum(r['serf']['score'] for r in rows)/3
        result['image_successes']=sum(r['image']['success'] for r in rows)
        result['serf_successes']=sum(r['serf']['success'] for r in rows)
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
