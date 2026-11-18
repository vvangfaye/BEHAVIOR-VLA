"""Verify the paired runs and write a compact result for this pilot."""
import argparse
import json
from collections import Counter
from pathlib import Path
from analysis_report import analyze, read_rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Original paired run root")
    root = parser.parse_args().output_dir
    decision=json.loads((root/'agent_decision.json').read_text())
    seen=set()
    counts={}
    rows={}
    for split,expected in [('discovery',8),('holdout',16)]:
        folder=root/split
        pairs,digest=read_rows(folder/'pairs.json')
        if split=='discovery':
            assert digest==decision.get("discovery_file_sha256", decision.get("discovery_sha256"))
        batches=[json.loads(f.read_text()) for f in sorted(folder.glob('batch_*.json'))]
        assert len(pairs)==expected and len({x['seed'] for x in pairs})==expected
        assert pairs==[pair for b in batches for pair in b['matched']]
        all_roots=Counter()
        matched=Counter()
        transitions=[]
        for b in batches:
            for old,new,pair in zip(b['root_old'],b['root_new'],b['matched'],strict=True):
                assert old['seed']==new['seed']==pair['seed']
                assert old['initial_hash']==new['initial_hash'] and old['initial_hash'] not in seen
                seen.add(old['initial_hash'])
                outcome=(bool(old['success_once']),bool(new['success_once']))
                names={(True,True):'both_success',(False,False):'both_failure',(True,False):'old_only',(False,True):'new_only'}
                all_roots[names[outcome]]+=1
                if pair['signed_front_m'] is None:
                    matched['unavailable']+=1
                else:
                    snap=json.loads((folder/f"{old['seed']}_handoff.json").read_text())
                    assert snap['initial_hash']==old['initial_hash'] and not snap['success_once']
                    assert len(snap['actions'])==snap['step']==pair['handoff_step']
                    matched[names[(pair['old_success'],pair['new_success'])]]+=1
                if outcome[0]!=outcome[1]:
                    transitions.append(dict(seed=old['seed'],root_result=names[outcome],
                        new_handoff_step=new['handoff_step'],old_handoff_step=old['handoff_step'],
                        new_end_metrics=new['end_metrics'],matched_result=[pair['old_success'],pair['new_success']]))
        counts[split]={'from_reset':dict(all_roots),'from_common_handoff':dict(matched),'root_flips':transitions}
        rows[split]=pairs
    report=analyze(rows['discovery'],rows['holdout'])
    assert report['cutoff_m']==decision['cutoff_m']
    result={'dataset':'24 fresh LIBERO-Long drawer starts, PPO vs SFT','counts':counts,
            'unique_initial_state_hashes':len(seen),'association_test':report,
            'training_performed':False}
    (root/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,indent=1,ensure_ascii=False))


if __name__ == "__main__":
    main()
