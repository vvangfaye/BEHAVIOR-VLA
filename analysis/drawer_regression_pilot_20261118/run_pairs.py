"""Measure paired root and handoff continuations using the same per-step policy noise."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

SEEDS = {'discovery': list(range(181101,181109)), 'holdout': list(range(182101,182117))}
HORIZON = 520

def save(path, obj):
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, allow_nan=False, default=lambda x:x.tolist())+'\n')
    tmp.replace(path)

def noise(seed, mode, index):
    return int.from_bytes(hashlib.sha256(f'drawer18/{seed}/{mode}/{index}'.encode()).digest()[:4],'big')

def frame(obs, path):
    Image.fromarray(np.asarray(obs['main_images'],dtype=np.uint8)).save(path, quality=83)

def roll(envs, seeds, policy, starts, snaps, out, arm, stall_probe=None):
    prefix = snaps is None
    obs = [env.reset(s) if prefix else env.restore(snaps[i]) for i,(s,env) in enumerate(zip(seeds,envs))]
    starts_geom = [env.metrics() for env in envs]
    first_hash = [env.initial_hash for env in envs]
    hits = [False]*len(seeds)
    first_at = [None]*len(seeds)
    captured = [None]*len(seeds)
    measures = [None]*len(seeds)
    boundaries = [None]*len(seeds)
    for delta in range(0,HORIZON-min(starts),10):
        active = [i for i in range(len(seeds)) if not hits[i] and starts[i]+delta<HORIZON]
        if not active: break
        phase = 'root' if prefix else 'handoff'
        chunk = policy.actions([obs[i] for i in active],obs[active[0]]['task_descriptions'],
                   noise_seeds=[noise(seeds[i],phase,starts[i]+delta) for i in active])
        for j,i in enumerate(active):
            for h,action in enumerate(chunk[j]):
                obs[i],metric=envs[i].step(action)
                if metric['success']:
                    hits[i],first_at[i]=True,starts[i]+delta+h+1
                    break
            if stall_probe is not None and not hits[i]:
                stall_probe.observe_boundary(i, metric, envs[i])
            if prefix and not hits[i] and captured[i] is None and metric['bowl_in'] and not metric['drawer_closed'] and delta+10<HORIZON:
                captured[i]=envs[i].snapshot()
                measures[i]=metric
                boundaries[i]=delta+10
                frame(obs[i],out/f'{seeds[i]}_{arm}_handoff.jpg')
            if delta in (0,90,190,290,390,510) or hits[i]:
                frame(obs[i],out/f'{seeds[i]}_{arm}_{starts[i]+delta:03}.jpg')
    rows=[dict(seed=s,success_once=hits[i],first_success_step=first_at[i],initial_hash=first_hash[i],start_metrics=starts_geom[i],
               handoff_step=boundaries[i],handoff_metrics=measures[i],end_metrics=envs[i].metrics()) for i,s in enumerate(seeds)]
    return rows,captured

def add_policy_arguments(parser):
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--config-name", required=True, help="Hydra evaluation config name")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--ppo-checkpoint", type=Path, required=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=SEEDS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--stall-probe", action="store_true", help="Capture secondary holdout landmarks")
    add_policy_arguments(parser)
    args = parser.parse_args()
    if args.batch < 1 or (args.stall_probe and args.split != "holdout"):
        parser.error("batch must be positive and the stall probe requires holdout")
    if args.split == "holdout":
        from analysis_report import read_rows

        decision_path = args.output_dir / "agent_decision.json"
        pairs_path = args.output_dir / "discovery" / "pairs.json"
        if not decision_path.is_file() or not pairs_path.is_file():
            parser.error("holdout requires the discovery pairs and frozen agent_decision.json")
        decision = json.loads(decision_path.read_text())
        _, discovery_hash = read_rows(pairs_path)
        frozen_hash = decision.get("discovery_file_sha256", decision.get("discovery_sha256"))
        if frozen_hash != discovery_hash:
            parser.error("holdout decision does not match the discovery pairs")
    from drawer_env import DrawerEnv
    from policy import Policy

    out = args.output_dir / args.split
    out.mkdir(exist_ok=True, parents=True)
    seeds = [seed + args.seed_offset for seed in SEEDS[args.split]]
    old = Policy(args.config_dir, args.model_dir, None, config_name=args.config_name)
    new = Policy(args.config_dir, args.model_dir, args.ppo_checkpoint, config_name=args.config_name)
    envs = []
    try:
        for _ in range(args.batch):
            envs.append(DrawerEnv(cameras=True))
        for j in range(0,len(seeds),args.batch):
            group=seeds[j:j+args.batch]; file=out/f'batch_{group[0]}.json'
            probe_enabled = args.stall_probe
            if file.exists():
                if probe_enabled and not all((out/'stall'/f'paired_{s}.json').is_file() for s in group):
                    raise RuntimeError("Existing primary batch is missing its preregistered supplementary capture")
                continue
            current=envs[:len(group)]
            root_old, snapshots=roll(current,group,old,[0]*len(group),None,out,'old_root')
            probe = None
            if probe_enabled:
                from stall_probe import StallProbe
                probe = StallProbe(group, HORIZON)
            root_new,_=roll(current,group,new,[0]*len(group),None,out,'new_root',probe)
            assert all(r['initial_hash']==t['initial_hash'] for r,t in zip(root_old,root_new)), 'root identity mismatch'
            idx=[i for i,s in enumerate(snapshots) if s is not None]
            for i in idx: save(out/f'{group[i]}_handoff.json',snapshots[i])
            rows=[dict(seed=s, branch='old_first_open_drawer_handoff',signed_front_m=None,old_success=None,new_success=None) for s in group]
            if idx:
                selected=[current[i] for i in idx]; ids=[group[i] for i in idx]; times=[root_old[i]['handoff_step'] for i in idx]; states=[snapshots[i] for i in idx]
                left,_=roll(selected,ids,old,times,states,out,'old_branch')
                right,_=roll(selected,ids,new,times,states,out,'new_branch')
                for i,l,r in zip(idx,left,right):
                    assert l['start_metrics']==r['start_metrics'], 'mismatched branch metrics'
                    rows[i].update(signed_front_m=-float(root_old[i]['handoff_metrics']['bowl_depth_m']),
                                   handoff_step=root_old[i]['handoff_step'],old_success=l['success_once'],new_success=r['success_once'],
                                   old_first_success=l['first_success_step'],new_first_success=r['first_success_step'])
            if probe is not None:
                probe.finish(root_new, current, old, new, out, roll, save)
            save(file,dict(root_old=root_old,root_new=root_new,matched=rows))
            print('completed',args.split,group, 'paired handoffs',len(idx), flush=True)
        allrows=[r for s in seeds[::args.batch] for r in json.loads((out/f'batch_{s}.json').read_text())['matched']]
        assert len(allrows)==len(seeds)
        save(out/'pairs.json',allrows)
    finally:
        for env in envs: env.close()

if __name__=='__main__': main()
