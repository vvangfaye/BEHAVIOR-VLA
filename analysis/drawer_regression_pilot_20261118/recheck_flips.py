"""Post-selection reproducibility check using four fresh action-noise streams on the same scene."""
import argparse
import json
from pathlib import Path
from run_pairs import add_policy_arguments, roll, save

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Original paired run root")
    add_policy_arguments(parser)
    args = parser.parse_args()
    folder = args.output_dir / "holdout"
    out = args.output_dir / "repeatability"
    if not (folder / "pairs.json").is_file():
        parser.error("complete holdout needed")
    from drawer_env import DrawerEnv
    from policy import Policy

    out.mkdir(exist_ok=True)
    candidates={'root':[], 'handoff':[]}
    for file in sorted(folder.glob('batch_*.json')):
        b=json.loads(file.read_text())
        for a,n,h in zip(b['root_old'], b['root_new'], b['matched'],strict=True):
            if a['success_once'] != n['success_once']:
                candidates['root'].append((a['seed'],a['success_once'],n['success_once']))
            if h['old_success'] is not None and h['old_success'] != h['new_success']:
                candidates['handoff'].append((a['seed'],h['old_success'],h['new_success']))
    # Original flips only; stratify on root vs handoff, cap first four per category, without looking at repeat outcomes.
    candidates={k:v[:4] for k,v in candidates.items()}
    save(out/'selection.json',dict(cases=candidates,repeats=4,noise_protocol='Four new action-noise streams, paired across policies. Diagnostic on selected cases, not an estimate of prevalence.'))
    old = Policy(args.config_dir, args.model_dir, None, config_name=args.config_name)
    new = Policy(args.config_dir, args.model_dir, args.ppo_checkpoint, config_name=args.config_name)
    envs = []
    try:
        for _ in range(4):
            envs.append(DrawerEnv(cameras=True))
        for kind,items in candidates.items():
            for seed,oo,on in items:
                source=(folder/f'{seed}_handoff.json')
                if kind=='handoff':
                    snap=json.loads(source.read_text())
                else:
                    envs[0].reset(seed); snap=envs[0].snapshot()
                # roll uses seed IDs for ODE noise. Physical seed comes from the verified snapshot.
                ids=[seed+10_000_000*(i+1) for i in range(4)]
                starts=[snap['step']]*4
                prefix=[snap]*4
                left,_=roll(envs,ids,old,starts,prefix,out,f'{kind}_repeat_old')
                right,_=roll(envs,ids,new,starts,prefix,out,f'{kind}_repeat_new')
                outcomes=[(o['success_once'],n['success_once']) for o,n in zip(left,right)]
                assert all(o['initial_hash']==n['initial_hash']==snap['initial_hash'] and o['start_metrics']==n['start_metrics'] for o,n in zip(left,right))
                save(out/f'{kind}_{seed}.json',dict(seed=seed,kind=kind,at_step=snap['step'],input_state_sha256=snap['state_hash'],original=[oo,on],repeats=outcomes,
                         original_direction_repeated=sum(x==(oo,on) for x in outcomes),old_only=sum(o and not n for o,n in outcomes),
                         new_only=sum(n and not o for o,n in outcomes)))
                print('completed repeat',kind,seed,outcomes,flush=True)
    finally:
        for e in envs: e.close()


if __name__ == "__main__":
    main()
