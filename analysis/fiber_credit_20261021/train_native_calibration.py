"""Fixed, episode-disjoint calibration pilot on existing native VLA inputs.

Four bounded CPU workers. The actor stays frozen; only a shared contrast
amplitude and angle are trained. Held-out data are opened for scoring only
after the fixed training schedule. This is not a closed-loop learning run.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import math
import multiprocessing as mp
import queue
import signal
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def write_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj, indent=2)+'\n')
    temp.replace(path)


def make_plan():
    import numpy as np
    data = ROOT/'analysis/action_support_20261021/vla_pilot'
    description = json.loads((data/'data_plan.json').read_text())
    tasks = sorted({s['task_name'] for s in description['samples']})
    selection = {}
    for split, folds in [('train', {2, 3, 4}), ('test', {0})]:
        chosen = []
        for task in tasks:
            episodes = sorted({s['episode_index'] for s in description['samples']
                               if s['task_name'] == task and s['fold'] in folds})
            if len(episodes) < 2:
                raise ValueError('Need two distinct episodes per task and split')
            for episode in episodes[:2]:
                candidates = [(i, s) for i, s in enumerate(description['samples'])
                              if s['episode_index'] == episode and s['task_name'] == task]
                index, sample = min(candidates, key=lambda v:abs(v[1]['frame_index']-v[1]['episode_length']/2))
                for draw in range(2):
                    chosen.append(dict(id=f'{split}-{len(chosen):02d}', index=index, task=task,
                                       episode=episode, frame=sample['frame_index'], draw=draw,
                                       seed=49100+1000*(split=='test')+len(chosen)))
        selection[split] = chosen
    assert not ({s['episode'] for s in selection['train']} & {s['episode'] for s in selection['test']})
    files = [Path(__file__).resolve(), HERE/'learned_contrast.py', HERE/'null_transport.py',
             HERE/'check_native_meta.py', ROOT/'analysis/native_score_implicit_20261021/implicit_geometry.py',
             ROOT/'analysis/relative_flow_20261021/adapter.py', ROOT/'analysis/action_support_20261021/vla_runtime.py',
             ROOT/'analysis/action_support_20261021/vla_data.py']
    return dict(schema=1, actor_checkpoint='1688977', actor_frozen=True, new_demonstrations=0,
                simulation=False, cloud=False, workers=4, threads_per_worker=4, max_seconds=7200,
                expected_peak_total_rss_gib=200, precision='FP32 actor / FP64 geometry',
                selection=selection, calibration_parameters=['scaled_amplitude', 'pair_rotation_angle'],
                initial_phi=[0., math.pi/4], epochs=2, batch_size=4, adam_lr=.05,
                amplitude_limits=[-2., 2.], angle_limits=[.05, 1.52], shuffle_seed=8192,
                control_scale='sqrt(sum train HH / sum train CC) at initial angle; computed on training split only',
                gradient_normalizer='mean training HH; fixed for every calibration update',
                selection_rule='First two episode IDs per task/split; anchor nearest episode midpoint; two fixed noise draws',
                data_plan_sha256=hashlib.sha256((data/'data_plan.json').read_bytes()).hexdigest(),
                source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                evaluation=dict(metric='paired reward-free gradient second moment ratio; theoretical mean is zero',
                                bootstrap='10000 paired episode-cluster resamples, fixed seed 719',
                                advance_if_point_ratio_at_most=.70, advance_if_bootstrap_95_upper_below=1.,
                                interpretation='Pilot gate only; does not establish return-weighted variance or task-success gain'))


def worker(number, jobs, events, plan, directory):
    import sys
    import resource
    import torch
    from torch.func import functional_call, jvp
    sys.path[:0] = [str(HERE), str(ROOT/'analysis/native_score_implicit_20261021')]
    from check_native_meta import NativeFunction, NativeAdapter, VlaPilotData, load_base
    from implicit_geometry import geometry, local_surrogate
    from null_transport import reference_parts, null_parts
    from learned_contrast import contrast, directional_variance_objective
    torch.set_num_threads(plan['threads_per_worker'])
    torch.manual_seed(20261021)
    directory = Path(directory)
    try:
        config = json.loads((ROOT/'analysis/action_support_20261021/local_vla_pilot_ada01/plan.json').read_text())
        base = load_base(config['checkpoint'], config['tokenizer'], device='cpu', precision='fp32')
        adapter = NativeAdapter(base).eval()
        data = VlaPilotData(config['data_directory'], base)
        front = torch.tensor([t*32+a for t in range(16) for a in range(23)])
        tail = front+512
        events.put(dict(kind='ready', worker=number))
        while True:
            job = jobs.get()
            if job is None:
                break
            start = time.monotonic()
            sample = job['sample']
            sample_id = sample['id']
            obs, _, _ = data.batch([sample['index']])
            assert int(data.episode[sample['index']]) == sample['episode']
            assert int(data.frame[sample['index']]) == sample['frame']
            sampler = NativeFunction(adapter, adapter.prepare(obs))
            trainable = [(n, p) for n, p in sampler.named_parameters() if p.requires_grad]
            names, parameters = zip(*trainable)
            assert sum(p.numel() for p in parameters) == 430098464
            cache_path = directory/'geometry'/f'{sample_id}.pt'
            if job['action'] == 'prepare':
                for p in parameters:
                    p.requires_grad_(False)
                rng = torch.Generator().manual_seed(sample['seed'])
                z = torch.randn(1024, generator=rng)
                e = (torch.randint(0, 2, (1024,), generator=rng)*2-1).double()
                zgrad = z.clone().requires_grad_(True)
                output = sampler(zgrad)
                rows = []
                for first in range(0, 368, 8):
                    last = min(first+8, 368)
                    rows.append(torch.autograd.grad(output[first:last], zgrad,
                                grad_outputs=torch.eye(last-first), is_grads_batched=True,
                                retain_graph=True, create_graph=False)[0].detach().double())
                    if last % 128 == 0 or last == 368:
                        events.put(dict(kind='progress', worker=number, sample=sample_id,
                                        stage='jacobian', rows=last))
                j = torch.cat(rows)
                del rows, output, zgrad
                gc.collect()
                b, m, singular = geometry(j)
                inverse_error = float((j@b-torch.eye(368, dtype=j.dtype)).abs().max())
                assert inverse_error < 1e-5
                parts = reference_parts(sampler, z, j, b, m, e)
                l = torch.zeros(1024, 368, dtype=j.dtype)
                l[tail, torch.arange(368)] = 1
                c, dc = null_parts(parts, l)
                projection_error = float((j@c).abs().max())
                assert projection_error < 1e-5
                assert torch.isfinite(dc).all() and torch.isfinite(parts['div_b']).all()
                cached = dict(z=z, e=e, b=b, div_b=parts['div_b'], c=c, dc=dc,
                              sample=sample, inverse_error=inverse_error,
                              projection_error=projection_error, condition_number=float(singular[0]/singular[-1]))
                torch.save(cached, cache_path)
                del j, b, m, c, dc, parts, l
                gc.collect()
                for p in parameters:
                    p.requires_grad_(True)
            else:
                cached = torch.load(cache_path, map_location='cpu', weights_only=True)
                assert cached['sample'] == sample
            z, e, b, db, c, dc = [cached[k] for k in ('z', 'e', 'b', 'div_b', 'c', 'dc')]

            def control(function, angle):
                feature = lambda noise:contrast(function, noise, front, tail,
                                                angle.to(noise.dtype).expand(len(front)))
                value, tangent = jvp(feature, (z,), (e.to(z.dtype),))
                return ((c.T@z.double()-dc)*value.double()).sum()-((c.T@e)*tangent.double()).sum()

            if job['action'] in ('stats', 'prepare'):
                # Test geometry is prepared early, but no held-out gradient
                # statistic is returned before the calibration is frozen.
                if job['action'] == 'prepare' and sample_id.startswith('test-'):
                    result = dict(geometry_only=True)
                else:
                    angle = torch.tensor(job.get('angle', math.pi/4), dtype=torch.float64)
                    h = torch.autograd.grad(local_surrogate(sampler, z, b, db, e), parameters, allow_unused=True)
                    cgrad = torch.autograd.grad(control(sampler, angle), parameters, allow_unused=True)
                    hh = hc = cc = 0.
                    for p, h1, c1 in zip(parameters, h, cgrad):
                        if h1 is not None:
                            assert torch.isfinite(h1).all()
                            hh += float(h1.double().square().sum())
                        if c1 is not None:
                            assert torch.isfinite(c1).all()
                            cc += float(c1.double().square().sum())
                        if h1 is not None and c1 is not None:
                            hc += float((h1.double()*c1.double()).sum())
                    result = dict(HH=hh, HC=hc, CC=cc)
                    del h, cgrad
            elif job['action'] == 'train':
                phi = torch.nn.Parameter(torch.tensor(job['phi'], dtype=torch.float64))
                scale = job['control_scale']
                total = local_surrogate(sampler, z, b, db, e)+scale*phi[0]*control(sampler, phi[1])
                gradients = torch.autograd.grad(total, parameters, allow_unused=True)
                g = tuple(torch.zeros_like(p) if value is None else value.detach()
                          for p, value in zip(parameters, gradients))
                assert all(torch.isfinite(value).all() for value in g)
                energy = sum(float(value.double().square().sum()) for value in g)
                del total, gradients
                gc.collect()
                def control_only(values):
                    function = lambda noise:functional_call(sampler, dict(zip(names, values)), (noise,))
                    return scale*phi[0]*control(function, phi[1])
                meta = directional_variance_objective(control_only, parameters, g)/job['normalizer']
                dg = torch.autograd.grad(meta, phi)[0]
                assert torch.isfinite(dg).all()
                result = dict(energy=energy, meta_gradient=dg.tolist())
                del g, dg, meta
            else:
                raise ValueError(job['action'])
            result.update(sample=sample_id, action=job['action'], seconds=time.monotonic()-start,
                          peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
            write_json(directory/'jobs'/f'{job["id"]}.json', result)
            events.put(dict(kind='result', id=job['id'], result=result))
            del sampler, cached, obs
            gc.collect()
    except BaseException:
        events.put(dict(kind='error', worker=number, error=traceback.format_exc()))
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--plan-only', action='store_true')
    args = ap.parse_args()
    if args.plan_only:
        print(json.dumps(make_plan(), indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    for sub in ('geometry', 'jobs'):
        (args.output/sub).mkdir()
    plan = make_plan()
    write_json(args.output/'plan.json', plan)
    start = time.monotonic()
    state = dict(complete=False, stage='starting', started_unix=time.time(), actor_frozen=True)
    def record(stage, **kwargs):
        state.update(stage=stage, elapsed_seconds=time.monotonic()-start, **kwargs)
        write_json(args.output/'state.json', state)
        print(json.dumps(dict(stage=stage, **kwargs)), flush=True)
    def stopped(signum, frame):
        raise RuntimeError(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, stopped)
    ctx = mp.get_context('spawn')
    jobs, events = ctx.Queue(), ctx.Queue()
    children = [ctx.Process(target=worker, args=(i, jobs, events, plan, str(args.output)))
                for i in range(plan['workers'])]
    def event():
        while True:
            if time.monotonic()-start > plan['max_seconds']:
                raise TimeoutError('Fixed calibration runtime limit exceeded')
            try:
                answer = events.get(timeout=5)
            except queue.Empty:
                if any(not p.is_alive() for p in children):
                    raise RuntimeError('Calibration worker exited unexpectedly')
                continue
            if answer['kind'] == 'error':
                raise RuntimeError(answer['error'])
            if answer['kind'] == 'progress':
                print(json.dumps(answer), flush=True)
                continue
            return answer
    def batch(tasks):
        for task in tasks:
            jobs.put(task)
        found = {}
        while len(found) < len(tasks):
            answer = event()
            assert answer['kind'] == 'result'
            found[answer['id']] = answer['result']
            record(state['stage'], completed=len(found), total=len(tasks), last_id=answer['id'])
        return [found[t['id']] for t in tasks]
    try:
        for p in children:
            p.start()
        ready = 0
        while ready < len(children):
            assert event()['kind'] == 'ready'
            ready += 1
        record('preparing_geometry')
        prepare_jobs = [dict(id='prepare-'+s['id'], action='prepare', sample=s)
                        for split in ('train', 'test') for s in plan['selection'][split]]
        preparation = batch(prepare_jobs)
        train_stats = preparation[:len(plan['selection']['train'])]
        mean_hh = sum(s['HH'] for s in train_stats)/len(train_stats)
        mean_cc = sum(s['CC'] for s in train_stats)/len(train_stats)
        assert mean_hh > 0 and mean_cc > 0
        scale = math.sqrt(mean_hh/mean_cc)
        record('training', control_scale=scale, normalizer=mean_hh)
        import numpy as np
        import torch
        phi = torch.nn.Parameter(torch.tensor(plan['initial_phi'], dtype=torch.float64))
        optimizer = torch.optim.Adam([phi], lr=plan['adam_lr'])
        rng = np.random.default_rng(plan['shuffle_seed'])
        updates = []
        train = plan['selection']['train']
        for epoch in range(plan['epochs']):
            order = rng.permutation(len(train)).tolist()
            for first in range(0, len(order), plan['batch_size']):
                step = len(updates)
                pending = [dict(id=f'train-step{step:02d}-'+train[i]['id'], action='train', sample=train[i],
                                phi=phi.detach().tolist(), control_scale=scale, normalizer=mean_hh)
                           for i in order[first:first+plan['batch_size']]]
                answers = batch(pending)
                gradient = torch.tensor([r['meta_gradient'] for r in answers]).mean(0)
                optimizer.zero_grad()
                phi.grad = gradient.to(phi.dtype)
                optimizer.step()
                with torch.no_grad():
                    phi[0].clamp_(*plan['amplitude_limits'])
                    phi[1].clamp_(*plan['angle_limits'])
                updates.append(dict(step=step, epoch=epoch, phi=phi.detach().tolist(),
                                    gradient=gradient.tolist(), mean_energy=sum(r['energy'] for r in answers)/len(answers)))
                write_json(args.output/'training.json', updates)
                record('training', updates=len(updates), phi=phi.detach().tolist())
        frozen = dict(phi=phi.detach().tolist(), control_scale=scale, normalizer=mean_hh,
                      training_updates=len(updates), fit_split='train only')
        write_json(args.output/'frozen_calibrator.json', frozen)
        record('heldout_scoring', frozen_calibrator=frozen)
        test_jobs = [dict(id='test-'+s['id'], action='stats', sample=s, angle=frozen['phi'][1])
                     for s in plan['selection']['test']]
        tests = batch(test_jobs)
        amplitude = frozen['phi'][0]*scale
        grouped = {}
        for sample, result in zip(plan['selection']['test'], tests):
            hh, hc, cc = [result[k] for k in ('HH', 'HC', 'CC')]
            corrected = hh+2*amplitude*hc+amplitude**2*cc
            assert corrected >= -1e-8*max(hh, 1)
            result.update(corrected_second_moment=corrected, ratio=corrected/hh, episode=sample['episode'])
            group = grouped.setdefault(sample['episode'], [0., 0.])
            group[0] += hh
            group[1] += corrected
        group = np.array(list(grouped.values()))
        ratio = float(group[:, 1].sum()/group[:, 0].sum())
        boot_rng = np.random.default_rng(719)
        ids = boot_rng.integers(0, len(group), size=(10000, len(group)))
        bootstrap = group[ids].sum(1)
        interval = np.quantile(bootstrap[:, 1]/bootstrap[:, 0], [.025, .975]).tolist()
        passed = ratio <= plan['evaluation']['advance_if_point_ratio_at_most'] and interval[1] < 1.
        result = dict(complete=True, stage='complete', reward_free_second_moment_ratio=ratio,
                      episode_bootstrap_95_interval=interval, advance_to_actor_learning_pilot=bool(passed),
                      heldout_results=tests, frozen_calibrator=frozen,
                      scope='Twelve held-out noise draws from six existing episodes. Actor frozen. Does not establish return-weighted variance or closed-loop success.',
                      elapsed_seconds=time.monotonic()-start)
        write_json(args.output/'result.json', result)
        record('complete', complete=True, ratio=ratio, bootstrap_95_interval=interval,
               advance_to_actor_learning_pilot=bool(passed))
    except BaseException as exc:
        record('failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
        for child in children:
            child.join(timeout=10)


if __name__ == '__main__':
    main()
