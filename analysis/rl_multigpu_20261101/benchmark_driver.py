"""Read-only KL batch equivalence and throughput on the saved PPO models."""
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

import torch
import torch.distributed as dist
import training as base
from common import torch_model, write
from experiment import seed_all
from parallel_training import initialize
from kl_batch import full_kl, collate


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    plan = json.loads(Path(__file__).with_name('plan.json').read_text())
    p = initialize()
    root = Path(plan['output'])
    status = {'validation_only': True, 'optimizer_steps': 0}

    def publish(**fields):
        if p.rank:
            return
        status.update(fields, updated_at=time.time())
        write(root/'status.json', status)
        request = urllib.request.Request(os.environ['B1K_STATUS_PUT'],
            data=json.dumps(status).encode(), method='PUT')
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=60):
            pass
        print(json.dumps(status), flush=True)

    publish(stage='loading_model')
    seed_all(plan['model_seed'])
    model = torch_model(plan)
    cold = base.hash_parameters((n, v) for n, v in model.named_parameters() if v.requires_grad)
    assert cold == plan['source_identity']['cold_trainable_hash']
    frozen = base.hash_parameters((n, v) for n, v in model.named_parameters() if not v.requires_grad)
    rows = []
    for number, item in enumerate(plan['source_rollout_receipts'], 1):
        assert sha(item['path']) == item['sha256']
        episode = torch.load(item['path'], map_location='cpu', weights_only=True)
        rows.extend(episode['transitions'])
        publish(stage='loading_rollouts', episodes=number)
    assert len(rows) == 23908
    shapes = {k: list(v.shape) for k, v in rows[0]['forward_inputs'].items()}
    results = {'validation_only': True, 'optimizer_steps': 0, 'shapes': shapes,
               'runtime_signature': plan['distributed']['runtime_signature'],
               'tolerances': plan['kl_benchmark']['tolerances'], 'variants': {},
               'hardware': [torch.cuda.get_device_name(p.rank)], 'world_size': p.world}
    for variant, source in plan['kl_benchmark']['sources'].items():
        directory = Path(source['output'])/source['metadata']['relative_path']
        metadata = json.loads((directory/'resume.json').read_text())
        assert metadata == source['metadata']
        base.load_overlay(model, directory, metadata['trainable_sha256'])
        source_files = {name: sha(directory/name) for name in ['trainable.pt', 'optimizer.pt', 'buffers.pt', 'resume.json']} if p.rank == 0 else None
        subset_indices = torch.linspace(0, len(rows)-1, 257).round().long().unique().tolist()
        selected = rows if variant == 'control' else [rows[i] for i in subset_indices]
        report = {'samples': len(selected), 'accepted_updates': metadata['state']['accepted'],
                  'trainable_hash': metadata['trainable_sha256'], 'source_files': source_files, 'batches': {}}
        reference = {}
        for size in [1, 4, 8]:
            publish(stage='batch_warmup', variant=variant, batch_size=size, samples=len(selected))
            failed = 0
            try:
                with torch.no_grad():
                    inputs, _ = collate(selected, list(range(size)), p.device)
                    model.default_forward(inputs, compute_values=False)
                    del inputs
                    torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError:
                failed = 1
            failed = p.scalar(failed, dist.ReduceOp.MAX)
            torch.cuda.empty_cache()
            if failed:
                report['batches'][str(size)] = {'passed': False, 'error': 'CUDA out of memory'}
                continue
            actual = {}
            def observe(indices, now, ratio, value):
                logp, ratios, values = now.cpu(), ratio.cpu(), value.cpu()
                for offset, index in enumerate(indices):
                    actual[index] = (logp[offset].clone(), ratios[offset].clone(), values[offset].clone())
            p.barrier()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            divergence = full_kl(model, selected, p, publish, size, observe)
            torch.cuda.synchronize()
            seconds = p.scalar(time.monotonic()-started, dist.ReduceOp.MAX)
            peak = p.scalar(torch.cuda.max_memory_allocated(), dist.ReduceOp.MAX)
            measurement = {'seconds': seconds, 'peak_allocated_bytes': int(peak), 'kl': divergence}
            if size == 1:
                reference = actual
                measurement.update(passed=True, max_logprob_error=0., max_logratio_error=0.)
            else:
                tol = plan['kl_benchmark']['tolerances']
                assert actual.keys() == reference.keys()
                logp_error = ratio_error = 0.
                close = True
                for index, (logp, ratio, _) in actual.items():
                    ref_logp, ref_ratio, _ = reference[index]
                    logp_error = max(logp_error, float((logp-ref_logp).abs().max()))
                    ratio_error = max(ratio_error, float((ratio-ref_ratio).abs()))
                    close &= torch.allclose(logp, ref_logp, atol=tol['logprob_atol'], rtol=tol['logprob_rtol'])
                close = p.scalar(int(close), dist.ReduceOp.MIN) == 1
                logp_error = p.scalar(logp_error, dist.ReduceOp.MAX)
                ratio_error = p.scalar(ratio_error, dist.ReduceOp.MAX)
                ref_kl = report['batches']['1']['kl']['mean']
                kl_error = abs(divergence['mean']-ref_kl)
                passed = (close and ratio_error <= tol['logratio_atol'] and
                          kl_error <= tol['kl_atol']+tol['kl_rtol']*abs(ref_kl) and
                          (divergence['mean'] <= .01) == (ref_kl <= .01))
                measurement.update(passed=passed, max_logprob_error=logp_error,
                                   max_logratio_error=ratio_error, mean_kl_error=kl_error,
                                   speedup=report['batches']['1']['seconds']/seconds)
            report['batches'][str(size)] = measurement
            results['variants'][variant] = report
            publish(stage='batch_complete', variant=variant, batch_size=size, measurement=measurement)
            if p.rank == 0:
                write(root/'benchmark.json', results)
            del actual
            gc.collect()
        # No optimizer exists; verify model and all saved training state stayed intact.
        assert base.hash_parameters((n,v) for n,v in model.named_parameters() if v.requires_grad) == metadata['trainable_sha256']
        if p.rank == 0:
            assert source_files == {name: sha(directory/name) for name in source_files}
        del reference
        gc.collect()
    assert base.hash_parameters((n,v) for n,v in model.named_parameters() if not v.requires_grad) == frozen
    valid = [size for size in [4,8] if all(r['batches'][str(size)]['passed'] for r in results['variants'].values())]
    results['passed'] = bool(valid)
    results['selected_batch_size'] = min(valid, key=lambda s: results['variants']['control']['batches'][str(s)]['seconds']) if valid else None
    results['training_state_unchanged'] = True
    if p.rank == 0:
        write(root/'benchmark.json', results)
        publish(state='complete', stage='benchmark_complete', result=results)
    p.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
