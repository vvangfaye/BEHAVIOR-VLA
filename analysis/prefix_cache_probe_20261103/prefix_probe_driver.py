"""Prefix-cache probe driver (analysis/prefix_cache_probe_20261103): one GPU, measurement only.

Loads the learner model exactly as the multi-round learner does (fp32, JAX-matched primitives) and a few
of the 32 restored modem rollouts, then measures on the same rows:
  1. the current KL forward (default_forward, batch 8): time, GPU busy %, peak memory;
  2. the frozen-VLM prefix alone (_build_prefix_cache_for_actor): time, bytes per chunk, valid tokens;
  3. the same KL forward with the prefix served from a per-row cache: time, equality with (1);
  4. batch-4 KL with and without the cache (the batching difference that already exists today);
  5. the microbatch-4 gradient pass (backward_batched) with and without the cache, plus microbatch 1
     (the existing single-row path) as the reference for normal batching noise: time, memory, gradients;
  6. a host-pinned cache: host-to-device copy cost;
  7. one AdamW step: optimizer-state memory.
Nothing is trained, saved or uploaded except the result JSON.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import tarfile
import threading
import time
import traceback
import urllib.request

import torch

import kl_batch
import training as base
from common import torch_model, write
from experiment import seed_all
from parallel_training import Parallel, backward_batched, buffers


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def upload(url, data):
    request = urllib.request.Request(url, data=data, method='PUT')
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=90) as response:
        assert response.status in (200, 201, 204)


class Busy:
    """nvidia-smi once per second: utilization.gpu = % of time at least one kernel was running."""

    def __init__(self):
        self.samples, self._stop, self._thread = [], threading.Event(), None

    def __enter__(self):
        self.samples = []
        self._stop.clear()

        def loop():
            while not self._stop.is_set():
                try:
                    out = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                                          '--format=csv,noheader,nounits'], capture_output=True, text=True,
                                         timeout=5).stdout
                    self.samples.append([float(x) for x in out.strip().splitlines()[0].split(',')])
                except Exception:
                    pass
                self._stop.wait(1.0)
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=10)

    def summary(self):
        if not self.samples:
            return None
        busy = [s[0] for s in self.samples]
        return {'samples': len(busy), 'busy_percent_mean': statistics.fmean(busy),
                'busy_percent_median': statistics.median(busy), 'memory_used_mib_max': max(s[1] for s in self.samples)}


def split_rows(tree, count):
    """Nested tuple/list of batch-first tensors -> list of per-row structures (owned clones)."""
    if isinstance(tree, torch.Tensor):
        assert tree.shape[0] == count, (tuple(tree.shape), count)
        return [tree[i:i + 1].clone() for i in range(count)]
    if isinstance(tree, (tuple, list)):
        parts = [split_rows(x, count) for x in tree]
        return [type(tree)(p[i] for p in parts) for i in range(count)]
    if tree is None:
        return [None] * count
    raise TypeError(type(tree))


def join_rows(items, device):
    first = items[0]
    if isinstance(first, torch.Tensor):
        return torch.cat([x.to(device, non_blocking=True) for x in items], dim=0)
    if isinstance(first, (tuple, list)):
        return type(first)(join_rows([x[j] for x in items], device) for j in range(len(first)))
    if first is None:
        return None
    raise TypeError(type(first))


def nbytes(tree):
    if isinstance(tree, torch.Tensor):
        return tree.numel() * tree.element_size()
    if isinstance(tree, (tuple, list)):
        return sum(nbytes(x) for x in tree)
    return 0


def pinned(tree):
    if isinstance(tree, torch.Tensor):
        return tree.cpu().pin_memory()
    if isinstance(tree, (tuple, list)):
        return type(tree)(pinned(x) for x in tree)
    return tree


class PrefixCache:
    """Serve (prefix_out, prefix_mask, kv_cache) per row from memory instead of recomputing the frozen VLM.

    kl_batch.collate is wrapped to record which rows form the batch being collated (full_kl,
    backward_batched and replay checks all collate through it); the model's prefix builder then returns
    those rows' cached tensors, after checking the batch's prompt tokens match the cached rows.
    """

    def __init__(self, model):
        self.model, self.entries, self.tokens = model, {}, {}
        self.current, self.enabled, self.hits = None, False, 0
        self.original_prefix = model._build_prefix_cache_for_actor
        self.original_collate = kl_batch.collate
        cache = self

        def collate(rows, indices, device):
            cache.current = list(indices)
            return cache.original_collate(rows, indices, device)

        def prefix(observation):
            if not cache.enabled:
                return cache.original_prefix(observation)
            indices = cache.current
            assert indices is not None and all(i in cache.entries for i in indices), 'Row missing from the cache'
            tokens = observation.tokenized_prompt
            assert tokens.shape[0] == len(indices)
            expected = torch.cat([cache.tokens[i] for i in indices]).to(tokens.device)
            assert torch.equal(tokens, expected), 'Cached rows do not match this batch'
            cache.hits += len(indices)
            return join_rows([cache.entries[i] for i in indices], tokens.device)

        kl_batch.collate = collate
        model._build_prefix_cache_for_actor = prefix

    @torch.no_grad()
    def build(self, rows, indices, batch_size, device):
        for group in kl_batch.batches(rows, indices, batch_size):
            inputs, _ = self.original_collate(rows, group, device)
            observation = self.model._observation_from_forward_inputs(inputs)
            pieces = split_rows(self.original_prefix(observation), len(group))
            for j, index in enumerate(group):
                self.entries[index] = pieces[j]
                self.tokens[index] = observation.tokenized_prompt[j:j + 1].cpu()


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return value, time.perf_counter() - start


def kl_pass(model, rows, indices, batch_size, device):
    """The learner's KL forward (as kl_batch.full_kl): per-row logprobs and the sampler's old logprobs."""
    now, old = {}, {}
    with torch.no_grad():
        for group in kl_batch.batches(rows, indices, batch_size):
            inputs, previous = kl_batch.collate(rows, group, device)
            out = model.default_forward(inputs, compute_values=False)['logprobs'].float()
            for j, index in enumerate(group):
                now[index], old[index] = out[j:j + 1].cpu(), previous[j:j + 1].cpu()
    return now, old


def compare(a, b):
    """Max |difference| of per-row logprob tensors and of their summed log-ratios."""
    diff = max(float((a[i] - b[i]).abs().max()) for i in a)
    ratio = max(float((a[i].double().sum() - b[i].double().sum()).abs()) for i in a)
    return {'max_abs_logprob_difference': diff, 'max_abs_row_logratio_difference': ratio}


def ratio_error(now, old):
    """As replay_check_tolerant: max |p_now/p_old - 1| over rows, and the mean KL estimate."""
    worst, kl = 0.0, 0.0
    for i in now:
        logratio = (now[i] - old[i]).reshape(-1).sum().double()
        worst = max(worst, float(torch.expm1(logratio).abs()))
        kl += float(torch.expm1(logratio) - logratio)
    return {'max_probability_ratio_error': worst, 'mean_kl': kl / len(now)}


def gradients(model):
    return [p.grad.detach().double().cpu() for p in model.parameters() if p.requires_grad and p.grad is not None]


def agreement(a, b):
    assert len(a) == len(b)
    d2 = sum(float((x - y).square().sum()) for x, y in zip(a, b))
    r2 = sum(float(y.square().sum()) for y in b)
    n2 = sum(float(x.square().sum()) for x in a)
    dot = sum(float((x * y).sum()) for x, y in zip(a, b))
    return {'relative_l2': (d2 / max(r2, 1e-300)) ** .5, 'cosine': dot / max((n2 * r2) ** .5, 1e-300)}


def memory_snapshot():
    info = {}
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            key, _, value = line.partition(':')
            if key in ('MemTotal', 'MemAvailable'):
                info[key + '_gib'] = int(value.split()[0]) / 1024**2
    except OSError:
        pass
    for name in ('memory.max', 'memory.current', 'memory.peak'):
        path = Path('/sys/fs/cgroup') / name
        if path.is_file():
            info['cgroup_' + name] = path.read_text().strip()
    info['cpu_count'] = len(os.sched_getaffinity(0))
    return info


def main():
    plan = json.loads(Path(__file__).with_name('plan.json').read_text())
    probe = plan['probe']
    root = Path(plan['output'])
    root.mkdir(parents=True, exist_ok=True)
    attempt = int(os.environ['B1K_ATTEMPT'])
    torch.cuda.set_device(0)
    device = torch.device('cuda', 0)
    parallel = Parallel(device)
    state = {'state': 'running', 'phase': 'prefix_probe', 'attempt': attempt, 'started_at': time.time()}
    result = {'plan': {k: probe[k] for k in sorted(probe)}, 'started_at': state['started_at']}

    def publish(**fields):
        state.update(fields, updated_at=time.time())
        write(root / 'status.json', state)
        try:
            upload(os.environ['B1K_STATUS_PUT'], json.dumps(state, allow_nan=False).encode())
        except Exception as error:
            print('Status upload:', type(error).__name__, flush=True)
        print(json.dumps(state, allow_nan=False), flush=True)

    def interrupted(*_):
        raise KeyboardInterrupt('Interrupted (Spot reclaim or time limit)')

    signal.signal(signal.SIGTERM, interrupted)
    try:
        publish(stage='loading_model')
        seed_all(plan['model_seed'])
        model, load_seconds = timed(lambda: torch_model(plan))
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert base.hash_parameters((n, p) for n, p in model.named_parameters() if p.requires_grad) \
            == plan['source_identity']['cold_trainable_hash']
        frozen = base.hash_parameters((n, p) for n, p in model.named_parameters() if not p.requires_grad)
        assert frozen == plan['source_identity']['frozen_hash']
        result['hardware'] = {'device': torch.cuda.get_device_name(0),
                              'total_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
                              'model_allocated_bytes': torch.cuda.memory_allocated(device),
                              'trainable_parameters': sum(p.numel() for p in trainable),
                              'all_parameters': sum(p.numel() for p in model.parameters()),
                              'parameter_dtypes': sorted({str(p.dtype) for p in model.parameters()}),
                              'float32_matmul_precision': torch.get_float32_matmul_precision(),
                              'allow_tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
                              'model_load_seconds': load_seconds, 'host': memory_snapshot()}
        write(root / 'prefix_probe.json', result)

        publish(stage='loading_rollouts')
        episodes = []
        for number in probe['episodes']:
            item = plan['source_rollout_receipts'][number]
            path = Path(item['path'])
            assert path.is_relative_to(plan['source_rollout_root']) and sha(path) == item['sha256']
            episode = torch.load(path, map_location='cpu', weights_only=True)
            assert episode['instance'] == item['instance'] and episode['seed'] == item['seed']
            episodes.append(episode)
        rows, data = buffers(episodes, plan)
        count = min(probe['rows'], len(rows))
        indices = sorted(set(torch.linspace(0, len(rows) - 1, count).round().long().tolist()))
        train = indices[::max(1, len(indices) // probe['train_rows'])][:probe['train_rows']]
        host = indices[:probe['host_rows']]
        signatures = {repr(kl_batch.signature(rows[i])) for i in indices}
        result['rows'] = {'loaded': len(rows), 'measured': len(indices), 'train': len(train), 'host': len(host),
                          'input_signatures': len(signatures)}
        cache = PrefixCache(model)
        batch = probe['kl_batch_size']

        # Warm-up (allocator, kernels); excluded from all timings.
        kl_pass(model, rows, indices[:batch], batch, device)

        publish(stage='baseline_kl')
        torch.cuda.reset_peak_memory_stats(device)
        with Busy() as busy:
            (base8, old), seconds = timed(lambda: kl_pass(model, rows, indices, batch, device))
        timing = {'baseline_kl_batch8_seconds_per_chunk': seconds / len(indices)}
        memory = {'peak_baseline_kl_batch8_bytes': torch.cuda.max_memory_allocated(device)}
        utilization = {'baseline_kl_batch8': busy.summary()}

        publish(stage='building_prefix_cache')
        torch.cuda.reset_peak_memory_stats(device)
        _, seconds = timed(lambda: cache.build(rows, indices, batch, device))
        timing['prefix_only_batch8_seconds_per_chunk'] = seconds / len(indices)
        sample = cache.entries[indices[0]]
        prefix_out, prefix_mask, kv = sample
        valid = [int(cache.entries[i][1].sum()) for i in indices]
        memory.update(cache_bytes_per_chunk={'kv_cache': nbytes(kv), 'prefix_out': nbytes(prefix_out),
                                             'prefix_mask': nbytes(prefix_mask)},
                      cache_resident_bytes=sum(nbytes(cache.entries[i]) for i in indices),
                      peak_prefix_build_bytes=torch.cuda.max_memory_allocated(device))
        result['prefix'] = {'length': int(prefix_mask.shape[1]), 'kv_layers': len(kv),
                            'kv_shape': [list(t.shape) for t in kv[0]], 'kv_dtype': str(kv[0][0].dtype),
                            'valid_tokens_min': min(valid), 'valid_tokens_median': statistics.median(valid),
                            'valid_tokens_max': max(valid)}

        publish(stage='cached_kl')
        cache.enabled = True
        kl_pass(model, rows, indices[:batch], batch, device)
        torch.cuda.reset_peak_memory_stats(device)
        with Busy() as busy:
            (cached8, _), seconds = timed(lambda: kl_pass(model, rows, indices, batch, device))
        timing['cached_kl_batch8_seconds_per_chunk'] = seconds / len(indices)
        memory['peak_cached_kl_batch8_bytes'] = torch.cuda.max_memory_allocated(device)
        utilization['cached_kl_batch8'] = busy.summary()
        exact = {'cached_vs_baseline_same_batching': compare(cached8, base8),
                 'ratio_error_vs_sampler': {'baseline_batch8': ratio_error(base8, old),
                                            'cached_batch8': ratio_error(cached8, old)}}

        publish(stage='batch4_kl')
        cache.enabled = False
        (base4, _), seconds = timed(lambda: kl_pass(model, rows, indices, 4, device))
        timing['baseline_kl_batch4_seconds_per_chunk'] = seconds / len(indices)
        cache.enabled = True
        (cached4, _), seconds = timed(lambda: kl_pass(model, rows, indices, 4, device))
        timing['cached_kl_batch4_seconds_per_chunk'] = seconds / len(indices)
        exact['baseline_batch4_vs_batch8_existing'] = compare(base4, base8)
        exact['cached_batch4_vs_baseline_batch4'] = compare(cached4, base4)
        exact['ratio_error_vs_sampler']['cached_batch4'] = ratio_error(cached4, old)
        result.update(timing=timing, memory=memory, exactness=exact, utilization=utilization)
        write(root / 'prefix_probe.json', result)

        publish(stage='gradient_passes')
        grads = {}
        for label, enabled, microbatch in [('baseline_mb4', False, probe['microbatch']),
                                           ('cached_mb4', True, probe['microbatch']), ('baseline_mb1', False, 1)]:
            cache.enabled = enabled
            model.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(device)
            with Busy() as busy:
                loss, seconds = timed(lambda: backward_batched(model, rows, data, train, plan, parallel, microbatch,
                                                               None, distributed=False))
            timing[f'{label}_gradient_seconds_per_chunk'] = seconds / len(train)
            memory[f'peak_{label}_gradient_bytes'] = torch.cuda.max_memory_allocated(device)
            utilization[f'{label}_gradient'] = busy.summary()
            grads[label] = gradients(model)
            result.setdefault('losses', {})[label] = loss
        exact['gradient'] = {'cached_mb4_vs_baseline_mb4': agreement(grads['cached_mb4'], grads['baseline_mb4']),
                             'baseline_mb4_vs_baseline_mb1_existing': agreement(grads['baseline_mb4'],
                                                                                grads['baseline_mb1']),
                             'cached_mb4_vs_baseline_mb1': agreement(grads['cached_mb4'], grads['baseline_mb1'])}
        del grads
        write(root / 'prefix_probe.json', result)

        publish(stage='host_pinned_cache')
        cache.enabled = True
        _, seconds = timed(lambda: kl_pass(model, rows, host, batch, device))
        timing['cached_kl_batch8_gpu_resident_subset_seconds_per_chunk'] = seconds / len(host)
        for i in host:
            cache.entries[i] = pinned(cache.entries[i])
        torch.cuda.empty_cache()
        group = host[:batch]
        _, seconds = timed(lambda: join_rows([cache.entries[i] for i in group], device))
        timing['host_to_device_seconds_per_chunk'] = seconds / len(group)
        kl_pass(model, rows, host[:batch], batch, device)
        _, seconds = timed(lambda: kl_pass(model, rows, host, batch, device))
        timing['cached_kl_batch8_host_pinned_seconds_per_chunk'] = seconds / len(host)
        result['cache_hits'] = cache.hits

        publish(stage='optimizer_step')
        cache.enabled = False
        optimizer = base.build_optimizer(model, plan)
        model.zero_grad(set_to_none=True)
        backward_batched(model, rows, data, train[:8], plan, parallel, probe['microbatch'], None, distributed=False)
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.step()
        torch.cuda.synchronize()
        memory['after_optimizer_step_allocated_bytes'] = torch.cuda.memory_allocated(device)
        memory['peak_optimizer_step_bytes'] = torch.cuda.max_memory_allocated(device)
        memory['max_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
        result['hardware']['host_end'] = memory_snapshot()
        result['finished_at'] = time.time()
        write(root / 'prefix_probe.json', result)
        assert base.hash_parameters((n, p) for n, p in model.named_parameters() if not p.requires_grad) == frozen
        publish(state='complete', stage='probe_complete')
    except BaseException as error:
        write(root / 'failure.json', {'type': type(error).__name__, 'traceback': traceback.format_exc()})
        publish(state='failed', stage='failed', error_type=type(error).__name__, error=str(error)[:400])
        raise
    finally:
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode='w:gz') as archive:
            for path in sorted(root.rglob('*.json')):
                try:
                    body = path.read_bytes()
                    json.loads(body)
                except (OSError, ValueError):
                    continue
                item = tarfile.TarInfo(str(path.relative_to(root)))
                item.size = len(body)
                archive.addfile(item, io.BytesIO(body))
        try:
            upload(os.environ['B1K_RESULT_PUT'], stream.getvalue())
        except Exception as error:
            print('Result upload:', type(error).__name__, flush=True)


if __name__ == '__main__':
    main()
