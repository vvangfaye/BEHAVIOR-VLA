"""Bounded simulator initialization comparison; no controls or model updates."""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import tarfile
import threading
import time
import traceback
import urllib.request

import torch

from common import ExecutionStopped, Simulator, torch_model, write
from experiment import seed_all
from training import load_overlay, hash_parameters


def memory_state():
    result = {}
    for name in ('memory.current', 'memory.peak', 'memory.max', 'memory.events'):
        p = Path('/sys/fs/cgroup') / name
        if p.exists():
            result[name] = p.read_text().strip()
    return result


def process_state(process):
    status = {'returncode': process.poll()}
    p = Path('/proc') / str(process.pid)
    try:
        status['status'] = '\n'.join(x for x in (p/'status').read_text().splitlines()
                                    if x.split(':')[0] in ('State', 'VmRSS', 'VmPeak', 'Threads'))
        status['fd_count'] = len(list((p/'fd').iterdir()))
    except FileNotFoundError:
        status['gone'] = True
    return status


def put(url, data):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(urllib.request.Request(url, data=data, method='PUT'), timeout=180):
        pass


def main():
    here = Path(__file__).resolve().parent
    plan = json.loads((here/'plan.json').read_text())
    root = Path(plan['output']); root.mkdir(parents=True, exist_ok=False)
    write(root/'plan.json', plan)
    state = {'state': 'running', 'stage': 'loading', 'started_at': time.time()}
    def publish(**fields):
        state.update(fields, updated_at=time.time())
        write(root/'status.json', state)
        try: put(os.environ['B1K_STATUS_PUT'], (root/'status.json').read_bytes())
        except Exception: pass
        print(json.dumps(state), flush=True)
    def deadline(*args):
        raise ExecutionStopped('Diagnostic deadline reached')
    signal.signal(signal.SIGALRM, deadline); signal.alarm(plan['compute_timeout_seconds'])
    rows = []
    try:
        seed_all(0)
        model = torch_model(plan)
        load_overlay(model, plan['diagnostic']['existing_round1_checkpoint'],
                     plan['diagnostic']['expected_round1_hash'])
        before = hash_parameters((n,p) for n,p in model.named_parameters() if p.requires_grad)
        write(root/'python.json', {'version': sys.version, 'executable': sys.executable,
                                  'model_hash': before, 'memory': memory_state()})
        for trial in plan['diagnostic']['trials']:
            name = trial['name']; trial_root = root/name
            os.environ['B1K_INIT_DUMP_ENABLED'] = '1' if trial['enabled'] else '0'
            os.environ['B1K_INIT_DUMP_INTERVAL'] = str(trial['interval'])
            row = {'name':name, 'started_at':time.time(), 'before':memory_state(), 'no_controls':True}
            original_abort = Simulator.abort
            def abort_with_evidence(sim):
                evidence = {'at_eof': process_state(sim.process), 'memory': memory_state()}
                try: sim.process.wait(timeout=2)
                except Exception: pass
                evidence['after_two_seconds'] = process_state(sim.process)
                write(sim.output/'pre_abort_evidence.json', evidence)
                row['pre_abort_evidence'] = evidence
                original_abort(sim)
            Simulator.abort = abort_with_evidence
            publish(stage='initializing', trial=name)
            seed_all(2010)
            try:
                sim = Simulator({**plan,'seed':2010}, trial_root, 1, 'train', plan['train_steps_per_round'])
                row.update(ready=True, initial_status=sim.current['status'])
                assert sim.current['status']['steps'] == 0
                sim.close()
            except Exception as error:
                row.update(ready=False, error_type=type(error).__name__, error=str(error),
                           traceback=traceback.format_exc())
            finally:
                Simulator.abort = original_abort
            row.update(elapsed=time.time()-row['started_at'], after=memory_state())
            rows.append(row); write(root/'trials.json', rows)
            publish(stage='trial_complete', trial=name, result={k:v for k,v in row.items()
                    if k in ('ready','elapsed','pre_abort_evidence','error_type')})
        after = hash_parameters((n,p) for n,p in model.named_parameters() if p.requires_grad)
        assert after == before
        write(root/'result.json', {'trials':rows, 'model_unchanged':True,
                                  'controls_executed':0,'optimizer_steps':0})
        publish(state='complete',stage='complete')
    except BaseException as error:
        write(root/'failure.json', {'type':type(error).__name__,'traceback':traceback.format_exc()})
        publish(state='failed',stage='failed',error_type=type(error).__name__)
    finally:
        signal.alarm(0)
        archive = root.parent/'report.tar.gz'
        with tarfile.open(archive,'w:gz') as t:
            for p in root.rglob('*'):
                if p.is_file() and p.suffix in ('.json','.log'):
                    t.add(p,arcname=str(p.relative_to(root)),recursive=False)
        put(os.environ['B1K_RESULT_PUT'], archive.read_bytes())
        put(os.environ['B1K_STATUS_PUT'], (root/'status.json').read_bytes())


if __name__ == '__main__':
    main()
