"""Accept the measured KL gate, then resume the four saved learners once."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from deployment import HERE, ENV, save, describe, aoss_client, submit, worker_plan, storage_proof
from train_controller import all_active_gpus, workers_exited


def main():
    with (HERE/'kl_resume.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        benchmark = json.loads((HERE/'jobs/kl-benchmark-a1/submission.json').read_text())
        client = aoss_client()
        deadline = time.time()+12*3600
        try:
            while time.time() < deadline:
                job = describe(benchmark['job'])
                assert job['uid'] == benchmark['uid']
                try:
                    progress = json.loads(client.get_object(Bucket='260010165',
                        Key=benchmark['aoss_prefix']+'status.json')['Body'].read())
                except client.exceptions.ClientError as error:
                    if error.response['Error']['Code'] not in ('NoSuchKey','404'):
                        raise
                    progress = {}
                save(HERE/'kl_benchmark_status.json', {'state':job['state'],'progress':progress,'at':time.time()})
                if job['state'] == 'SUCCEEDED':
                    gate = progress['result']
                    assert gate['passed'] and gate['training_state_unchanged']
                    assert gate['optimizer_steps'] == 0 and gate['selected_batch_size'] in (4,8)
                    # Benchmark has no optimizer. Reserve its saved state size
                    # plus 2 GiB allocator/driver headroom before selecting a
                    # batch for the 32 GiB training cards.
                    sources = json.loads((HERE/'kl_batch_plan.json').read_text())['sources']
                    usable = []
                    for size in (4,8):
                        okay = all(r['batches'][str(size)]['passed'] and
                            r['batches'][str(size)]['peak_allocated_bytes']+
                            sources[v]['metadata']['optimizer_bytes']+2*1024**3 <= 32*1024**3
                            for v,r in gate['variants'].items())
                        if okay:
                            usable.append(size)
                    assert usable, 'No measured batch leaves room for training optimizer state'
                    gate['selected_batch_size'] = min(usable, key=lambda s: gate['variants']['control']['batches'][str(s)]['seconds'])
                    gate['training_memory_reserve'] = 'saved optimizer byte size + 2 GiB, within a 32 GiB card'
                    assert gate['runtime_signature'] == worker_plan('control',4)['distributed']['runtime_signature']
                    tests = json.loads((HERE/'kl_batch_cpu_tests_result.json').read_text())
                    assert tests['passed']
                    for name, digest in tests['sources'].items():
                        assert hashlib.sha256((HERE/'runtime'/name).read_bytes()).hexdigest() == digest
                    save(HERE/'kl_batch_gate.json', gate)
                    break
                if job['state'] in ('FAILED','SUSPENDED','STOPPED','CANCELED','TERMINATED'):
                    save(HERE/'kl_resume_status.json', {'stage':'benchmark_failed','job':job['name'],'progress':progress})
                    return
                time.sleep(45)
            else:
                raise TimeoutError('Bounded KL benchmark watcher deadline')

            # Old controller cannot launch a competing retry during the handoff.
            process_path = HERE/'controller_process.json'
            if process_path.exists():
                pid = json.loads(process_path.read_text())['pid']
                proc = Path(f'/proc/{pid}/cmdline')
                if proc.exists():
                    command = proc.read_bytes().split(b'\0')
                    if any(x.endswith(b'/rl_multigpu_20261101/train_controller.py') or x == b'analysis/rl_multigpu_20261101/train_controller.py' for x in command):
                        os.kill(pid, signal.SIGTERM)
                        for _ in range(20):
                            if not proc.exists():
                                break
                            time.sleep(.2)
            for variant in ['control','gamma1_ppo','balanced_ppo','group_outcome']:
                receipt = HERE/'jobs'/('t-'+variant.replace('_','-')+'-a4')/'submission.json'
                if receipt.exists():
                    # A durable receipt makes re-running this watcher idempotent.
                    continue
                old = json.loads((HERE/'jobs'/('t-'+variant.replace('_','-')+'-a3')/'submission.json').read_text())
                assert workers_exited(old), 'Old checkpoint writer is still active'
                while time.time() < deadline:
                    try:
                        storage_proof(worker_plan(variant,4))
                    except AssertionError as error:
                        save(HERE/'kl_resume_status.json', {'stage':'waiting_for_storage','variant':variant,
                            'reason':str(error),'checkpoint_preserved':True,'at':time.time()})
                        time.sleep(60)
                        continue
                    active, seconds = all_active_gpus()
                    assert active+4 <= 32 and seconds+4*43200 <= 512*3600
                    row = submit(variant,4)
                    save(HERE/'kl_resume_status.json', {'stage':'resuming','variant':variant,
                        'job':row['job'],'uid':row['uid'],'at':time.time()})
                    break
                else:
                    raise TimeoutError('Bounded resume waiting for storage')
            # Reuse the training controller and all already completed evaluations.
            log = (HERE/'train_controller.log').open('ab')
            child = subprocess.Popen([str(Path(sys.executable).absolute()),'-u',str(HERE/'train_controller.py')],
                env=ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(HERE/'controller_process.json', {'pid':child.pid,'at':time.time(),'generation':'batched-full-kl-resume'})
            save(HERE/'kl_resume_status.json', {'stage':'four_learners_submitted','at':time.time(),
                'kl_batch_size':gate['selected_batch_size'],'saved_updates_preserved':True})
        finally:
            client.close()


if __name__ == '__main__':
    deadline = time.time()+12*3600
    while time.time() < deadline:
        try:
            main()
            break
        except BlockingIOError:
            raise
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            # Quota/API failures must not silently disable automatic recovery.
            # Semantic/identity assertions remain hard failures.
            save(HERE/'kl_resume_transient_error.json', {'at':time.time(),'type':type(error).__name__})
            time.sleep(20)
