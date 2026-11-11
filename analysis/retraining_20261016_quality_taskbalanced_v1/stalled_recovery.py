"""Admit one restart per confirmed stalled episode after its old writer stops."""
import hashlib
import shlex
import subprocess
import time
from pathlib import Path

from run_dual_evaluation import module, read, save
from mixed_recovery import fetch

ROOT = Path(__file__).resolve().parent
CASES = ((0, 'paired_fixed20_step5000_stall_recovery_g1'),
         (1, 'expert_fixed20_step1000_stall_recovery_g1'),
         (0, 'paired_fixed20_step5000_c11_recovery_g2'),
         (0, 'paired_fixed20_step5000_c05_recovery_g2'))


def tick(active_by_model):
    counts = [0, 0]
    for model, folder in CASES:
        dst = ROOT / folder
        if not (dst / 'recovery_manifest.json').exists():
            continue
        spec, plan = read(dst / 'recovery_manifest.json'), read(dst / 'plan.json')
        src = ROOT / spec['original_folder']
        expected = read(src / 'frozen_sha256.json')
        assert expected == read(dst / 'frozen_sha256.json')
        assert expected == {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (dst/'frozen').glob('*.py')}
        allowed = {1: ('getting_organized_for_work', 1, 2),
                   11: ('rearrange_your_room', 2, 3),
                   5: ('tidying_bathroom', 2, 3)}
        task, generation, attempt = allowed[spec['index']]
        assert (spec['generation'], spec['attempt']) == (generation, attempt)
        assert plan['pairs'][spec['index']] == [task, 301]
        id = spec['id']
        assert id == 'candidate-%02d' % spec['index']
        ctl = module('restart_' + folder, src / 'controller.py')
        client = ctl.aoss_client()
        path = dst / 'intents' / (id + '.json')
        canonical = src / 'reports' / (id + '.json')
        if not path.exists() and (dst / 'ADMIT').exists():
            proof = read(dst / 'staging_accepted.json')
            assert proof['sha256'] == spec['source_sha256'] and proof['path'] == spec['source_path']
            stopped = read(dst / 'old_writer_stopped.json')
            assert stopped['old_uid'] == spec['old_uid'] and stopped['worker_absent'] is True
            old = read(ROOT / spec.get('old_folder', spec['original_folder']) / 'intents' / (spec['old_intent'] + '.json'))
            state = ctl.observe(old)
            assert state['uid'] == spec['old_uid']
            if state['state'] not in ctl.TERMINAL or active_by_model[model] + counts[model] >= 5:
                continue
            if canonical.exists() and read(canonical).get('state') == 'accepted':
                continue
            intent = dict(id=spec['id'], name=spec['name'], uid=None, gpu=1, image=plan['image'],
                          generation=generation, attempts=attempt, original_uid=spec['old_uid'],
                          creation_attempted=False, created_at=time.time())
            ctl.new(path, intent)
            loader = ('from pathlib import Path;import hashlib;p=Path(' + repr(spec['source_path']) +
                      ');b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==' +
                      repr(spec['source_sha256']) + ';exec(compile(b,str(p),"exec"))')
            command = shlex.join(['timeout','--signal=TERM','--kill-after=60','29000',
                                 plan['runtime']+'/envs/b1k-eval/bin/python','-c',loader])
            args = [ctl.CLI,'acp','jobs','copy','--workspace-name=share-space',
                    '--copy-job-name='+plan['gpu_copy_template'],'--name='+intent['name'],
                    '--job-name='+intent['name'],'--worker-spec=n12lp.nn.i10a.1',
                    '--worker-nodes=1','--quota-type=spot','--retry-times=0',
                    '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs',
                    '--container-image-url='+plan['image'],'--env=NVIDIA_DRIVER_CAPABILITIES:all',
                    '--command='+command]
            intent['creation_attempted'] = True
            save(path, intent)
            try:
                cp = subprocess.run(args, env=ctl.environment(), capture_output=True, text=True, timeout=90)
                intent['cli_exit_code'] = cp.returncode
            except subprocess.TimeoutExpired:
                intent['outcome'] = 'unknown'
            save(path, intent)
        if not path.exists():
            continue
        intent = read(path)
        state = ctl.observe(intent)
        intent.update(uid=state['uid'], state=state['state'])
        save(path, intent)
        save(dst / 'observations' / (id + '.json'), state)
        counts[model] += int(state['state'] not in ctl.TERMINAL)
        report = fetch(client, plan['object_prefix']+'/reports/'+id+'.json')
        if not report:
            if state['state'] in ctl.TERMINAL:
                save(dst/'recovery_error.json', {'reason':'Terminal without report','state':state})
            continue
        save(dst/'reports'/(id+'.json'), report)
        if report.get('state') != 'accepted':
            save(dst/'recovery_error.json', {'reason':'Restart not accepted','state':state})
            continue
        assert report['id'] == id and report['worker_bundle_sha256'] == spec['bundle_sha256']
        assert report['protocol_sha256'] == plan['protocol_sha256']
        assert report['checkpoint_manifest_sha256'] == plan['expected_checkpoint_manifest_sha256']
        rec = report['record']
        assert [rec['task'], rec['instance_id']] == plan['pairs'][spec['index']]
        assert rec['policy_seed'] == rec['environment_seed'] == 0 and rec['policy_seed_acknowledged']
        assert rec['artifact_accepted'] and (rec['terminated'] or rec['truncated'])
        if canonical.exists():
            prior = read(canonical)
            if prior.get('state') == 'accepted':
                assert prior['record'] == rec
                continue
            backup = dst / 'prior_original_report.json'
            if not backup.exists():
                ctl.new(backup, prior)
        report = dict(report, recovery_provenance=dict(generation=generation, attempt=attempt,
                      old_uid=spec['old_uid'], source_report_key=plan['object_prefix']+'/reports/'+id+'.json'))
        save(canonical, report)
    return counts
