"""Run a closed-loop public-test shard against one immutable checkpoint on SCO."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request


def run(payload):
    code = Path(payload['code'])
    runtime = Path(payload['runtime'])
    checkpoint = Path(payload['checkpoint'])
    root = Path(payload['assignment_root'])
    root.mkdir(parents=True, exist_ok=True)
    provenance_bytes = (code / '_PUBLIC_TEST_SOURCE_READY').read_bytes()
    assert hashlib.sha256(provenance_bytes).hexdigest() == payload['source_manifest_sha256']
    provenance = json.loads(provenance_bytes)
    assert provenance['checkpoint_step'] == 1688977
    base_bytes = (code / '_CODE_SNAPSHOT_READY').read_bytes()
    assert hashlib.sha256(base_bytes).hexdigest() == provenance['base_source_sha256']
    base = json.loads(base_bytes)
    for name, expected in base['overlay_files_sha256'].items():
        if name not in provenance['files_sha256']:
            assert hashlib.sha256((code / name).read_bytes()).hexdigest() == expected
    assert json.loads((checkpoint / '_READY').read_text())['checkpoint_step'] == 1688977
    for name, expected in provenance['files_sha256'].items():
        assert hashlib.sha256((code / name).read_bytes()).hexdigest() == expected
    for row in provenance['checkpoint_files']:
        assert (checkpoint / row['relative_path']).stat().st_size == row['bytes']
    assert (runtime / '_RUNTIME_READY').is_file()
    assert (runtime / 'graphics-runtime/_READY').is_file()
    graphics_bytes = (runtime / 'graphics-runtime/_READY').read_bytes()
    assert hashlib.sha256(graphics_bytes).hexdigest() == base['graphics_runtime_manifest_sha256']
    graphics_ready = json.loads(graphics_bytes)
    for name, expected in graphics_ready['libraries_sha256'].items():
        assert hashlib.sha256((Path(graphics_ready['library_path']) / name).read_bytes()).hexdigest() == expected
    library = provenance['supplemental_library']
    library_root = Path(library['root'])
    library_bytes = (library_root / '_READY').read_bytes()
    assert hashlib.sha256(library_bytes).hexdigest() == library['ready_manifest_sha256']
    library_ready = json.loads(library_bytes)
    for name, expected in library_ready['libraries_sha256'].items():
        assert hashlib.sha256((Path(library_ready['library_path']) / name).read_bytes()).hexdigest() == expected
    assert payload['instance_ids'] and all(301 <= i <= 320 for i in payload['instance_ids'])
    assert len(set(payload['instance_ids'])) == len(payload['instance_ids'])
    assert payload['expected_rollouts'] == len(payload['instance_ids'])
    output = root / 'results'
    sys.path.insert(0, str(code / 'scripts'))
    from pi05_eval_queue import EvaluationQueue
    queue = EvaluationQueue(root / 'queue.sqlite3')
    if not (root / 'queue.sqlite3').exists():
        queue.initialize([(payload['task'], i) for i in payload['instance_ids']], max_attempts=3,
                         metadata={'run_id': payload['run_id'], 'mode': 'public_test',
                                   'num_rollouts': 1, 'require_videos': payload['write_video'],
                                   'checkpoint_step': 1688977,
                                   'source_manifest_sha256': payload['source_manifest_sha256']})
    else:
        assert queue.metadata(readonly=True)['run_id'] == payload['run_id']
        assert {(r['task'], r['instance_id']) for r in queue.snapshot(readonly=True)['jobs']} == {
            (payload['task'], i) for i in payload['instance_ids']}
    gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total',
                                        '--format=csv,noheader'], text=True).strip().splitlines()
    assert len(gpu_rows) == 1 and any(n in gpu_rows[0] for n in ('5090', 'N12LP'))
    driver = gpu_rows[0].split(',')[2].strip()
    libraries = [p for parent in ('/usr/local/nvidia/lib64', '/usr/lib/x86_64-linux-gnu',
                                  '/usr/lib64', '/lib/x86_64-linux-gnu', '/lib64')
                 for p in Path(parent).glob('libEGL_nvidia.so.' + driver)]
    assert libraries
    graphics = root / 'graphics'
    graphics.mkdir(exist_ok=True)
    vk = graphics / 'nvidia_icd.json'
    egl = graphics / '10_nvidia.json'
    vk.write_text(json.dumps({'file_format_version': '1.0.0',
                              'ICD': {'library_path': str(libraries[0]), 'api_version': '1.3.242'}}))
    egl.write_text(json.dumps({'file_format_version': '1.0.0',
                               'ICD': {'library_path': str(libraries[0])}}))
    cache = Path('/var/tmp/behavior-1k-runtime') / payload['execution_name']
    cache.mkdir(parents=True, exist_ok=False)
    assert cache.stat().st_dev != runtime.stat().st_dev
    assert shutil.disk_usage(cache).free >= 40 * 1024**3
    env = os.environ.copy()
    for name in list(env):
        if name.startswith(('PI05_', 'SUBTASK_', 'EXTERNAL_SUBTASK', 'PROVIDED_SUBTASK', 'GOALKEEPER_')):
            env.pop(name)
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    env.pop('MAX_STEPS', None)
    env.update(PYTHONNOUSERSITE='1', OMNI_KIT_ACCEPT_EULA='YES', OMNIGIBSON_HEADLESS='1',
               OMNIGIBSON_DISABLE_TORCH_COMPILE='1', OPENPI_DATA_HOME=str(runtime / 'openpi-cache'),
               OMNIGIBSON_DATA_PATH=str(runtime / 'omnigibson-data'),
               PI05_ENV=str(runtime / 'envs/b1k-pi05'), EVAL_ENV=str(runtime / 'envs/b1k-eval'),
               PI05_CKPT=str(checkpoint), PI05_REPO_ID='b1k_full100_joint_subtask_v1_drop_episode_s1',
               SUBTASK_PREDICTION='1', MAX_TOKEN_LEN='512', MAX_SUBTASK_LEN='64', EXECUTE_HORIZON='16',
               PI05_MEM_FRACTION='0.35', XLA_PYTHON_CLIENT_PREALLOCATE='false',
               MODEL='pi05', SERVER_GPU='0', EVAL_GPU='0', PORT='8910',
               MODE='public_test', PROMPT_MODE='whole_task', NUM_ROLLOUTS='1',
               INSTANCE_INDICES=' '.join(str(i - 301) for i in payload['instance_ids']),
               WRITE_VIDEO='1' if payload['write_video'] else '0', OUTPUT_ROOT=str(output),
               SCO_EPHEMERAL_RUNTIME_ROOT=str(cache),
               VK_ICD_FILENAMES=str(vk), __EGL_VENDOR_LIBRARY_FILENAMES=str(egl),
               LD_LIBRARY_PATH=library_ready['library_path'] + ':' +
                   str(runtime / 'graphics-runtime/usr/lib/x86_64-linux-gnu') + ':' +
                   str(runtime / 'envs/b1k-eval/lib') + ':' + env.get('LD_LIBRARY_PATH', ''),
               PI05_EVAL_QUEUE_DB=str(root / 'queue.sqlite3'), PI05_EVAL_QUEUE_WORKER='gpu0',
               PI05_EVAL_QUEUE_RUN_ID=payload['run_id'], PI05_EVAL_QUEUE_GENERATION=str(payload['generation']),
               PI05_EVAL_QUEUE_ATTEMPT_ROOT=str(root / 'attempts'))
    status = {'state': 'running', 'checkpoint_step': 1688977, 'task': payload['task'],
              'instance_ids': payload['instance_ids'], 'expected_rollouts': payload['expected_rollouts'],
              'gpu_rows': gpu_rows, 'source_manifest_sha256': payload['source_manifest_sha256'],
              'mode': 'public_test', 'prompt_mode': 'whole_task', 'privileged_prompts': False,
              'started_at': time.time()}

    def publish():
        status['updated_at'] = time.time()
        (root / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
        print('STATUS_JSON=' + json.dumps(status), flush=True)
        if payload.get('report_put'):
            try:
                request = urllib.request.Request(payload['report_put'], data=json.dumps(status).encode(), method='PUT')
                urllib.request.urlopen(request, timeout=30).close()
            except Exception:
                print('report_upload_failed', flush=True)

    publish()
    try:
        with (root / 'driver.log').open('a', buffering=1) as log:
            result = subprocess.run(['bash', str(code / 'scripts/eval_b1k.sh'), payload['task']],
                                    env=env, cwd=code, stdout=log, stderr=subprocess.STDOUT)
        status['driver_exit_code'] = result.returncode
        assert result.returncode == 0, 'Evaluator or policy server failed; retain all attempts for recovery'
        artifact = queue.create_result_manifest(output, root / 'RESULT_MANIFEST.json', model='pi05',
                                               num_rollouts=1, require_videos=payload['write_video'])
        assert artifact['rollout_count'] == payload['expected_rollouts']
        for i in payload['instance_ids']:
            row = json.loads((output / 'pi05' / payload['task'] / 'json' /
                              f"{payload['task']}_{i}_0.json").read_text())
            assert row['task'] == payload['task'] and row['instance_id'] == i and row['rollout_id'] == 0
            assert isinstance(row['success'], bool) and row['steps'] > 0
            assert math.isfinite(row['q_score']['final']) and 0 <= row['q_score']['final'] <= 1
            assert row['conditioning']['mode'] == 'whole_task' and row['conditioning']['privileged'] is False
            assert row['conditioning']['oracle'] is False
            assert any(event['source'] == 'generated' for event in row['subtask_events'])
            assert all(event['source'] in ('generated','cached') for event in row['subtask_events'])
        if payload['pilot']:
            # Decode actual simulator output before admitting further GPU shards.
            video = output / 'pi05' / payload['task'] / 'videos' / f"{payload['task']}_{payload['instance_ids'][0]}_0.mp4"
            verify_video = '''import cv2,json,sys
from pathlib import Path
p=Path(sys.argv[1]);out=Path(sys.argv[2]);cap=cv2.VideoCapture(str(p))
assert cap.isOpened()
count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS))
assert count>1 and fps>0
frames=[]
for n in [0,count//2,count-1]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,n);ok,frame=cap.read();assert ok and frame.size>0
 path=out/f'pilot-frame-{n}.png';assert cv2.imwrite(str(path),frame)
 frames.append({'frame':n,'path':str(path),'width':int(frame.shape[1]),'height':int(frame.shape[0])})
cap.release()
print(json.dumps({'video':str(p),'frames':count,'fps':fps,'decoded_frames':frames}))
'''
            video_check = subprocess.check_output(
                [str(runtime / 'envs/b1k-eval/bin/python'), '-c', verify_video, str(video), str(root)],
                env=env, text=True, timeout=90)
            status['video_validation'] = json.loads(video_check)
        status.update(state='complete', rollout_count=artifact['rollout_count'], completed_at=time.time())
        publish()
        (root / ('_PILOT_SUCCESS' if payload['pilot'] else '_ASSIGNED_SUCCESS')).write_text(
            json.dumps(status, indent=2) + '\n')
    except BaseException as error:
        status.update(state='failed', error_type=type(error).__name__)
        publish()
        raise
