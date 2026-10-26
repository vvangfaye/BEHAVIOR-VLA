import base64,hashlib,json
from pathlib import Path
config={'train_instances': [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4], 'train_seeds': [[2000, 2001, 2002, 2003, 250004, 250005, 250006, 250007, 250008, 250009, 250010, 250011, 250012, 250013, 250014, 250015, 250016, 250017, 250018, 250019, 250020, 250021, 250022, 250023, 250024, 250025, 250026, 250027, 250028, 250029, 250030, 250031], [2010, 2011, 2012, 2013, 251004, 251005, 251006, 251007, 251008, 251009, 251010, 251011, 251012, 251013, 251014, 251015, 251016, 251017, 251018, 251019, 251020, 251021, 251022, 251023, 251024, 251025, 251026, 251027, 251028, 251029, 251030, 251031]], 'resume': {'source_output': '/mnt/afs/260010165/behavior_1k/rlinf_sampling32_20261025/pilot01/attempt02/results', 'completed_round': 1, 'checkpoint_directory': '/mnt/afs/260010165/behavior_1k/rlinf_sampling32_20261025/pilot01/attempt02/results/checkpoints/round_01', 'expected_trainable_sha256': '5da5dac04c4455b59d5626d82e0194f5dec44ecbfeb6c06dd6579691b438a1c6', 'source_result_sha256': '143aa88a712281e491932436a3ad4347b50a78653023a12e6387ab769967e14d', 'model_file_bytes': 1731476107, 'optimizer_file_bytes': 3462986187, 'prior_initialization_failures': {'round2_episode1': 2, 'round2_episode2': 1}, 'new_generation_initialization_attempts_per_case': 2, 'previous_jobs': [{'name': 'b1k-s32-resume-1026-r0', 'uid': '81fad3d84b0a4d2aa09ef3c5cf106699'}, {'name': 'b1k-s32-resume-1026-r1', 'uid': '59842713e6a541cdb70b385aaf72aa77'}], 'first_round_will_not_repeat': True, 'optimizer_will_not_reset': True, 'runtime_repair': 'remove periodic asynchronous faulthandler dump; preserve natural process exit code', 'diagnostic_evidence_sha256': '6b7d229124e6b95ffc02f9a83b9d6d70dcfb4d397c828772d177d5eecb17a942', 'resource_ceiling_accounting': {'prior_two_workers_hours_upper_bound': 17, 'initial_aborted_workers_hours_upper_bound': 0.667, 'diagnostic_gpu_hours_upper_bound': 1, 'new_workers_gpu_hours_max': 28.833, 'combined_gpu_hours_upper_bound': 47.667, 'original_ceiling_gpu_hours': 48, 'failed_resume_gpu_hours_upper_bound': 0.167}, 'round2_receipts': [{'rank': 1, 'round': 2, 'episode': 2, 'policy_hash': '5da5dac04c4455b59d5626d82e0194f5dec44ecbfeb6c06dd6579691b438a1c6', 'rollout_path': '/mnt/afs/260010165/behavior_1k/rlinf_sampling32_20261025/pilot01/attempt02/results/training/round_02/episode_02/rollout.pt', 'sha256': 'dcab5932d0d84dda3bd3ac79df9dc9570142d97f7eefcf46407222ca810ba521', 'reused': False, 'instance': 2, 'seed': 2011, 'status': {'steps': 1121, 'score': 1.0, 'success': True}}], 'remaining_training_episodes': 31, 'source_jobs': [{'name': 'b1k-s32-1025-a2-r0', 'uid': '5ebc29934b8d4ebc87772dfa7d00dfec'}, {'name': 'b1k-s32-1025-a2-r1', 'uid': '8d1001daef7642bdab4b9a029601e4f2'}], 'recovery_correction': 'Reuse the late-completed round2 episode 2, verified from final AFS contents; early rank0 archive predated this result'}}
prior=Path(config['resume']['source_output'])
expected={r['episode']:r for r in config['resume']['round2_receipts']}
actual={json.loads(x.read_text())['episode']:json.loads(x.read_text()) for x in (prior/'coordination').glob('round_02_episode_*.json')}
assert actual==expected
verified=[]
for r in actual.values():
 i=r['episode']-1
 assert r['instance']==config['train_instances'][i] and r['seed']==config['train_seeds'][1][i]
 assert r['policy_hash']==config['resume']['expected_trainable_sha256']
 p=Path(r['rollout_path']);h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 assert h.hexdigest()==r['sha256']
 verified.append({'episode':r['episode'],'bytes':p.stat().st_size,'sha256':h.hexdigest()})
for p in (prior/'training/round_02').glob('episode_*/progress.json'):
 assert int(p.parent.name.split('_')[-1]) in expected
old_output=Path('/mnt/afs/260010165/behavior_1k/rlinf_sampling32_20261025/resume_20261026/pilot01/results')
assert not old_output.exists(), 'Failed bootstrap unexpectedly executed the driver'
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps({'passed':True,'late_episode_verified':verified,'failed_bootstrap_executed_no_driver':True}).encode()).decode()+'B1KPROBE_END')
