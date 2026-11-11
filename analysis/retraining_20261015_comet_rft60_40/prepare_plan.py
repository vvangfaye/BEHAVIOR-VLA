"""Prepare a two-source plan from frozen metadata; no cloud writes or training."""
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
OLD = WORKSPACE / 'analysis/retraining_20261013'
NEW = WORKSPACE / 'analysis/retraining_20261015_round3_v2'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def anchor_counts(index):
    episodes = {e['episode_index']: e for e in index['episodes']}
    assert len(episodes) == len(index['episodes'])
    totals = defaultdict(int)
    ends = {}
    for interval in index['intervals']:
        episode = interval['episode_index']
        assert 0 <= interval['start'] < interval['end'] <= episodes[episode]['length']
        assert interval['start'] >= ends.get(episode, 0)
        assert interval['subtask'].strip()
        ends[episode] = interval['end']
        totals[episode] += max(0, interval['end']-interval['start']-31)
    assert all(totals[e] > 0 for e in episodes)
    return totals


def summarize(rows):
    counts = Counter(r['task_name'] for r in rows)
    pairs = Counter((r['task_name'], r['task_instance_id']) for r in rows)
    return dict(episodes=len(rows), frames=sum(r['length'] for r in rows), tasks=len(counts),
                unique_task_instance_pairs=len(pairs),
                legal_32_action_windows=sum(r['legal_32_action_windows'] for r in rows),
                per_task_episode_min=min(counts.values()), per_task_episode_max=max(counts.values()),
                per_task_episodes=dict(sorted(counts.items())),
                repeated_task_instance_pairs=sum(n > 1 for n in pairs.values()),
                repeated_instance_is_not_content_duplicate=True)


def main():
    parent_path = OLD / 'preparation/data_preparation_manifest.json'
    assert sha(parent_path) == 'bcfef52fd6e8be6892c409161e01e2e773aa2bcee48702544124042dc86b6357'
    parent = read(parent_path)
    holdout = {tuple(x) for x in parent['held_out_task_instances']}
    assert len(holdout) == 102
    replay_path = OLD / 'training_integration_v1/replay_index.json'
    replay = read(replay_path)
    assert replay['source_kind'] == 'replay'
    replay_anchors = anchor_counts(replay)
    expert_rows = [dict(e, root_id=replay['root_id'], legal_32_action_windows=replay_anchors[e['episode_index']])
                   for e in replay['episodes']]
    assert len(expert_rows) == 15949
    recovered = read(OLD / 'training_integration_v1/recovered_index.json')
    original_ids = {e['episode_index'] for e in expert_rows}
    assert not original_ids.intersection(e['episode_index'] for e in recovered['episodes'])
    old_content = read(NEW / 'old_collected_content.json')
    assert old_content['parent_sha256'] == sha(parent_path)
    old_content_by_key = {r['key']: r['episodes'] for r in old_content['indexes']}
    old_indexes = {read(p)['root_id']: p for p in (OLD/'training_integration_v1/compiled_v2/indexes').glob('*_full_episode.json')}
    old_proofs = [a for a in parent['archives'] if a['kind'] == 'full_episode']
    rows = []
    index_sources = []
    for proof in old_proofs:
        assert proof['accepted'] and proof['target']['full_destination_readback']
        path = old_indexes[proof['root_id']]
        index = read(path)
        assert index['source_kind'] == 'full_episode'
        assert index['provenance']['language_loss_enabled'] is False
        anchors = anchor_counts(index)
        contents = old_content_by_key[proof['index_target']['key']]
        assert len(contents) == len(index['episodes']) == 1
        episode = index['episodes'][0]
        assert contents[0]['content_sha256'] == episode['content_sha256']
        assert proof['acceptance']['complete'] and proof['acceptance']['frames'] == episode['length']
        rows.append(dict(episode, generation='old_188', root_id=proof['root_id'],
                         legal_32_action_windows=anchors[episode['episode_index']],
                         training_oss_accepted=True, source_archive=proof['source_key'],
                         export_target=proof['target'], index_target=proof['index_target'],
                         complete_success_evidence='frozen accepted preparation receipt and original selected content index'))
        index_sources.append(dict(path=str(path), sha256=sha(path), root_id=proof['root_id']))
    assert len(rows) == 188
    old_rows = list(rows)
    destination = {r['source_key']: r for r in map(read,(HERE/'evidence/destination_receipts').glob('*.json'))}
    frozen = read(NEW/'source_archives.json')
    new_sources = {r['key']:r for r in frozen['source_archives']}
    new_proofs = {r['root_id']:r for r in map(read,(NEW/'cloud_receipts').glob('*.json'))}
    new_paths = sorted((HERE/'evidence/new_full_indexes').glob('*.json'))
    assert len(new_paths) == len(new_sources) == 82
    for path in new_paths:
        descriptor = read(path)
        proof = new_proofs[descriptor['root_id']]
        source = new_sources[proof['source_key']]
        assert sha(path) == proof['index_target']['sha256']
        index = descriptor['index']
        anchors = anchor_counts(index)
        assert len(descriptor['episodes']) == len(descriptor['action_only_episodes']) == 1
        episode = index['episodes'][0]
        outcome = descriptor['episodes'][0]['environment_result']
        assert outcome['success'] is True and outcome['task_success'] is True
        assert outcome['terminated'] is True and outcome['truncated'] is False
        assert outcome['final_goal_state']['success'] is True
        assert descriptor['action_only_episodes'][0]['language_loss_enabled'] is False
        assert episode['content_sha256'] == source['content_sha256']
        assert episode['length'] == source['frames']
        accepted = proof['source_key'] in destination
        if accepted:
            dest = destination[proof['source_key']]
            assert dest['target']['sha256'] == proof['target']['sha256']
        rows.append(dict(episode, generation='round3_frozen_82', root_id=proof['root_id'],
                         legal_32_action_windows=anchors[episode['episode_index']],
                         training_oss_accepted=accepted, source_archive=proof['source_key'],
                         export_target=destination[proof['source_key']]['target'] if accepted else proof['target'],
                         index_target=destination[proof['source_key']]['index_target'] if accepted else proof['index_target'],
                         complete_success_evidence='native environment success, terminal, no truncation, final goal success'))
        index_sources.append(dict(path=str(path), sha256=sha(path), root_id=proof['root_id']))
    assert len(rows) == 270
    content_hashes = [r['content_sha256'] for r in rows]
    assert len(set(content_hashes)) == len(content_hashes)
    assert not holdout.intersection((r['task_name'],r['task_instance_id']) for r in expert_rows+rows)
    assert sum(r['length'] for r in rows) == 2529538
    available = [r for r in rows if r['training_oss_accepted']]
    ckpt = WORKSPACE/'checkpoints/pi05_b1k_full100_joint_1688977'
    checkpoint = read(ckpt/'SOURCE_MANIFEST.json')
    prior_verification = read(ckpt/'TRANSFER_VERIFIED.json')
    assert prior_verification['state'] == 'verified' and prior_verification['optimizer_state_included'] is False
    assert all((ckpt/e['relative_path']).stat().st_size == e['size'] for e in checkpoint['files'])
    norm_path=ckpt/'assets/b1k_full100_joint_subtask_v1_drop_episode_s1/norm_stats.json'
    norm_sha=sha(norm_path)
    assert norm_sha == '52d63e89f8359143d793ca80d0661c7facb0daff43c7d8c08f0d05572dd528fb'
    audit = dict(kind='two_source_RFT_metadata_inventory', observed_at=time.time(),
                 cloud_observation=read(HERE/'evidence/cloud_metadata_check.json'),
                 original_expert=summarize(expert_rows), old_full_successes=summarize(old_rows),
                 new_full_successes=summarize(rows[188:]), target_full_successes=summarize(rows),
                 training_oss_accepted_full_successes=summarize(available),
                 pending_transfer_full_successes=summarize([r for r in rows if not r['training_oss_accepted']]),
                 metadata_index_audit_passed=True, raw_video_redecode_performed=False,
                 new_content_duplicates=0, holdout_overlaps=0, original_102_holdouts_retained=True,
                 excluded_recovered_episodes=len(recovered['episodes']), excluded_collected_subtask_episodes=5175,
                 checkpoint=dict(source=checkpoint['source'], directory_step=1688977, logical_completed_updates=1688978,
                                 source_manifest_sha256=sha(ckpt/'SOURCE_MANIFEST.json'),
                                 prior_byte_verification_sha256=sha(ckpt/'TRANSFER_VERIFIED.json'),
                                 prior_full_byte_verification=prior_verification['completed_at'],
                                 current_all_21_file_sizes_verified=True, current_weight_full_rehash=False,
                                 normalization_sha256=norm_sha),
                 no_training_or_gpu_smoke_started=True)
    save('data_audit.json',audit)
    save('selected_full_episodes.json',dict(kind='frozen_candidate_data_selection_not_training_ready',
          source_cutoff=read(NEW/'training_data_cutoff_authorization.json'), episodes=rows,
          desired_count=270, training_oss_accepted_count=len(available), content_deduplicated=True,
          index_sources=index_sources, raw_sources_retained=True))
    save('selected_original_expert.json',dict(index_path=str(replay_path),index_sha256=sha(replay_path),
          original_episode_selection_sha256=replay['provenance']['episode_selection_sha256'],
          episodes=expert_rows, imputation_allowed=False, holdout_pairs=sorted(holdout)))

    path=WORKSPACE/'BEHAVIOR-VLAs/openpi/src/openpi/shared/learning_rate_contract.py'
    spec=importlib.util.spec_from_file_location('existing_lr_contract',path)
    lr=importlib.util.module_from_spec(spec);spec.loader.exec_module(lr)
    @dataclass
    class CosineDecaySchedule:
        warmup_steps: int = 200
        peak_lr: float = 1e-6
        decay_steps: int = 20000
        decay_lr: float = 2e-7
    contract=lr.make_contract(CosineDecaySchedule(),peak_limit=1e-6)
    milestones=[dict(completed_updates=n,rate_for_next_update=lr.expected_rate(contract,n))
                for n in [0,1,100,199,200,1000,2500,5000,10000,15000,19999,20000,40000]]
    assert max(lr.expected_rate(contract,n) for n in range(40001)) <= 1e-6
    assert all(lr.expected_rate(contract,n+1) <= lr.expected_rate(contract,n)+1e-20 for n in range(200,40000))
    save('learning_rate_contract.json',contract)
    save('learning_rate_planning_check.json',dict(milestones=milestones,
         formula_checked_every_integer_step_0_to_40000=True, post_warmup_increases=0,
         contract_matches_existing_guard_schema=True, real_JAX_Optax_runtime_checked=False,
         GPU_smoke_or_resume_executed=False))
    full_tasks=audit['target_full_successes']['tasks']
    save('training_plan.json',dict(
        kind='b1k_comet_inspired_two_source_finetune',version=1,created_at=time.time(),
        status='plan_only_not_authorized_to_launch',training_allowed=False,optimizer_updates_allowed=False,
        training_started=False,data_ready_for_target_270=False,training_integration_ready=False,new_jobs_created=0,
        human_scope=dict(checkpoint=1688977,source_mixture={'original_expert':0.6,'successful_full_episode':0.4},
                         peak_lr=1e-6,exclude=['standalone_collected_subtask','recovered_imputed_expert'],
                         clarification='Only exclude standalone subtask dataset; retain original model structure and original expert labels'),
        dataset=dict(original_expert=audit['original_expert'],target_full_successes=audit['target_full_successes'],
                     training_oss_accepted_full_successes=audit['training_oss_accepted_full_successes'],
                     remaining_full_episodes_to_transfer=270-len(available),
                     exact_original_selection='selected_original_expert.json',exact_full_selection='selected_full_episodes.json',
                     no_silent_fallback_to_188_or_204=True,no_new_rollouts_or_later_arrivals=True,
                     original_102_holdouts_retained=True,public_collected_training_seen=True),
        initialization=dict(params_path=str(ckpt/'params'),checkpoint=audit['checkpoint'],
                            mode='1688977_weights_only_fresh_AdamW_and_new_update_clock_zero',
                            load_5000_10000_20000_finetuned_weights=False,load_old_optimizer_or_scheduler=False,
                            normalization='reuse exact 1688977 stats, never recompute'),
        model=dict(pi05=True,action_horizon=32,fps=30,original_joint_subtask_structure_retained=True,
                   action_conditioning='original task plus subtask format; preserve train/inference compatibility',
                   original_expert_loss='action loss + existing human subtask CE',
                   successful_full_loss='action only; recorded generated subtask as conditioning, CE strictly zero',
                   subtask_CE_weight=1.0,action_loss_weight=1.0,ema_decay=0.99,
                   no_new_architecture_or_LoRA=True),
        sampling=dict(unit='legal 32-step action window; probabilities are sample draws, not stored episode ratios',
                      weights={'replay':0.6,'full_episode':0.4,'recovered':0.0,'collected':0.0},
                      global_batch_size=192,proposed_devices=16,proposed_topology='2 workers x 8 GPUs, reuse prior Nebula environment',
                      source_selection='deterministic seeded weighted draws, expectation 60:40, not exact per individual batch',
                      within_source='uniform task, uniform available task-instance, uniform episode within instance; valid-anchor-weighted intervals, uniform anchor',
                      no_cross_episode_or_subtask_boundary=True,seed=42,
                      expected_samples_per_batch={'original':115.2,'full':76.8},
                      full_task_coverage=f'{full_tasks}/100',
                      combined_task_exposure=dict(with_full_successes=0.6/100+0.4/full_tasks,
                                                  without_full_successes=0.6/100,
                                                  maximum_to_minimum=(0.6/100+0.4/full_tasks)/(0.6/100)),
                      mixture_is_not_globally_uniform_over_100_tasks=True),
        optimizer=dict(type='AdamW',beta1=0.9,beta2=0.95,epsilon=1e-8,weight_decay=1e-10,clip_global_norm=1.0),
        learning_rate=dict(contract_file='learning_rate_contract.json',contract_sha256=contract['sha256'],
                           peak=1e-6,warmup_updates=200,decay_endpoint=20000,floor=2e-7,
                           independent_of_execution_budget=True,initial_next_update_lr=milestones[0]['rate_for_next_update'],
                           resume='restore new-run params+optimizer+EMA+step+RNG+data position; verify same LR contract; never repeat warmup',
                           budget_extension='never move decay endpoint; after 20000 remain at floor; extension requires separate authorization'),
        proposed_budget=dict(initial_updates=5000,conditional_maximum_updates=20000,
                             not_a_training_launch_authorization=True,
                             samples_at_5000={'total':960000,'original':576000,'full':384000},
                             samples_at_20000={'total':3840000,'original':2304000,'full':1536000},
                             these_are_repeated_action_window_draws_not_unique_episodes=True,
                             save_every_updates=1000,extra_save_at_updates=[2500],
                             first_scientific_comparison_updates=[2500,5000],
                             continuation_requires_review_and_explicit_scope=True),
        evaluation=dict(regression_panel='freeze 100 tasks x instance301, seed0, native budgets, identical 1688977 baseline',
                        baseline_reference='analysis/retraining_20261015_step10000_task100/baseline_accepted.json',
                        report_training_seen=True,development_holdout_task_instances=102,
                        holdouts_unseen_to_1688977_pretraining_guaranteed=False,
                        metrics=['matched full success rate','matched mean Q','paired gains and regressions','old successful task retention'],
                        invalid_or_missing_runs='separate infrastructure failures; never count as policy failure',
                        choose='prefer baseline if gains are unconvincing; never choose latest checkpoint automatically',
                        no_automatic_evaluation_jobs_created=True),
        gates_before_any_training=[
            'User explicitly authorizes a new training run; this plan grants no authorization.',
            'Finish remaining66 frozen compact exports to training-accessible storage; publish a separate immutable two-source manifest.',
            'Verify candidate270 full exports, exact15949 expert selection, no content duplicates or holdout collisions, correct root mapping.',
            'Build new two-source adapter/config and real decoded CPU batches; old four-source smoke does not validate new mix.',
            'After explicit execution authorization: disposable two-update GPU and optimizer-resume smoke; reload1688977 for main run.',
            'Check actual first-update LR, new clock0, exact unchanged norm hash and zero full-episode CE; persist LR/data contracts in checkpoints.'
        ],
        provenance=dict(comet_postchallenge_example='https://github.com/mli0603/openpi-comet/blob/main/src/openpi/training/config.py',
                        not_an_exact_reproduction='Our joint subtask architecture and collected task coverage differ;60:40 and1e-6 reference their later public config, not a claim of identical 2025 competition setup',
                        data_audit_file='data_audit.json'),
        limitations=['66 cloud-prepared exports not yet transferred to training OSS; no complete270 DATA_READY.',
                     'Metadata and prior byte/reader receipts checked; no new video decode or GPU training verification.',
                     'Original expert library is not rescanned or reuploaded; existing accepted objects are reused.',
                     'Higher full-episode sampling does not guarantee closed-loop gains.']))
    save('HANDOFF.json',dict(updated_at=time.time(),objective='Prepare only:1688977 two-source60:40,peak1e-6',
         latest_human_clarification='Exclude standalone subtask collection and recovered/imputed source; retain original model structure and human expert labels',
         plan='training_plan.json',data_audit='data_audit.json',learning_rate_contract='learning_rate_contract.json',
         original_expert_episodes=15949,target_complete_successful_episodes=270,target_complete_successful_frames=2529538,
         training_oss_accepted_full_episodes=len(available),pending_transfer_full_episodes=270-len(available),
         source_cutoff='Frozen old188 plus new82; no later arrivals or evaluation videos added',
         training_allowed=False,optimizer_updates_allowed=False,new_jobs_created=0,new_transfers_started=0,
         data_ready=False,training_integration_ready=False,
         proposed_first_stage_updates=5000,proposed_conditional_ceiling_updates=20000,
         cloud_check='82 original-byte indexes and export lengths checked through read-only AOSS requests;16 accepted training-OSS receipts found',
         not_verified_this_turn=['fresh AliOSS object readback','full model weight rehash','decoded CPU training batch','GPU optimizer/resume smoke'],
         next_steps=['Finish training-accessible staging of66 compact exports; signing host currently unavailable per prior user report',
                     'Prepare separate two-source training config and immutable manifest after data staging',
                     'CPU batch verification, then only after explicit authorization GPU smoke and formal training'],
         old_artifacts_preserved=True,existing_training_modified=False,background_controller_started=False))
    manifest={p.name:sha(p) for p in HERE.glob('*') if p.suffix in {'.json','.py'} and p.name!='artifact_sha256.json'}
    save('artifact_sha256.json',manifest)
    print(json.dumps(dict(original=audit['original_expert'],full_target=audit['target_full_successes'],
                          training_oss_accepted=audit['training_oss_accepted_full_successes'],
                          missing=audit['pending_transfer_full_successes'],training_allowed=False,
                          learning_rate_milestones=milestones),ensure_ascii=False))


if __name__=='__main__':
    main()
