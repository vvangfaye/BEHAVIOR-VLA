"""Publish a diagnostic-only curation proposal; frozen training inputs stay untouched."""
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parent
CAMPAIGN=ROOT.parent
BUNDLE=CAMPAIGN/'formal_stage/bundle'
sys.path.insert(0,str(BUNDLE/'openpi/src'))
from openpi.training.subtask_mixture import MixturePlan


def main():
    selection=json.loads((ROOT/'review_plan.json').read_text())['episodes']
    reviews=[json.loads((ROOT/e['exposure']['root_id']/'review.json').read_text()) for e in selection]
    visual=[json.loads((ROOT/e['exposure']['root_id']/'visual_samples.json').read_text()) for e in selection]
    assert len(reviews)==len(visual)==12
    assert all(r['finite_actions'] and r['finite_states'] and r['outcome']['success'] for r in reviews)
    comments={
      'freeze_fruit':('local_low_progress_candidate','128–140秒：夹爪在柜门把手附近反复微调，三路画面未见明显开门进展；10Hz关节仍变化。不是全身静止，也没有接触/逐步开门谓词证明完全无效。保留后续成功开门和取盒子。'),
      'organizing_school_stuff':('local_low_progress_candidate','450–462秒：右手在地面笔附近持续闭合、小幅调整，画面未见取笔完成；1Hz的439–505秒为更大复核范围。稀疏状态和图像不足以精确判定每次抓取成败。保留最终取笔/放入袋子的有效片段。'),
      'spraying_for_bugs':('label_completion_timing_review','喷药过程有持续工具操作；关开关阶段约148秒按钮已红，之后标签仍为turn off直至约161秒，存在完成后标签延续。可包含合法撤手动作，不能直接认定错误标签或猜测替换。'),
      'clean_your_rusty_garden_tools':('navigation_efficiency_review','长move-to片段可见绕车辆、建筑物接近工具箱；不能把长导航或持物时手部稳定当作停滞。建议降低重复曝光并复核路线效率。'),
      'collecting_childrens_toys':('long_manipulation_review','两段长push-to操作中物体与桌椅相对位置有变化，同时有反复接近/推移；需要目标谓词或连续画面判断每次推移是否有效，不直接判为标签错误。'),
      'cook_broccolini':('sampled_keep_candidate','抽查的持盘导航、移锅上炉过程有可见阶段变化，与记录标签大体相符；未确认无效抓取或对象错配。'),
      'hanging_pictures':('sampled_keep_candidate','抓取海报和挂墙阶段有对位、提起和靠墙操作；动作较长，未确认可剔除的空抓循环。'),
      'sweeping_garage':('sampled_keep_candidate','持扫帚移动与扫地属于持续工具操作；手部/标签长期不变不能单独作为删除依据。'),
      'organizing_art_supplies':('sampled_keep_candidate','取笔、持物与搬运收纳盒时可见对象状态和场景变化；抽查未发现确定的标签对象错配。'),
      'installing_smoke_detectors':('sampled_keep_candidate','取报警器后持物导航到墙面，画面和标签阶段基本相符；不能按手腕稳定误删持物阶段。'),
      'tidying_bathroom':('sampled_keep_candidate','导航和取卫生纸过程有持续接近、对位和抬手变化；尚未确认局部无效循环。训练见过该实例并不保证评测成功。'),
      'clean_up_broken_glass':('sampled_keep_candidate','接近碎玻璃及持物走向垃圾桶的画面与标签基本相符，未发现可确认的对象错配。')}
    budget=json.loads((ROOT/'proposed_sampling_budget.json').read_text())
    quotas={e['root_id']:e['proposed_max_draws'] for t in budget['task_budgets'] for e in t['episodes']}
    rows=[]
    for r,v in zip(reviews,visual):
        decision,notes=comments[r['task']]
        rows.append(dict(task=r['task'],instance_id=r['instance_id'],root_id=r['root_id'],
            frames=r['frames'],actual_old_draws=r['actual_training_draws'],
            proposed_max_draws_before_quality_filter=quotas[r['root_id']],
            state_action_samples_1Hz=r['sampled_frames'],
            exact_static_runs_over10sec=len(r['low_motion_runs']),
            visual_timepoints=sum(len(w['frames']) for w in v['windows']),
            decision=decision,notes=notes,
            visual_evidence=';'.join(str(ROOT/r['root_id']/w['contact_sheet']) for w in v['windows'])))
    with (ROOT/'review_summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    holds=[dict(root_id='full_episode_02e04ff5b9a1d5b0a51f',start_frame=13500,end_frame=13860,
        reason='visual-supported low-progress candidate,10Hz follow-up450–462s',status='provisional_hold_pending_semantic_review'),
        dict(root_id='full_episode_b2a8c79775ac38186eff',start_frame=3840,end_frame=4200,
        reason='visual-supported low-progress candidate,10Hz follow-up128–140s',status='provisional_hold_pending_semantic_review')]
    mask=dict(kind='diagnostic_only_proposed_quality_mask',enabled_for_training=False,
        source_data_modified=False,interval_convention='[start_frame,end_frame)',
        proposed_hold_intervals=holds,
        window_rule='Reject a32-frame target window if [anchor,anchor+32) intersects any held/rejected frame. Never concatenate separated retained ranges.',
        no_relabeling_without_review=True,whole_episodes_deleted=0,
        interpretation='These are temporarily held candidates, not measured ground-truth failure intervals or a full270-episode acceptance.')
    (ROOT/'quality_mask.proposed.json').write_text(json.dumps(mask,ensure_ascii=False,indent=2)+'\n')
    plan=MixturePlan.load(BUNDLE/'inputs/compiled/mixture.json')
    source=next(s for s in plan._phases[0]['sources'] if s['kind']=='full_episode')
    totals={};windows={};removed=0
    for task,instances in source['balanced_tasks']:
        count=0
        for _,episodes in instances:
            for (root,number),segments in episodes:
                candidates=[]
                for s in segments:
                    for frame in range(s['start'],s['end']-31,32):
                        if any(h['root_id']==root and frame<h['end_frame'] and frame+32>h['start_frame'] for h in holds):continue
                        candidates.append(frame)
                    n=s['end']-s['start']-31
                    for h in holds:
                        if h['root_id']==root:
                            removed+=max(0,min(s['end']-31,h['end_frame'])-max(s['start'],h['start_frame']-31))
                assert len(candidates)==len(set(candidates))
                quota=min(256,int(1920/len(instances)/len(episodes)),2*len(candidates))
                # A metadata-only window schedule proves the limits are feasible.
                rng=random.Random('high-exposure-proposal-v1:'+root)
                first=list(candidates);rng.shuffle(first)
                if quota<=len(first):chosen=first[:quota]
                else:
                    second=list(candidates);rng.shuffle(second);chosen=first+second[:quota-len(first)]
                assert len(chosen)==quota<=256
                assert max((chosen.count(x) for x in set(chosen)),default=0)<=2
                windows[root]=dict(episode_index=number,task=task,draws=quota,
                    unique_stride32_windows=len(candidates),anchor_frames=chosen)
                count+=quota
        assert count<=1920
        totals[task]=count
    assert len(windows)==270
    total=sum(totals.values())
    schedule=dict(kind='metadata_only_candidate_window_schedule_not_loader_input',training_enabled=False,
        seed='high-exposure-proposal-v1',global_draws=960000,per_task_draws=9600,
        full_draws=total,original_draws=960000-total,full_fraction=total/960000,
        max_full_fraction_within_any_task=.2,max_draws_per_full_episode=256,
        action_horizon=32,anchor_stride=32,max_repetitions_of_same_full_window=2,
        task_full_draws=totals,full_windows=windows,
        raw_quality_unknown_windows_still_require_full_library_review=True)
    (ROOT/'candidate_window_schedule.metadata_only.json').write_text(json.dumps(schedule,ensure_ascii=False,separators=(',',':'))+'\n')
    findings=dict(kind='high_exposure_complete_success_review',created_at=time.time(),
        scope=dict(selection='top10 highest actual sampler exposures plus2 exact-instance regressions',trajectories=12,
            source_frames=sum(r['frames'] for r in reviews),one_Hz_action_state_timepoints=sum(r['sampled_frames'] for r in reviews),
            camera_views=432,dense_followup_action_state_timepoints=242,dense_followup_camera_views=42,
            full_continuous_video_review=False,full270_quality_acceptance=False,
            new_training_jobs=0,cloud_jobs_created=0,source_or_training_bundle_modified=False),
        findings=dict(all_sampled_arrays_finite=True,static_runs_over10sec_detected=0,
            gripper_cycle_threshold='4 or more1Hz width-state transitions within30s of an unchanged label',
            gripper_cycle_candidates_at_that_threshold=0,
            confirmed_wrong_object_label_intervals=0,visual_low_progress_candidates=holds,
            label_timing_review_task='spraying_for_bugs317,about148–161s',
            caveat='No object/contact/attachment/per-step goal trace in archive. Samples can miss brief failures. No claim that ambiguous grasps are proven empty or all labels correct.'),
        per_trajectory=rows,
        proposed_filtering=dict(
            integrity='Require verified full success, terminated and not truncated, finite aligned action/state/video and unchanged identity. Reject corrupt episodes; preserve raw bytes.',
            low_progress='Use low-motion>=10s or repeated attempts only to trigger review. Confirm no object/goal progress from video or goal traces before rejecting action windows.',
            grasp='Review>=3 close/reopen cycles in30s on same object with no visible pickup. Keep final successful attempt and useful recovery; gripper width alone is insufficient.',
            label='Wrong object/skill or stale completion label: hold affected windows. Correct only when reviewer can identify label and exact boundary; no guessed automatic fill. Full CE remains zero.',
            protected='Retain real navigation, carrying, continuous sweeping/spraying and required physical waiting. Long duration or stable wrists alone are not rejection criteria.',
            locality='Reject/hold32-frame windows intersecting suspect frames; retain complete provenance and original source. Do not splice across gaps.',
            library_gate='Apply the same analysis and recorded decisions to all270 before claiming a fully filtered dataset. Current12 are a targeted pilot.'),
        proposed_sampling=dict(
            task_selection='Uniform over100 tasks; exact9600 draws/task for a960000-draw budget.',
            full_within_task_ceiling=.2,
            source_fallback='No qualified full windows or quota exhausted: use original expert from same task.',
            hierarchy='Uniform full task-instance then episode; deterministic per-episode quotas capped at256; spare capacity falls back to expert rather than other full episodes.',
            decorrelation='32-frame windows with stride32 inside each retained label span; sample without replacement, at most2 passes.',
            full_before_quality_filter=budget['full_draws_before_quality_filter'],
            full_after_two_provisional_holds=total,
            original_after_two_provisional_holds=960000-total,
            fraction_full_after_two_provisional_holds=total/960000,
            removed_all_stride1_anchors_by_two_holds=removed,
            quality_filter_of_entire_library_pending=True,
            tuning_status='Conservative proposed cap, not empirically established optimal mixture. 5000-step budget used only for like-for-like arithmetic, not a training authorization.'),
        unchanged_for_a_future_control='Start from1688977; retain norm statistics, original text loss, full text mask and existing LR contract; isolate data/sampling changes before changing architecture/learning rate.',
        limitations=['Selection intentionally biased to high exposure; cannot estimate bad-frame rate over270.',
            'Two12-second hold candidates are too small to establish the cause of18-pair regression.',
            'Sparse visual evidence supports localized review, not a claim that every intervening frame is bad.',
            'Attribute content hash matches frozen index, and reads are bound to objectETag. No new full-archive content rehash.'],
        evidence_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(ROOT.glob('full_episode_*/review.json'))+sorted(ROOT.glob('full_episode_*/visual_samples.json'))})
    (ROOT/'findings_and_proposal.json').write_text(json.dumps(findings,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(episodes=12,timepoints=findings['scope']['one_Hz_action_state_timepoints'],
        camera_views=474,full_proposed_draws=total,full_fraction=total/960000,
        removed_anchors_by_two_holds=removed,training_enabled=False),ensure_ascii=False))


if __name__=='__main__':main()
