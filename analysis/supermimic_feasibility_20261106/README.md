# B1K differentiable world-model feasibility

Purpose: test whether real B1K sampling can be amortized by a learned action-conditioned model. This directory is a separate experiment. It does not alter an existing PPO run, source checkpoint, reward or evaluator.

Completed first-stage result: 64 real trajectories / 10,748 chunks, two three-seed CPU studies, and 39 implementation tests. Action conditioning improves short-term proprioception prediction, but task reward and goal-event prediction do not beat the no-action control, including a separate terminal-inclusive study. This version is not ready to replace real PPO sampling. See `feasibility_report.md` and `evidence_summary.json` for the scoped conclusion.

## What is implemented

1. `profiling/analyze_existing.py` reconstructs actual costs from the completed multi-round campaign, including a separate analysis of available evaluation initialization logs. Missing training-side component timings remain missing.
2. `data/export_transitions.py` converts trusted, checksum-verified original trajectories into compact chunk transitions. It preserves executed physical actions and duration masks, holds out whole instances, and marks missing final next observations explicitly. The current visual descriptor is only 8x8 pooled RGB per camera, not a learned semantic or complete physical state.
3. `model/train_world_model.py` compares an action-conditioned residual predictor with an equal-capacity no-action predictor and persistence. Normalization and visual PCA use training instances only. The primary probe excludes partial chunks: their actual early-stop length is outcome information unavailable before execution. Validation selects epochs; test data never do.
4. `actor/differentiable_chunk.py` reproduces the existing Flow-SDE sampling law while preserving gradients from one action chunk to the action expert. The factual vision/language prefix remains fixed. An explicitly selected ODE path corresponds to the evaluation law. No predicted observation is passed off as factual PPO replay.
5. `instrumentation/recorder.py` provides opt-in physical-state, control-count, terminal-observation and timing hooks for a future separately budgeted simulator probe. It has not yet been validated in a live simulator.
6. `bridge/` connects a saved small world model to the differentiable action interface. A gradient through the surrogate is an implementation result, not evidence that the corresponding real action improves task return.
7. `model/proposal_outcome_probe.py` separately retains terminal labels using only the pre-action full proposal and time budget. It is an outcome predictor, not a physical model of the unexecuted suffix. This checks that the terminal filtering in the dynamics study did not by itself cause the negative task-prediction result.

## Scope and resource accounting

See `plan.json`. Phase one uses old data, CPU compute, and one bounded CPU-only export job if the AFS data are not locally mounted. There is no new environment interaction or new GPU job in phase one. The export reads the first completed round's 64 original trajectories, verifies receipt SHA256, and uploads only compact derived features to the user's existing private AOSS bucket. It writes no new AFS artifacts. Source trajectories stay unchanged.

Training/validation/test use instances 1–10 / 11–13 / 14–16. These are held-out **training-pool** instances for model validation, not the public evaluation instances 301+. Generalization across these three groups is not a test of unobserved action interventions or other tasks.

## Reproduce

Use the already installed project environment, for example:

```bash
PYTHON=analysis/rlinf_adaptation_20261020/venv/bin/python
$PYTHON analysis/supermimic_feasibility_20261106/profiling/analyze_existing.py
$PYTHON analysis/supermimic_feasibility_20261106/submit_export.py
# The preceding command is a dry run; an actual submission is deliberately separate.
$PYTHON analysis/supermimic_feasibility_20261106/fetch_export.py
$PYTHON analysis/supermimic_feasibility_20261106/model/train_world_model.py \
  --data analysis/supermimic_feasibility_20261106/data/transitions.npz \
  --output analysis/supermimic_feasibility_20261106/model/real_round1 \
  --epochs 30 --threads 8 --seed 7 --ensemble 3
```

The export submitter checks the project submission freeze, global submission cadence and queue limits. A recorded submission intent is not silently retried. Do not launch another export merely because no final artifact is visible yet.

## Interpretation requirements

- Separate interpolation on logged actions from correctness on alternative actions. A model can predict factual observations well while giving misleading action gradients.
- Compare against no-action and persistence, not just absolute reconstruction loss. Unchanged goal bits dominate this dataset, so report change-event precision/recall/AP and counts per goal.
- A condition on actual early-stop duration leaks the event being predicted. Primary action-relevance results must use full requested chunks or an explicitly modeled stopping process.
- World-model reward/goal outputs are learned surrogates. Event-weighted training scores are not calibrated real success probabilities.
- One/two/four-chunk open-loop tests replay recorded actions without resetting the latent to future observations. They still do not test a new closed-loop policy.
- CPU model throughput excludes the VLA, renderer and physics. It is not a measured end-to-end speedup.
- A useful decision to expand requires real simulator checks of action ranking and directions, after validating restoration and logging. Prediction metrics or small Gemma gradient tests alone cannot establish that.

## Relation to published methods

- [SuperMimic](https://arxiv.org/abs/2610.28480) motivates short differentiable model rollouts and a bootstrapped critic. Its rigid-body tracking setup and state/reward access differ from B1K's visual manipulation interface.
- [PWM](https://arxiv.org/abs/2407.02466) studies policy gradients through learned models and explains why reconstruction accuracy alone is insufficient for control.
- [WoVR](https://arxiv.org/abs/2602.13977) directly studies VLA optimization in learned visual simulators, with limited-horizon initializations and model-policy distribution alignment. This is an alternative architectural route, not evidence that a small B1K transition MLP can replace the current simulator.

The engineering implementation here is not a novelty claim or a reproduction of any paper's reported performance.
