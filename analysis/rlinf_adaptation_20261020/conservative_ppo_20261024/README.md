# Two-round conservative PPO pilot — 2026-10-24

Job: `b1k-ppoc2-1024-a01`, UID `9b71a258461e47fcb40e1b54e81e5269`.
One RTX 5090, eight-hour ceiling. All four owner reserved GPUs were occupied at admission, so this submission uses Spot. Other jobs are unchanged.

## Frozen sequence

1. Reuse all six original1688977 baseline results and the three completed old PPO final cases. Recover only old final cases `(302,0)`, `(302,1)`, `(302,2)` with the exact old round-five weights.
2. Reload original1688977 and verify its trainable-parameter hash. Initialize a fresh optimizer; do not restore the old PPO actor or its optimizer.
3. Fit only the value head on up to 64 observations from each prior truly terminal training episode, using discounted observed returns. The truncated failed episode is excluded because its terminal continuation is unknown. Fixed 100 head-only epochs; held-out chunks are reported but never used for checkpoint selection. This is an initialization check, not proof of critic calibration.
4. Two PPO rounds. Each collects four complete on-policy trajectories before any update, using training instances 1–4 and seeds 2000–2003, then 2010–2013. All four episodes share unchanged policy weights within a round. Maximum 25,792 new training controls.
5. Compute GAE within each episode, normalize valid advantages across the whole four-episode batch, shuffle action chunks deterministically, accumulate 32 chunks, and apply one pass. Actor LR 1e-7, value LR 1e-4. Gamma .99, lambda .95, reward, precision, robot, cameras, Flow-SDE sampler and evaluation remain unchanged.
6. After each candidate optimizer step, replay the whole rollout batch. If the nonnegative sampled KL estimate exceeds .01 or becomes nonfinite, restore the complete pre-step trainable parameters and optimizer and stop that update round. This bounds the measured batch estimate, not true KL or arbitrary unseen-state behavior.
7. Evaluate only the prescribed final round-two checkpoint, with instances 301/302, seeds 0/1/2 and the original deterministic evaluation policy. No intermediate checkpoint selection. Check initial states and report regressions as well as gains.

## Recovery and persistence

An initialization attempt can retry once only before any environment controls have executed; it preserves its failed directory and terminates its own simulator child group. Runtime failures after controls are not silently retried. A complete failed episode is a valid result and is never rerun for a better score.

The original experiment directory remains immutable. New outputs and checkpoints are isolated. Each round keeps its actor/value overlay; the latest complete round also keeps its optimizer. Logs and JSON reports go to AOSS separately from large checkpoint files. The background observer is read-only and never resubmits a cloud job.

## Validation

`training_test_result.json` records real tiny-model tests covering head-only warmup, episode return boundaries, whole-batch normalization, shuffling, actual actor gradients, full parameter/Adam rollback on rejected updates, and skipping completed evaluation cases. `afs_preflight.json` verifies instances 1–4 and previous checkpoints/rollouts. See `plan.json`, `submission_intent.json`, and `submission_admission.json` for frozen configuration and admission evidence.
