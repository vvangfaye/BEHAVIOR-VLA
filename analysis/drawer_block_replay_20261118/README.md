# Blocked drawer replay analysis

Analyze saved SFT and PPO continuations from the same recorded action prefix in
the LIBERO drawer experiment. The analysis checks the starting state, original
520-action horizon, matched noise streams, and agreement between each recorded
outcome and its native end-of-episode metrics. It reports the paired difference
in success, a conservative pointwise interval, and exact McNemar tests with Holm
adjustment across the supplied blocks. A stable difference also requires both
the screening and confirmation runs to agree in direction.

Run with Python 3.10 or newer from the root of this repository. The scripts use
the sibling drawer pilot's paired-interval code. Plotting requires Matplotlib.
Alternate-host reconstruction checks additionally require Pillow.

```sh
python analysis/drawer_block_replay_20261118/analyze.py \
    --input-dir replay-records --reference-dir recorded-prefixes \
    --plan analysis-plan.json --source-manifest provenance.json \
    --output analysis.json --table blocks.csv
python analysis/drawer_block_replay_20261118/plot_results.py analysis.json blocks.png
```

Supply the inputs from your own evaluation:

- `replay-records/seed_<id>/source.json` records the state and observation hashes
  for each tested block. `screen/block_<step>.json` and
  `confirm/block_<step>.json` under that seed directory store the paired
  continuations.
- `recorded-prefixes` contains `prefix_<id>.json` and `paired_<id>.json`
  for each original rollout and landmark. Use the formal holdout records
  from the [drawer pilot](../drawer_regression_pilot_20261118/).
- The analysis plan records the `states` (ordered action steps per scene),
  `landmarks` (one action step per scene), and `phases` (pair counts per
  `screen` or `confirm`).
- The provenance manifest records `ppo_actor_sha256` and a `scenes` object
  keyed by scene ID. Each entry specifies `landmark`, `prefix_sha256`, and
  `record_sha256`, recorded from the original verified prefix and paired
  record before analysis. Keep this manifest independent of the continuations
  it verifies.
- For an alternate-host reconstruction, also pass `--anchor-dir` and provide the
  original root batch and four main-camera JPEGs per scene. Pin their basenames
  and digests in that scene's `reference_sha256s` object, and supply the
  reconstruction passes, gate evidence, and exported camera frames under
  `replay_diagnostics` in each replay directory.

The analysis is conditional on the selected trajectories and starting steps.
Original rollouts, checkpoints, and digest manifests are supplied separately.
