# HiPhys-RSI radio-task reproduction

This is an independent, single-task reproduction inspired by the [HiPhys-RSI technical report](https://behavior1k-champion.com/assets/HiPhys-RSI_tech_report.pdf). It targets the BEHAVIOR Challenge task `turning_on_radio` with the official RGB-D observation wrapper. The policy detects the radio from head-camera RGB-D and uses robot proprioception for approach and button pressing. The separate training capture can use simulator annotations to audit perception.

Use an activated BEHAVIOR Challenge evaluation environment with OmniGibson, the challenge assets, and the stock R1Pro URDF. The scripts also use NumPy, OpenCV, SciPy, and PyTorch. Run these commands from this directory, replacing paths as needed. Use a new output directory for each capture or evaluation.

```sh
python capture_train.py --instances 8 18 --output outputs/capture
python audit_capture.py outputs/capture

python run_eval.py --repo ../../BEHAVIOR-1K --urdf /path/to/r1pro.urdf \
  --mode train --train-ids 8 18 --output outputs/pipeline-train
```

For paired train selection, run the same full-horizon training instances with `--candidate vla --vla-port PORT`, pointing to a compatible policy server already running on localhost. Then build a routing card from the two summaries and use it as the freeze manifest for the routed candidate.

```sh
python policy_card.py --pipeline-train outputs/pipeline-train/summary.json \
  --vla-train outputs/vla-train/summary.json --output outputs/policy-card.json

python run_eval.py --repo ../../BEHAVIOR-1K --urdf /path/to/r1pro.urdf \
  --mode public_test --candidate routed --vla-port PORT \
  --freeze outputs/policy-card.json --output outputs/routed-public
```

The plan records the development and public splits. Public evaluation uses all ten declared cases at the official horizon. The runner checks the policy and URDF hashes against the freeze manifest and verifies training evidence before evaluating. Captures, traces, videos, summaries, and policy cards are generated locally.
