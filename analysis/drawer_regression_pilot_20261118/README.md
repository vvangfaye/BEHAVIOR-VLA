# LIBERO drawer regression pilot

This pilot compares an SFT policy and a PPO checkpoint over matched LIBERO-Long drawer rollouts. Each pair uses the same reset or action-prefix snapshot and shares per-step policy noise. The supplementary stall probe measures geometry at a fixed landmark 80 actions after the new policy first reaches an open drawer with the bowl inside.

Run the commands from this directory in a compatible RLinf/OpenPI evaluation environment with verified LIBERO assets, PyTorch, Hydra, NumPy, Pillow, and a CUDA device. Provide your own Hydra evaluation config, SFT model, PPO checkpoint, and output directory. The config must use the five-step pi0.5 ODE and ten-action LIBERO chunks. Use official, trusted LIBERO assets because the runtime loads LIBERO's serialized initial states.

```bash
python run_pairs.py discovery --output-dir "$PILOT_OUTPUT" \
  --config-dir "$CONFIG_DIR" --config-name "$CONFIG_NAME" \
  --model-dir "$MODEL_DIR" --ppo-checkpoint "$PPO_CHECKPOINT"

python analysis_report.py --discovery "$PILOT_OUTPUT/discovery/pairs.json" \
  --output "$PILOT_OUTPUT/agent_decision.json"

python run_pairs.py holdout --stall-probe --output-dir "$PILOT_OUTPUT" \
  --config-dir "$CONFIG_DIR" --config-name "$CONFIG_NAME" \
  --model-dir "$MODEL_DIR" --ppo-checkpoint "$PPO_CHECKPOINT"

python finalize.py --output-dir "$PILOT_OUTPUT"
python stall_report.py --input-dir "$PILOT_OUTPUT/holdout/stall" \
  --seeds 182101:182116 --output "$PILOT_OUTPUT/stall_report.json"
```

The holdout command checks the discovery digest against the frozen decision. With a custom seed offset, pass the resulting holdout seeds to `stall_report.py`. `recheck_flips.py` uses the same paths and policy options to rerun a limited diagnostic on originally discordant pairs with fresh noise streams. `analysis_report.py`, `finalize.py`, and `stall_report.py` can analyze existing records without loading the simulator or a model.
