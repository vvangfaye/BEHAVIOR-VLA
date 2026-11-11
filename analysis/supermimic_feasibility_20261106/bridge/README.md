# Differentiable fitted-model score

`FrozenWorldModelScore` loads the `full.pt` saved by `model/train_world_model.py`. It reuses that exact `WorldModel` architecture and its saved training-only normalization/PCA buffers. Its forward path is all torch: gradients flow from predicted reward and next-state value back into physical execution action commands; fitted world-model parameters remain frozen.

```python
from bridge import FrozenWorldModelScore

world_score = FrozenWorldModelScore.from_checkpoint("/actual/path/full.pt", device="cpu")

# All four inputs are torch tensors. Current observation context is fixed for
# this one-step update. The action is the differentiable actor's output.
score_fn = lambda action: world_score(
    obs_state.detach(), goals.detach(), visual_features.detach(), action,
    goal_valid=goal_valid, visual_valid=visual_valid,
)
loss, metrics = model_actor_loss(sample, score_fn, frozen_modules=[world_score])
```

The argument named `obs_state` is the 23-dimensional checkpoint-normalized proprioception saved by the exporter, before the probe's extra training normalization. The action is `[batch,16,23]` physical execution commands, not flow-model actions or a denoising chain. `visual_features` is `[batch,576]` raw `[0,1]` pooled RGB. Goal columns must follow the data/exporter's column order. The returned score has shape `[batch]` and uses the recorded training reward's original units, including any shaping already present in that reward.

The bridge assumes a complete proposed 16-step chunk and uses a constant all-one mask. It has no argument for hindsight actual execution length. It does not establish that an unseen candidate will execute all 16 actions or that a model trained only on full chunks correctly predicts early termination.

Optional continuation value:

```python
score = world_score(
    obs_state, goals, visual_features, execution_actions,
    value_callback=frozen_value, gamma=0.99, continuation=1.0,
)
```

`frozen_value` receives a dictionary containing `next_obs_state`, `next_visual_features`, `next_goals`, `raw_reward`, `next_normalized_state`, and `next_projected_visual`; it returns one torch value per example. A module callback must have all parameters frozen. Function callbacks must similarly use no trainable parameters, or have their modules frozen by the surrounding actor-loss context. Do **not** wrap the value/model call in `torch.no_grad`: that would destroy the needed derivative through its inputs.

`continuation` is required when adding value because the probe has no trained termination head. It may be a declared assumption such as 1.0 or an independently validated prediction, not the future observed `done` label. The bridge does not manufacture a calibrated continuation probability. The reward head and weighted-BCE goal scores are also uncalibrated. This is a differentiable surrogate score, not a proven Q function, true simulator derivative, policy-improvement guarantee, or evidence of reduced sampling cost.

CPU tests load checkpoints built and briefly trained with the actual `WorldModel` architecture, compare against offline transforms and ensemble predictions, verify finite nonzero action/value-path gradients and absent world-model gradients, and reject incompatible checkpoints and malformed inputs:

```bash
PYTHONPATH=/mnt/data1/faye/code/libero_rlinf_20261103/RLinf \
/mnt/data1/faye/code/libero_rlinf_20261103/RLinf/.venv/bin/python -m unittest \
  discover -s analysis/supermimic_feasibility_20261106/bridge -p 'test_*.py' -v
```
