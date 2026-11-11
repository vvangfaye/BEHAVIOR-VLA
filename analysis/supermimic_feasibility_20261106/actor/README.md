# Single-chunk differentiable B1K actor prototype

This isolated feasibility module consumes the existing `JointPi0RL` model and
one batch of recorded `forward_inputs`. It keeps factual image/state/subtask
tokens fixed, computes the frozen VLM prefix once, and differentiates through
the action expert's denoising chain and physical-action decoder. It does not
generate predicted images or claim a multi-chunk imagined closed loop.

## Contract

`sample_chunk_with_grad(model, forward_inputs, sampling_law="flow_sde", ...)`
returns model actions, physical policy actions, **execution actions**, detached
pooled prefix features, and the noise/chain trace. Stored CPU replay tensors
are moved to `model.device`. The output remains differentiable through the
action expert; prefix features do not carry gradients.

The default `flow_sde` uses the existing training behavior law: one denoising
step is stochastic, selected using the same Python random distribution and
`ignore_last` setting as pinned RLinf. Remaining denoising steps use its ODE
mean. Initial/step noise and the stochastic-step index can be supplied exactly.
The module reuses upstream `rl_sampler.sample_mean_var`; it does not reinterpret
stored PPO likelihoods or create synthetic on-policy samples.

The explicit alternative `sampling_law="flow_ode"` uses the evaluation sampler's
deterministic Euler law, with randomness only from initial noise. That is a
different action distribution from the current Flow-SDE behavior policy and
must be labelled separately in an experiment. The existing configured
execution mapping is applied to both laws; if its gripper snapping is enabled,
its piecewise derivative is retained rather than inventing a straight-through
gradient. The prototype assumes a full 16-step chunk; finite-horizon partial
chunks need an explicit executed-length mask in any future training objective.

`model_actor_loss(sample, score_fn, frozen_modules=(), reference_actions=None,
trust_coef=0)` maximizes one differentiable model score per sample. The callback
receives `execution_actions`, and can return predicted reward plus a terminal
critic value. Its world model and critic should be passed as `frozen_modules`;
parameter gradients are disabled during their forward passes while gradients
with respect to actions are preserved. Original flags are restored afterwards.
The optional physical-action MSE penalty is not a KL or PPO guarantee.

Example, with application-specific model outputs adapted in the callback:

```python
sample = sample_chunk_with_grad(policy, factual_forward_inputs)

def score(actions):
    # starting_state and prefix features describe the recorded factual state.
    prediction = world_model(starting_state.detach(), actions)
    return prediction.reward + gamma * prediction.continuation * critic(prediction.next_state)

loss, metrics = model_actor_loss(
    sample, score, frozen_modules=[world_model, critic],
    reference_actions=recorded_execution_actions, trust_coef=0.01,
)
loss.backward()
```

No existing experiment entrypoint imports this module. No cloud job or simulator
is launched. No model checkpoint is modified by the validation suite.

## Dependencies and CPU verification

- Pinned source: `analysis/rlinf_adaptation_20261020/upstream`.
- Joint model: `BEHAVIOR-VLAs/rlinf_adapter`.
- Existing real small Gemma fixture: `BEHAVIOR-VLAs/rlinf_adapter/tests/test_model.py`.

From the project root:

```sh
analysis/rlinf_adaptation_20261020/venv/bin/python -m pytest \
  analysis/supermimic_feasibility_20261106/actor/tests -q
```

Validation uses actual two-layer Gemma attention, KV caching, adaptive norm,
action projection and Flow-SDE sampler. The value/dynamics callback in the
gradient test is a small nonlinear network, not a fitted B1K world model.

The tests establish exact fixed-noise agreement with both existing behavior
and evaluation samplers, exact trace replay, nonzero finite gradients into
the action expert through the model score, unchanged frozen VLM parameters,
and no world-model/critic parameter gradients. These are implementation gates;
they do not establish predictive accuracy, useful counterfactual gradients,
B1K success improvements, or production GPU/memory cost.
