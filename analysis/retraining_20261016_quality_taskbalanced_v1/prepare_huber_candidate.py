"""Build an isolated, unsubmitted expert-only robust-loss comparison."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'expert_control/formal_stage/bundle'
DEST = ROOT / 'expert_huber_candidate/bundle'
RUN = 'b1k-expert-huber5-1688977-20261017'
DELTA = 5.0

LOSS_SOURCE = '''"""Action residual objectives; the default preserves squared error exactly."""
import math
import jax.numpy as jnp


def validate_delta(delta):
    if delta is not None and (type(delta) not in (float, int) or not math.isfinite(delta) or delta <= 0):
        raise ValueError("Huber delta must be finite and positive")


def contract(delta):
    validate_delta(delta)
    return {"schema_version": 1, "kind": "squared_error" if delta is None else "scaled_huber", "delta": delta}


def squared_or_huber(residual, delta=None):
    validate_delta(delta)
    if delta is None:
        return jnp.square(residual)
    magnitude = jnp.abs(residual)
    quadratic = jnp.minimum(magnitude, delta)
    return jnp.square(quadratic) + 2 * delta * (magnitude - quadratic)
'''


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + '\n')


def replace(path, old, new):
    source = path.read_text()
    if source.count(old) != 1:
        raise ValueError('Unexpected source: ' + str(path))
    path.write_text(source.replace(old, new))


def main():
    if DEST.exists():
        raise FileExistsError('Never overwrite a prepared experiment')
    old_manifest = json.loads((SOURCE / 'bundle_manifest.json').read_text())
    for name, expected in old_manifest.items():
        if hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Source bundle changed: ' + name)
    shutil.copytree(SOURCE, DEST, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shared = DEST / 'openpi/src/openpi/shared/action_loss.py'
    shared.write_text(LOSS_SOURCE)
    model = DEST / 'openpi/src/openpi/models/pi0.py'
    config = DEST / 'openpi/src/openpi/models/pi0_config.py'
    for path in (model, config):
        replace(path, 'from openpi.shared import array_typing as at',
                'from openpi.shared import array_typing as at\nfrom openpi.shared import action_loss as _action_loss')
    replace(config, '    subtask_loss_weight: float = 1.0',
            '    subtask_loss_weight: float = 1.0\n    action_loss_huber_delta: float | None = None')
    replace(config, '    def __post_init__(self):\n        if self.framesamp_modul:',
            '    def __post_init__(self):\n        _action_loss.validate_delta(self.action_loss_huber_delta)\n        if self.framesamp_modul:')
    replace(model, '        self.subtask_loss_weight = config.subtask_loss_weight',
            '        self.subtask_loss_weight = config.subtask_loss_weight\n        self.action_loss_huber_delta = config.action_loss_huber_delta')
    replace(model, '        action_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)\n        metrics = {"action_loss": jnp.mean(action_loss)}',
            '''        residual = v_t - u_t
        action_loss = jnp.mean(_action_loss.squared_or_huber(residual, self.action_loss_huber_delta), axis=-1)
        metrics = {"action_loss": jnp.mean(action_loss)}
        action_mse = action_loss
        if self.action_loss_huber_delta is not None:
            action_mse = jnp.mean(jnp.square(residual), axis=-1)
            metrics.update(action_mse=jnp.mean(action_mse),
                           action_robust_tail_fraction=jnp.mean(jnp.abs(residual) > self.action_loss_huber_delta))''')
    line = '                metrics[f"action_loss/{kind}"] = jnp.sum(jnp.where(mask[:, None], action_loss, 0)) / (n * action_loss.shape[1])'
    replace(model, line, line + '''
                if self.action_loss_huber_delta is not None:
                    metrics[f"action_mse/{kind}"] = jnp.sum(jnp.where(mask[:, None], action_mse, 0)) / (n * action_mse.shape[1])''')
    trainer = DEST / 'openpi/scripts/b1k/train_b1k.py'
    replace(trainer, 'from openpi.shared import learning_rate_contract',
            'from openpi.shared import learning_rate_contract\nfrom openpi.shared import action_loss as _action_loss')
    marker = '    data_loader.formal_run_contract = None if run_contract_path is None else json.loads(epath.Path(run_contract_path).read_text())'
    replace(trainer, marker, marker + '''
    loss_delta = config.model.action_loss_huber_delta
    if loss_delta is not None or (data_loader.formal_run_contract is not None and "action_loss" in data_loader.formal_run_contract):
        if data_loader.formal_run_contract is None or data_loader.formal_run_contract.get("action_loss") != _action_loss.contract(loss_delta):
            raise ValueError("Action loss differs from the frozen formal contract")''')
    argv_path = DEST / 'inputs/compiled/training_argv.json'
    argv = json.loads(argv_path.read_text())
    for flag, value in {'--exp-name': RUN, '--num-train-steps': '1000'}.items():
        argv['argv'][argv['argv'].index(flag) + 1] = value
    argv['argv'] += ['--model.action-loss-huber-delta', str(DELTA)]
    argv['launch_authorized'] = False
    save(argv_path, argv)
    identity_path = DEST / 'inputs/formal_run_contract.json'
    identity = json.loads(identity_path.read_text())
    identity.update(run_id=RUN, action_loss={'schema_version': 1, 'kind': 'scaled_huber', 'delta': DELTA})
    save(identity_path, identity)
    job_path = DEST / 'nebula/worker_job.json'
    job = json.loads(job_path.read_text())
    job.update(run_id=RUN, output='/data/oss_bucket_0/wangfaye/result/behavior-challenge/expert_huber/' + RUN,
               experiment='expert_only_huber5_control', resume=False, formal_training_authorized=False)
    save(job_path, job)
    auth_path = DEST / 'inputs/formal_authorization.json'
    authorization = json.loads(auth_path.read_text())
    authorization.update(formal_training_allowed=False, kind='prepared_robust_loss_candidate_pending_selection',
                         action_loss=identity['action_loss'], rationale='Bound extreme residual gradients while retaining squared error for small residuals.')
    save(auth_path, authorization)
    manifest = {str(path.relative_to(DEST)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in DEST.rglob('*') if path.is_file() and path.name != 'bundle_manifest.json' and '__pycache__' not in path.parts}
    changed = sorted(name for name, value in manifest.items() if old_manifest.get(name) != value)
    expected = sorted(['openpi/src/openpi/shared/action_loss.py', 'openpi/src/openpi/models/pi0.py',
        'openpi/src/openpi/models/pi0_config.py', 'openpi/scripts/b1k/train_b1k.py',
        'inputs/compiled/training_argv.json', 'inputs/formal_run_contract.json',
        'inputs/formal_authorization.json', 'nebula/worker_job.json'])
    if changed != expected:
        raise ValueError('Unexpected change in the controlled experiment')
    save(DEST / 'bundle_manifest.json', manifest)
    save(DEST.parent / 'preparation.json', {'run_id': RUN, 'delta': DELTA, 'updates': 1000,
        'execution_enabled': False, 'cloud_jobs_created': 0, 'changed_files': changed,
        'same_sampling_lr_optimizer_seed_batch_normalization': True,
        'source_manifest_sha256': hashlib.sha256((SOURCE / 'bundle_manifest.json').read_bytes()).hexdigest(),
        'candidate_manifest_sha256': hashlib.sha256((DEST / 'bundle_manifest.json').read_bytes()).hexdigest(),
        'runtime_model_forward_tested': False,
        'remaining': 'CPU JAX gradient and loss-integration validation; select after complete current evaluation.'})
    print(json.dumps({'prepared': True, 'execution_enabled': False, 'changed_files': changed}))


if __name__ == '__main__':
    main()
