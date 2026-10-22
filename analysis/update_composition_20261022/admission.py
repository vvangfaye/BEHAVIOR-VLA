"""Bind the expensive trial to the configuration actually verified on hardware."""

CONFIGURATION_KEYS = (
    'checkpoint', 'tokenizer', 'precision', 'rank', 'seed',
    'inner_optimizer', 'inner_lr', 'inner_epsilon',
)


def verified_configuration(plan):
    configuration = dict(plan)
    configuration.setdefault('inner_optimizer', 'adam')
    return {key: configuration[key] for key in CONFIGURATION_KEYS}


def validate_trial_configuration(plan, admission):
    if not (admission.get('passed') is True
            and admission.get('adaptation_effective') is True
            and admission.get('worker_returncode') == 0):
        raise ValueError('Real-model effective adaptation has not passed admission')
    verified = admission.get('verified_configuration')
    if not isinstance(verified, dict):
        raise ValueError('Admission does not bind the verified configuration')
    requested = verified_configuration(plan)
    mismatches = [key for key in CONFIGURATION_KEYS if requested[key] != verified.get(key)]
    if mismatches:
        raise ValueError('Trial differs from verified configuration: ' + ', '.join(mismatches))
