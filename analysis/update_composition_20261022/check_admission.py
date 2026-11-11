"""CPU regression checks for fail-closed admission to expensive training."""
from admission import verified_configuration, validate_trial_configuration


def main():
    plan = dict(checkpoint='/checkpoint', tokenizer='/tokenizer', precision='fp32',
                rank=4, seed=17, inner_lr=1e-4, inner_epsilon=1e-6)
    proof = dict(passed=True, adaptation_effective=True, worker_returncode=0,
                 verified_configuration=verified_configuration(plan))
    validate_trial_configuration(plan, proof)
    rejected = []
    for key, value in dict(precision='bf16', rank=8, seed=18,
                           checkpoint='/other', tokenizer='/other',
                           inner_lr=1e-3, inner_epsilon=1e-8,
                           inner_optimizer='sgd').items():
        try:
            validate_trial_configuration(dict(plan, **{key: value}), proof)
        except ValueError:
            rejected.append(key)
        else:
            raise AssertionError('Accepted unverified configuration: ' + key)
    for key, value in dict(passed=False, adaptation_effective=False,
                           worker_returncode=1, verified_configuration=None).items():
        try:
            validate_trial_configuration(plan, dict(proof, **{key: value}))
        except ValueError:
            rejected.append(key)
        else:
            raise AssertionError('Accepted invalid proof: ' + key)
    print({'passed': True, 'rejected_invalid_cases': rejected})


if __name__ == '__main__':
    main()
