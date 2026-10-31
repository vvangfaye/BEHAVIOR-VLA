"""Verify that the benchmark only adds an optional replay-validation hook."""


def original_training_source(current):
    edits = [
        ('def update_policy(model, optimizer, episodes, plan, round_index=1, publish=None, replay_validator=None):',
         'def update_policy(model, optimizer, episodes, plan, round_index=1, publish=None):'),
        ('            if replay_validator is None:\n'
         '                torch.testing.assert_close(actual,expected,atol=1e-4,rtol=1e-4)\n'
         '            else:\n'
         '                replay_validator(actual,expected)\n',
         '            torch.testing.assert_close(actual,expected,atol=1e-4,rtol=1e-4)\n'),
        ('    if replay_validator is not None:\n'
         '        replay_validator.finalize()\n', ''),
    ]
    for added, original in edits:
        assert current.count(added) == 1, 'Unexpected edits in training source'
        current = current.replace(added, original)
    return current
