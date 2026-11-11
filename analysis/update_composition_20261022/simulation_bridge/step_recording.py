"""Instrument the scalar adapter without modifying its wrapped vector env."""


def install_step_recorder(environment, record):
    original_step = environment.step

    def recording_step(*args, **kwargs):
        value = original_step(*args, **kwargs)
        record['reward'] = float(value[1])
        record['info'] = value[4]
        return value

    # SingleEnvironmentAdapter.__setattr__ forwards to the wrapped environment.
    # A regular assignment would replace the lower-level step that original_step
    # calls, recursively reentering the recorder.
    object.__setattr__(environment, 'step', recording_step)
