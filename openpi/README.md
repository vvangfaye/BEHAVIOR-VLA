# OpenPI for BEHAVIOR π0.5

This directory contains OpenPI and the BEHAVIOR R1Pro integration used for
π0.5 training, joint subtask language/action training, and inference.

Start with the [workspace guide](../README.md) and the
[subtask training and inference guide](../docs/subtask_training.md).
The workspace launchers configure data validation, normalization statistics,
checkpoint paths, and subtask options.

| Component | Entry point |
| --- | --- |
| BEHAVIOR training | [scripts/b1k/train_b1k.py](scripts/b1k/train_b1k.py) |
| Subtask data validation | [scripts/b1k/check_subtask_data.py](scripts/b1k/check_subtask_data.py) |
| Distributed preflight | [scripts/b1k/distributed_preflight.py](scripts/b1k/distributed_preflight.py) |
| Policy server | [scripts/b1k/serve_b1k.py](scripts/b1k/serve_b1k.py) |
| Normalization statistics | [scripts/compute_norm_stats.py](scripts/compute_norm_stats.py) |
| BEHAVIOR client example | [test_b1k_client.py](examples/simple_client/test_b1k_client.py) |
| JAX to PyTorch conversion | [convert_jax_model_to_pytorch.py](examples/convert_jax_model_to_pytorch.py) |

Shared model implementations, configuration classes, client libraries, and tests
remain compatible with OpenPI. Standalone ALOHA, DROID, LIBERO, and UR5 demos are
outside this workspace's scope and have been removed. Consult the
[upstream repository](https://github.com/Physical-Intelligence/openpi) for those examples.

OpenPI originates from [Physical Intelligence](https://github.com/Physical-Intelligence/openpi).
The repository [license](LICENSE) and [Gemma license](LICENSE_GEMMA.txt) are retained.
