# OpenPI for BEHAVIOR π0.5（JAX）

本目录包含 π0.5 模型、BEHAVIOR R1Pro 数据适配、JAX 训练器和策略服务。
使用方式以[仓库 README](../README.md)为入口；joint、strict subtask-only 和
pure executor 的区别、预检及闭环参数见 [subtask 指南](../docs/subtask_training.md)。

## 代码入口

| 环节 | 文件 | 职责 |
| --- | --- | --- |
| 启动脚本 | [train_pi05_b1k.sh](../scripts/train_pi05_b1k.sh) | 环境、模式选择、预检、norm stats、训练参数 |
| 数据配置 | [config.py](src/openpi/training/config.py) | `pi05_b1k` / `LeRobotB1KDataConfig` |
| 数据读取 | [data_loader.py](src/openpi/training/data_loader.py)、[subtask_data.py](src/openpi/training/subtask_data.py) | 原始 episode、标签对齐、动作边界与采样 |
| 语言输入 | [tokenizer.py](src/openpi/models/tokenizer.py) | 普通 prompt、joint 高低层序列与文本监督 mask |
| JAX 模型 | [pi0.py](src/openpi/models/pi0.py)、[pi0_config.py](src/openpi/models/pi0_config.py) | π0.5 动作 loss、文本 CE 和采样 |
| JAX 训练 | [train_b1k.py](scripts/b1k/train_b1k.py) | 更新参数、验证、EMA、保存和恢复 |
| 数据预检 | [check_subtask_data.py](scripts/b1k/check_subtask_data.py) | episode / 段表 / token / 动作预算 |
| 多机预检 | [distributed_preflight.py](scripts/b1k/distributed_preflight.py) | 设备与数据分片检查 |
| 服务加载 | [serve_b1k.py](scripts/b1k/serve_b1k.py)、[policy_config.py](src/openpi/policies/policy_config.py) | 加载 checkpoint、norm stats 和条件模式 |
| Joint 推理 | [subtask_policy.py](src/openpi/policies/subtask_policy.py) | 生成、缓存或接收 subtask，然后采样动作 |
| 闭环控制 | [eval_b1k_wrapper.py](src/openpi/shared/eval_b1k_wrapper.py) | 动作缓存、逐步执行和 episode reset |

π0.5 的 JAX 实现就在 `models/pi0.py`，由 `pi05=True` 切换。
JAX 数据读取仍依赖 Torch DataLoader 和视频解码组件；阅读模型代码不需要转到
`models_pytorch/`，但环境中这些共享依赖仍需保留。

## 专题文档

- [B1K 集成与数据流](docs/b1k.md)：机器人字段、训练入口和源码定位。
- [归一化统计](docs/norm_stats.md)：统计空间、子集复用与 checkpoint assets。
- [B1K 远程推理](docs/remote_inference.md)：观测格式、单步动作与 reset 协议。
- [旧客户端示例说明](examples/simple_client/README.md)、[Docker 配置范围](docs/docker.md)。
- [已有 PyTorch 实现](../docs/pytorch_training.md)：保留参考，不作为当前 JAX 主线。

本目录源自 [Physical Intelligence / OpenPI](https://github.com/Physical-Intelligence/openpi)。
保留通用组件及部分上游示例，实际 BEHAVIOR 工作流以本仓库入口为准。
[OpenPI license](LICENSE) 与 [Gemma license](LICENSE_GEMMA.txt) 保留。
