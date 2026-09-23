# BEHAVIOR / JAX 远程推理

BEHAVIOR 使用 [serve_b1k.py](../scripts/b1k/serve_b1k.py) 和
[websocket_b1k_server.py](../src/openpi/serving/websocket_b1k_server.py)。
仿真端与策略端可以使用独立环境。完整闭环优先运行仓库根目录的
[eval_b1k.sh](../../scripts/eval_b1k.sh)，由它启动策略服务与 OmniGibson evaluator。

这里描述 B1K 的逐步控制协议。上游通用 `scripts/serve_policy.py` 的输入字段和
动作块接口不能直接套到 B1K 服务。

## 启动服务

从仓库根目录启动普通 π0.5 服务：

```bash
cd openpi
.venv/bin/python scripts/b1k/serve_b1k.py \
  --robot b1k/R1Pro --task b1k/carrying_in_groceries \
  --repo-id behavior-1k/2026-challenge-demos \
  --policy.config pi05_b1k --policy.dir /path/to/checkpoint/step \
  --action-horizon 16 --port 8000
```

`--policy.dir` 指向具体 checkpoint step，`--repo-id` 匹配其中的
`assets/<REPO_ID>/norm_stats.json`。此入口没有 `policy:checkpoint` 子命令。
模型配置默认预测 32 步；这里 `--action-horizon 16` 指每执行 16 步重新请求动作块。

| 使用方式 | 在基础服务命令上增加的参数 |
| --- | --- |
| 普通 π0.5 | 无；使用完整任务 prompt |
| Joint 自主生成 / 外部标签对比 | `--subtask-prediction --max-token-len 512 --max-subtask-len 64` |

Joint 模式读取 checkpoint 的 `assets/subtask_conditioning.json`。
新 `subtask_only` 默认每 2 次动作块请求刷新技能，旧 `task_and_subtask` 默认每次刷新；
可用 `--subtask-interval` 显式覆盖。这个间隔按动作块请求计数，不按 websocket 消息数计数。
token 长度必须匹配训练。外部 selector 和固定技能的闭环启动方法见
[subtask 指南](../../docs/subtask_training.md#joint-checkpoint--外部技能-selector)。

## 连接、观测与响应

连接建立后，服务先发送一条 metadata，客户端应先接收它，再开始发送观测。
消息使用支持 NumPy 的 MessagePack 编码，可使用
[openpi_client.msgpack_numpy](../packages/openpi-client/src/openpi_client/msgpack_numpy.py)。

单环境观测字段由 [R1Pro 配置](../src/openpi/configs/robots/b1k.py)确定：

| 字段 | 内容 |
| --- | --- |
| `robot_r1::proprio` | 原始 proprio 数组，包含配置所需全部索引；不是提取后的 23 维 state |
| `robot_r1::robot_r1:zed_link:Camera:0::rgb` | 头部 RGB，uint8 HWC |
| `robot_r1::robot_r1:left_realsense_link:Camera:0::rgb` | 左腕 RGB，uint8 HWC |
| `robot_r1::robot_r1:right_realsense_link:Camera:0::rgb` | 右腕 RGB，uint8 HWC |
| `prompt` | 可选，完整任务指令；缺省使用服务的任务注册文本 |
| `subtask` | 可选的外部技能文本，用于 joint 对比推理 |

相机也可带第 4 个通道，wrapper 取前三通道并缩放。归一化和动作反变换在服务端完成。
Joint 未传 `subtask` 时自主生成或复用缓存；传入时来源为 `provided`，不会将外部标签留作
后续自主生成的缓存。

每次观测对应一个控制步响应：

| 响应字段 | 含义 |
| --- | --- |
| `action` | NumPy 数组；proprio 为一维时返回 `(23,)`，带 batch 维时返回 `(B, 23)` |
| `subtasks` | 可选，按环境顺序排列的当前技能文本列表 |
| `subtask_sources` | 可选，对应来源：`generated` / `cached` / `provided` |
| `server_timing` | 本次请求耗时等信息 |

内部 `Policy.infer()` / `SubtaskPolicy.infer()` 返回的 `actions` 是完整动作块；
[B1KPolicyWrapper](../src/openpi/shared/eval_b1k_wrapper.py)保存它，并逐请求返回一个动作。
所以仿真客户端每个控制步发送当前观测、执行响应中的一个动作；不要在客户端又把
`action` 当成 32 步动作块执行。

## Episode reset

新 episode 开始前发送 `{"reset": True}`。服务清空动作和技能缓存，**不返回应答**，
客户端不应在这条消息之后等待一次 `recv()`。重新连接本身不会自动调用 policy reset。
通用 `WebsocketClientPolicy.reset()` 当前是空实现，不能用它代替 B1K 的 reset 消息。

下面只演示单环境通信循环；`read_observation()` 应返回上表格式，`apply_action()` 接收
一个 23 维动作，环境创建、终止判定和结果记录由调用方负责。正式评测使用现有 evaluator。
客户端环境需安装 `openpi-client` 及其依赖。

```python
import numpy as np
from openpi_client import msgpack_numpy
from websockets.sync.client import connect


def run_control_steps(read_observation, apply_action, steps, uri="ws://localhost:8000"):
    packer = msgpack_numpy.Packer()
    with connect(uri, compression=None, max_size=None) as ws:
        metadata = msgpack_numpy.unpackb(ws.recv())
        ws.send(packer.pack({"reset": True}))  # reset 无应答
        for _ in range(steps):
            ws.send(packer.pack(read_observation()))
            payload = ws.recv()
            if isinstance(payload, str):
                raise RuntimeError(payload)
            reply = msgpack_numpy.unpackb(payload)
            action = np.asarray(reply["action"])
            if action.shape == (1, 23):
                action = action[0]
            if action.shape != (23,):
                raise ValueError(f"Expected one R1Pro action, got {action.shape}")
            apply_action(action)
    return metadata
```

每个独立环境应使用独立的有状态 policy/server 实例，避免多个连接交错覆盖缓存。
现有 B1K 多环境 batch 要求 `subtask_interval=1`；单独实例才能安全使用更大的缓存间隔。
任务或外部技能变化会使相关缓存失效，但不能代替 episode reset。

当前推理只读取本次观测并缓存技能 / 待执行动作，没有历史观测记忆，也没有独立的
subtask 完成判别器。技能切换是否正确仍需闭环评测，协议连通测试不能衡量任务成功率。
