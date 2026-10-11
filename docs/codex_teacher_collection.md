# Codex 辅助完整任务采集

本实验用 Agent 辅助 `turning_on_radio`、`putting_shoes_on_rack` 的实例
311–320 采集成功轨迹，混合原训练演示后微调，再在 301–310 上完全不带
Agent 评测。用户已确认这两个用途允许，并指定当前 Codex / ChatGPT 账号、
`gpt-6.1-sol`、`medium`。当前明确选择 `~/.codex/auth_weizhi.json`，
不自动切换到另外两份账号。初始化权重固定为 1688977。

## 当前实施状态（2026-10-11）

采集、纠正、代理生命周期、训练和配对评测代码及相应 CPU 测试已完成。
集群已用指定镜像和 `auth_weizhi.json` 完成真实 ChatGPT 登录、私有 mihomo
启动及一次 Sol 请求（输入 9347、输出 15 token）。CLI 返回码为 0；三条启动
提示已逐条核对，新审计器在集群复核原事件日志通过，未观察到工具执行事件。
旧的无效 `tools.view_image` 配置已移除，保留对应 feature 禁用。
六份原始示范的 20 个数据文件已下载完整（3437280706 字节）；真实 HDF5 使用
`config.task` 和 `terminated/truncated`。新版按发布 metadata 的 raw ID、规范
路径和冻结 SHA 绑定实例身份，不把录制配置保留的初始实例 0 当真实实例号。
独立 `code-v6` 已安装，六份原始示范均已通过真实 LeRobot reader 和图像验证，
采集器 CPU 依赖预检通过。鞋架长视频沿用原训练 reader 已有的 5 毫秒容差。
源数据保持在 `original-v3`，最终示范和聚合验收记录位于 `original-reader-v4`。
首个 radio 311 尝试在场景加载后因策略客户端尚未连接、metadata 为 None 而失败，
未执行动作、未调用教练。`code-v7` 在场景前连接并复用同一客户端，保留明确失败
记录，并防止仿真退出码 0 掩盖异常。独立恢复尝试 `pilot-002` 已提交，作业
`behavior-codex-radio311-pilot-v5-1011-051558`，UID
`37bf17cc0ef849d0901f5a3a396b44e1`；原失败记录保留。该恢复尝试实际执行了 3339 步，17 次教练决定全部通过协议验收，但因 `CorrectionError` 中止，尚未获得成功轨迹。平台 SUCCEEDED 仅表示外层驱动正常结束，实际报告为 failed。此次教练实测 input 241631、cached input 0、output 3840（其中 reasoning 1964）token。独立 CPU 审计确认：最后一次右手向上 1cm 的纠正因 IK 线性残差超过门限而未执行；之前执行的是 11 步夹爪纠正，其余 3328 步来自 VLA。`code-v8` 保持原控制边界，在环境步进前拒绝不满足边界的 IK 目标，把数值诊断反馈给教练，并在同一物理步重新决策；所有重规划都计入原来的 32 次预算，硬性数据或布局错误仍中止。新版本已通过独立复核；重跑 `pilot-003` 作业 `behavior-codex-radio311-pilot003-v8-1011-055227`（UID `5e8f4727afd54cc6a0f9fa37831d2f21`）已提交到单卡 reserved RTX 5090，待真实轨迹验收。
上述 `pilot-003` 已跑满 4500 步、23 次教练调用，未成功；两个 IK 拒绝均在执行前反馈重规划。随后 radio 317 的 `collect-001` 执行 1792 步、8 次教练调用，在首次闭合夹爪时因接口错误停止：纠正模块错误拒绝了原生控制器本来会裁剪的有限夹爪输入。

当前使用独立 `code-v9` 的 `native_with_corrections` 模式，普通动作保留模型自己的子任务推理；夹爪修复采用原控制器的 `[-1,1]` 裁剪语义，不改变运动限值或旧轨迹。冻结包为 `collection_code_v9_r2.tar.gz`，SHA256 `46bb77c810fde0cbdd5779a7cf49658b2066540edad71238fc86726d885fa512`。首份未包含该修复的 v9 候选包从未安装或提交，保留原始字节。新一轮 radio 317 `native-001`（UID `e12ca5e2bad5461981f667e0231c5b61`）已在单卡 reserved 资源启动，等待真实结果。

尚无真实新增采集成功轨迹，尚未开始本实验
训练或新模型的配对评测。代码或合成测试通过不能作为这些阶段已经完成的证据。

早先在 Mac 上的 Sol 三张空白图连通测试仅证明当时的账号、模型、图片和 JSON
输出接口可用；它返回 abort，不是机器人轨迹，也不能证明集群联网可用。
该单次调用为 input 19507、cached input 0、output 84 token，不能据此推算真实
任务的平均消耗。实际尝试逐次记录 token，字段缺失时记为 null。

## 集群内的教练与代理

最终执行路径是 SenseCore worker 内的 Codex CLI。用户指定复用的镜像为：

```text
registry.cn-sh-01.sensecore.cn/ccr-zhicheng-04/wangfaye-base-20261010:wangfaye-debug-20261010154634
```

`sco/run_codex_collection.py` 的 `teacher_backend=codex_local` 在同一 worker
启动私有 mihomo 进程，然后运行 π0.5 服务与采集器。代理使用用户已有配置的
私有副本，绑定本机回环端口；源配置保持不变，结束时只清理本次拥有的代理和
子进程。Codex 使用用户授权的集群私有账号目录，借助独占 lease 避免并发更新
同一份登录状态。凭据、代理配置和订阅链接不进入普通日志、代码包或实验报告。
本次架构不需要 Mac HTTP 桥或 AOSS 教练邮箱中转；相应兼容入口仍保留在代码中。

教练每次先核验 CLI 的 ChatGPT 登录状态，固定请求 `gpt-6.1-sol` / `medium`，
不改账号全局默认模型，不回退到 API key。教练只通过已提供的观测和演示给出
结构化决定。CLI 单次调用关闭已知 shell、应用、插件、多 Agent、搜索等工具
入口，枚举配置中的 MCP 名称并逐个禁用，同时使用 read-only sandbox。
调用结束后审核完整 JSON 事件流；出现工具、失败或未知事件的响应不被采集器
接受。**这是运行前收窄工具能力加运行后验收，不是所有 CLI 版本“绝对零工具”
的证明，也不能把事后审核描述成已阻止执行过的工具。**

普通 ACP 批量工作沿用 Spot。用户最新授权允许先用标准 reserved 资源验证实验；
仅在核验本人当前标准资源占用和适用额度后用于有限验证，不扩展成所有后续
批次的默认配额。不修改或停止其他实验的作业。

## 决策、纠正与成功判定

一次尝试连续执行完整任务，不用回放、传送或拼接重置前后的动作。教练看到三路
当前 RGB、proprio、近期子任务和控制状态，以及带明确标记的原成功演示图、
上次观测图；历史图不会冒充当前观测。首个 radio 尝试使用原训练 episode 28：

```text
/mnt/afs/260010165/behavior_1k/codex_pilot_20261011/original-reader-v4/demonstrations/28/manifest.json
```

该 manifest 的图像必须经过原始成功记录和真实 LeRobot reader 验证后使用。
一个 manifest 最多提供六个示例，按当前子任务检索；当前接口不直接合并多份
manifest。原训练演示的实例范围为 0–300。

历史 `teacher_conditioned` 模式让教练提供短子任务，由 π0.5 按该标签生成动作。
当前 `native_with_corrections` 模式先执行一个原生动作区间，再观察；教练继续时
子任务字段必须为空，不覆盖模型标签或清除动作缓存。原生动作逐帧记录服务端
实际生成或缓存的子任务。教练仍可执行有限纠正或中止。纠正路径使用
R1Pro 当前末端姿态和 Jacobian 做单臂相对控制，每次 1–5 步，每步平移范数
最多 0.02m、旋转范数最多 0.10rad、关节增量最多 0.15rad；保持底座、躯干和
另一手臂，夹爪只能 keep/open/close。纠正结束立即再次询问教练，恢复策略时
清除旧动作缓存。记录实际执行的 23 维动作、执行前观测、动态标签和动作来源。
原生段和纠正段的标签来源分别保存，混合 HDF5 标记为 `mixed_native_and_teacher`；
导出和训练准备须使用支持该来源的冻结 `code-v9` reader，不能改成 `provided` 绕过验收。

成功必须同时具备环境 task success、最终目标 success 和 terminal 证据。
教练不能自行宣布成功。正样本训练只接收经过校验的完整成功轨迹；失败尝试仍
保留为实验记录。所有采集产物标记 `official_benchmark=false`；教练、采集和
轨迹校验入口均拒绝 301–310。

## 有界试验计划与入口

第一个尝试是 radio 311，最多 4500 控制步、32 次教练决定、7200 秒；正常
教练间隔 256 步，纠正结束可提前触发。单次教练调用上限 120 秒，worker
上层再设置 8400 秒的有限生命周期。首个尝试结束后先检查证据再扩展。

首轮先以两项任务各一条完整成功轨迹跑通训练和配对评测，再依据效果决定是否扩展到各五条。每项最多十次尝试，全实验最多 640 次
教练决定；鞋架采集每次最多 10000 控制步，后续 code-v8 鞋架尝试的正常观察间隔改为 320 步，32 次纯继续可覆盖 10240 步；纠正仍可能提前耗尽预算。radio 仍为 256 步。预算是上限，不保证成功或自动无限
重试。每次尝试使用独立输出目录，失败后保留现场。

推荐通过 `sco/run_codex_collection.py` 的冻结 payload 启动，它负责源码、
机器人配置、演示、checkpoint、环境、代理和进程期限的校验。采集器关键参数：

```text
--task-name turning_on_radio --instance-id 311 --attempt-id pilot-001
--checkpoint-id pi05_b1k_full100_joint_1688977
--teacher-codex --codex-binary /opt/container-tools/codex/current/bin/codex
--teacher-timeout 120 --teacher-interval 256
--max-decisions 32 --max-steps 4500 --max-seconds 7200 --step-timeout 120
```

还必须提供本次策略服务的 host/port/server token、独立输出目录、冻结的
`r1pro.yaml` 和上面的 `--demonstration-manifest`；不要脱离 worker 的私有代理
环境与账号目录直接运行。采集器默认值不代替这份显式试验预算。

产物包含 HDF5、逐次教练请求/响应/用量、动态子任务段、动作来源、视频和完成
或失败记录。混合导出、真实 reader 验收、4GPU smoke/训练及最终 receipt 见
[训练试验说明](codex_teacher_training_pilot.md)。301–310 使用原 whole-task
入口和模型自主生成的子任务，不加载教练或外部子任务；保留全部评测结果。
