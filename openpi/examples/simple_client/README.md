# 客户端示例的适用范围

BEHAVIOR 的当前闭环入口是仓库根目录的
[eval_b1k.sh](../../../scripts/eval_b1k.sh)。自行对接 B1K 服务时，按
[远程推理文档](../../docs/remote_inference.md)处理原始 R1Pro 观测、单步 `action` 和 reset。

本目录保留历史及上游示例：

- [main.py](main.py)生成 ALOHA / DROID / LIBERO 的随机观测，用于通用服务的延迟测量。
  虽然枚举里有 `B1K`，当前观测函数映射没有对应分支，不能直接用 `--env B1K` 跑 BEHAVIOR。
- [test_b1k_client.py](test_b1k_client.py)是旧 HDF5 演示脚本，引用已不存在的
  `OpenPIWrapper`，且包含固定主机和数据路径，不是当前可直接运行的 B1K 客户端。
- 本目录 Docker 配置启动的是 `main.py`，不是 OmniGibson 闭环评测。

通用客户端的 `reset()` 当前不发送消息；B1K episode 切换需要显式发送
`{"reset": True}`，且不要等待 reset 应答。完整示例见上面的协议文档。
