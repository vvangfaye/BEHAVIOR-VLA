# Docker 配置范围

本目录保留 OpenPI 通用服务的 Docker 配置，默认启动 `scripts/serve_policy.py`。
它不是 BEHAVIOR 训练或闭环评测的完整镜像：B1K 使用 `scripts/b1k/serve_b1k.py`，
还需要匹配的数据、checkpoint assets、仿真环境和资产挂载。

当前 BEHAVIOR 环境入口是仓库根目录的
[setup_envs.sh](../../scripts/setup_envs.sh)、[setup_eval_env.sh](../../scripts/setup_eval_env.sh)
和 [setup_eval_assets.sh](../../scripts/setup_eval_assets.sh)，可用
[pack_envs.sh](../../scripts/pack_envs.sh)打包。启动方法见[仓库 README](../../README.md)。
下文保留通用 Docker 配方；本次文档更新未构建或验证这些镜像。

## Generic OpenPI Docker setup

Run the following commands from `openpi/`. Example-specific Compose commands apply only to
directories that contain their own `compose.yml`; they do not launch the BEHAVIOR evaluator.

- Basic Docker installation instructions are [here](https://docs.docker.com/engine/install/).
- Docker must be installed in [rootless mode](https://docs.docker.com/engine/security/rootless/).
- To use your GPU you must also install the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- The version of docker installed with `snap` is incompatible with the NVIDIA container toolkit, preventing it from accessing `libnvidia-ml.so` ([issue](https://github.com/NVIDIA/nvidia-container-toolkit/issues/154)). The snap version can be uninstalled with `sudo snap remove docker`.
- Docker Desktop is also incompatible with the NVIDIA runtime ([issue](https://github.com/NVIDIA/nvidia-container-toolkit/issues/229)). Docker Desktop can be uninstalled with `sudo apt remove docker-desktop`.


If starting from scratch and your host machine is Ubuntu 22.04, you can use accomplish all of the above with the convenience scripts `scripts/docker/install_docker_ubuntu22.sh` and `scripts/docker/install_nvidia_container_toolkit.sh`.

Build the Docker image and start the container with the following command:
```bash
docker compose -f scripts/docker/compose.yml up --build
```

To build and run an example that has a Compose configuration, set its directory name:
```bash
EXAMPLE_NAME=simple_client
docker compose -f "examples/${EXAMPLE_NAME}/compose.yml" up --build
```
The `simple_client` example targets the generic server; see its
[scope and limitations](../examples/simple_client/README.md).

During the first run of any example, Docker will build the images. Go grab a coffee while this happens. Subsequent runs will be faster since the images are cached.
