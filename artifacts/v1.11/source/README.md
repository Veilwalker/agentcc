# AgentCC 1.11

本快照以 1.6 的调度流程为基线，经 1.9 的 C++ 容量内核，再加入保持原选择规则的 C++ 最短路径计算。模型仍参与具体阶段宽度和重定时参数决策，独立 Replay 验证最终方案。没有采用 1.10/1.12 的新代码生成流程，因此本版主要作为保持旧流程的对照和回退。

`src/server.py` 调用官方 `AgentRuntime(MyAgent()).run()`。Python 负责 SDK、编排与 Replay；`native/capacity.cpp`、`native/shortest_paths.h` 负责数值计算，不需要 GPU。镜像运行时强制加载 C++，不静默回退。

## 配置

使用官方基础环境，按 `.env.example` 配置 `PLATFORM_LLM_URL`、`PLATFORM_API_KEY` 和 `PLATFORM_LLM_MODEL`。真实 Key 不进入源码和镜像。开发时先运行 `python native/build.py`，再执行 `python -m src.server`。评测提供本次完整任务和拓扑，不能按名称套固定答案。

## 构建

```bash
docker build --platform linux/amd64 -t agentcc-team:1.11 /home/czy/agentcc/artifacts/v1.11/source
docker save -o /home/czy/agentcc-images/agentcc-team-1.11.tar agentcc-team:1.11
```

用户只授权了以上两类 Docker 操作。镜像最终以非 root 用户运行，入口是官方 SDK，不自建同名 HTTP 服务。

## 验证

详见 `VALIDATION.md`。完整项目中保留公开材料、官方 SDK、历史版本与原始实验；源码包不复制这些依赖。独立源码环境可运行不依赖材料的核心测试：

```bash
python native/build.py
AGENTCC_CAPACITY_BACKEND=cpp python -m unittest tests.test_solver tests.test_native_capacity tests.test_native_routing -v
```

所有版本的可视化与原始对照在完整项目的 `artifacts/index.html`。1.11 不代表已经在平台测得提分。
