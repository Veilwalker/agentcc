# AgentCC 集合通信调度

本项目是 Made in Token !!! 队为 AgentCC 初赛实现的集合通信调度 Agent。给定一项 AllGather 或 AllReduce 任务，Agent 根据拓扑中的有向链路、交换机、链路带宽和共享资源池生成调度，确定每个通信轮次中分片的发送方、接收方和传输路径。初赛正式评测中，14 个实例全部通过 Replay，自动得分为 96.8257 / 100。

## 调度方法

1. **构造可行方案。** 从任务读取 rank、合法切分粒度和非空轮数上限，从拓扑建立有向路径与共享资源索引。AllGather 逐步复制各 rank 的原始分片。AllReduce 按分片槽位归约不同 rank 的贡献，再分发完整结果。
2. **选择优化方向。** Agent 根据当前任务构造模型输入。在适用的拓扑上，输入包括逐轮耗时、瓶颈资源和分片就绪情况。模型可以指定重点轮次与调整方向，或给出阶段组合策略。等待模型时，内置工具并行搜索候选方案。
3. **生成调度并回放。** 工具将候选展开为具体传输，检查轮首持有关系、归约贡献、路径、切分及轮数，并计算有向边与共享池的负载。Agent 根据整场通信耗时保留通过校验的方案。Python 实现协议和调度逻辑，C++17 扩展加速路径计价、候选比较与回放。

## 项目结构

| 路径 | 内容 |
| --- | --- |
| `src/` | 服务入口、模型交互、调度构造与回放。`my_agent.py` 对接比赛 SDK，`generated_service.py` 编排求解，`replay.py` 校验方案。 |
| `native/` | C++17 计算内核，以及为当前 Python 解释器编译扩展的 `build.py`。 |
| `tests/` | 调度正确性、资源约束、模型交互与服务协议的测试。 |
| `examples/` | 可直接发送给服务的双 rank AllGather 请求，含一个交换机和两个共享带宽池。 |
| `artifacts/v*/source/` | 各初赛版本的源码、测试及构建文件，包含对应的 SHA-256 文件清单。 |
| `Dockerfile` | 基于比赛镜像构建 Linux/AMD64 服务，以非 root 用户 `agent` 运行。 |

## 运行比赛服务

构建需要 Docker 和比赛基础镜像 `agenthack/base:1.1.1`。若基础镜像以归档交付，先用 `docker load -i <基础镜像.tar>` 导入。构建阶段使用系统包源安装 C++ 编译器。

```sh
docker build --platform linux/amd64 -t agentcc:preliminary .
cp .env.example .env
```

在 `.env` 中填写模型服务地址 `PLATFORM_LLM_URL`、访问密钥 `PLATFORM_API_KEY` 和模型名称 `PLATFORM_LLM_MODEL`。服务地址可使用兼容接口的 `/v1` 基址或完整的 `/chat/completions` 地址。通过环境变量在启动时注入配置。

```sh
docker run --rm --name agentcc -p 127.0.0.1:8080:8080 --env-file .env agentcc:preliminary
```

服务启动后，在另一个终端检查健康状态并提交示例任务。

```sh
curl -fsS http://127.0.0.1:8080/health
curl -fsS -H 'Content-Type: application/json' \
  --data-binary @examples/allgather-request.json \
  http://127.0.0.1:8080/solve > solution.json
```

请求的 `input.task` 描述算子、数据量、切分和轮数约束，`input.topology` 描述真实资源。成功响应的 `output.answer` 给出分片、初始持有关系、最终分片、逐轮传输、路径及耗时指标。`output.diagnostics.local_replay` 提供本地回放结果，正式评测采用平台 Replay。

## 本地验证

在仓库根目录使用 Python 3.10 或 3.11，可直接运行以下单元测试。它们使用自带的小型输入，无需比赛 SDK 或模型密钥。

```sh
python3 -m unittest tests.test_replay tests.test_topology tests.test_solver
```

下面的示例通过与服务相同的求解入口关闭模型调用，生成调度后再独立回放。输出依次为合法性、非空轮数和整场通信耗时（微秒）。

```sh
python3 - <<'PY'
import json
from pathlib import Path
from src.generated_service import solve_input
from src.replay import replay

request = json.loads(Path("examples/allgather-request.json").read_text())
inputs = request["input"]
answer = solve_input(inputs, use_llm=False)["answer"]
report = replay(inputs["task"], inputs["topology"], answer)
print(report["valid"], report["nonempty_rounds"], report["total_time_us"])
PY
```

本地默认支持 Python 后端。如需使用与镜像相同的 C++ 后端，安装 g++ 和对应 Python 开发头文件后，运行 `python3 native/build.py`，并设定 `AGENTCC_CAPACITY_BACKEND=cpp`。运行 SDK 协议测试和官方样例测试时，可使用比赛提供的 SDK，并将官方样例置于 `materials/spec/examples/`。
