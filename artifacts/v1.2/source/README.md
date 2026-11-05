# AgentCC V22 · 1.2

读取本次任务与有向拓扑，构造有效调度，再由官方 LLM 提出具体传输组的分流数量，编译成实际路径编辑，完整回放后保留更快的有效方案。当前实现面向 `agentcc-256-20261101-v22`，不按实例名称套用答案。

## 项目结构

- `src/server.py`：正式入口，执行官方 `AgentRuntime(MyAgent()).run()`。
- `src/my_agent.py`、`src/service.py`：官方 SDK 适配和求解流程。
- `src/config.py`、`src/llm.py`、`src/planning.py`：模型配置、当前输入摘要、具体数量决策与审计证据。
- `src/schedule_edit.py`：瓶颈目录、合法备用路径、具体编辑编译与完整校验。
- `src/route_broad.py`：保留已采纳的模型动作，继续改进其余传输的路径。
- `src/topology.py`：方向、重复边、共享资源解析及合法路径搜索。
- `src/solver.py`：先验证首解，再进行限时候选搜索，保留已验证答案。
- `src/collectives.py`：反向倍增 AllGather、XOR 多根 AllReduce、实际邻接带宽排序。
- `src/route_balance.py`：按每轮真实边与共享资源负载改进路径。
- `src/replay.py`：独立的符号状态与通信成本校验。
- `src/cli.py`：本地求解、回放和公开样例批量测试。
- `tests/`：正确性、输入变更、官方运行时及端到端测试。
- `materials/`：用户提供的原始资料；`materials/spec/` 为公开规范和样例。
- `vendor/`：从官方基础镜像只读提取的 SDK，仅供宿主机测试，不修改、不打包。
- `artifacts/`：测试响应、模型证据和汇总，不打包。

## 模型如何参与决策

每次请求重新解析全部实际边、方向、带宽、资源成员和任务约束，先完成受保护的有效初解。从当前方案提取具体瓶颈、相关容量、合法备用路径和互不重叠的传输组。每组可移动的是已有完整分片；AllReduce 的整个贡献表达式不可拆开。

模型收到各轮真实负载及每组动作对共享资源忙碌时间的影响，返回 `{"base":"当前基线标识","counts":[数量,数量]}`。数组与目录中的组一一对应，客户端只无损转换为带组编号的动作，不补齐、不裁剪、不替模型优化数量。模型实际决定特定轮次中的哪些分片改走备用路径，已经超出仅选择算法名称或路径权重。

模型网络等待期间，确定性路径改进独立生成对照方案。模型提案只应用于它看到的冻结基线；重新核对任务、拓扑和基线哈希后，联合编译全部编辑并完整 Replay。合法提案中，仅保留严格改善的完整轮次，其他轮次恢复基线；再逐轮与对照比较，组合后重新完整验证。已选中的模型移动前缀在后续路径改进中固定，其余传输仍可优化，因此不保证最终整组的分流比例等于模型最初提议的比例。

`output.answer` 只包含完整 solution。`output.diagnostics.llm` 保留真实请求、原始响应、解码后的数量和 Token 用量，不含 Key。`search.model_edit` 保留完整路径编辑清单、输入哈希和各阶段耗时，区分 `accepted`（至少一轮优于模型看到的基线）、`selected`（至少一轮还胜过确定性对照）和 `preserved_in_final`（选中移动前缀保留到最终方案）。`selected_edits` 是实际选中的子集，不把未提交的提案记为最终收益。SDK 记录真实调用次数。

无变化、非法或变慢的模型编辑保留拒绝原因，返回已验证方案；网络、认证或模型服务故障明确失败。`--offline` 只用于显式算法测试。旧的参数规划客户端保留用于对照和兼容，生产流程调用具体编辑接口。

[THEORY.md](THEORY.md) 给出官方容量公式、原子数据守恒、路径合法性及本地通信耗时不变差的条件。当前不训练神经网络，不使用 GPU；借鉴 SmartCCL 的候选与集中校正思路，不把交换机带宽平均分配给各边。LLM 的上下文整合能力是设计动机，是否优于确定性优化须由公平实验回答。

## 1.2 的计算优化

已构造候选先按精确的逐轮路径、边和资源成本进行筛选。成本已经不优于有效方案时，可以省去完整的符号语义回放；这种记录明确标为 `valid:null, cost_only:true`，不冒充已验证。首解、模型指定的旧式参数候选和所有可能胜出的候选仍做完整 Replay。最终模型编辑和路径改进也必须完成全方案验证。

1.1 源码、测试及文档保存在 `artifacts/baseline-1.1/`，生产文件哈希与原 1.1 验证清单逐项一致；没有复制 `.env`。本轮比较与限制见 [VALIDATION.md](VALIDATION.md)。

## 集合通信构造

AllGather 先跨较大的局部性分区交换原始数据，再向局部扩散，减少共享出口上的重复流量。AllReduce 按实际图建立层次顺序，用 XOR 排列把不同槽位的规约根分散到多个 rank，保持局部通信关系。两种算法都只使用当前输入能到达的路径；非二次幂和需要 rank 中继的图仍有通用后备构造。

同轮的大传输可以把已有原子分片分为两组；没有制造更小分片，也没有减少边和资源计费字节。候选确定后，可在短预算内重新分配有用的备选路径，每轮只有真实成本降低才采纳，最后完整回放。路径搜索使用整数索引、共享每源搜索结果，并对重复的参数选择做探测剪枝。

阅读资料包括 [TACCL](https://arxiv.org/abs/2111.04867)、[Blink](https://arxiv.org/abs/1910.04940) 与 [TACOS](https://arxiv.org/abs/2304.05301)。本项目的具体算法、计时和取舍以公开 AgentCC 规则及本地对照实验为依据。

## 配置与运行

填写项目 `.env` 的 `PLATFORM_LLM_URL`、`PLATFORM_API_KEY`、`PLATFORM_LLM_MODEL`。当前客户端使用 OpenAI 兼容的 Chat Completions 格式。进程环境变量优先于 `.env`。

在已有官方基础环境中，从 `/app` 执行：

```bash
python -m src.server
```

这会由官方 SDK 提供 `0.0.0.0:8080` 的 `/health` 和 `/solve`，没有另写替代 HTTP 服务。开发 Bash 容器的启动不等于服务已经启动。

HTTP 请求格式：

```json
{
  "task_id": "与task.task_id一致",
  "task_type": "agentcc",
  "input": {
    "challenge": "agentcc",
    "schema_version": "agentcc.joint-test.v1",
    "instance_id": "与task.instance_id一致",
    "task": {},
    "topology": {}
  },
  "constraints": {"timeout_ms": 180000}
}
```

`task` 与 `topology` 必须放完整实际内容。SDK 未收到 `timeout_ms` 时默认 30 秒。程序默认模型请求上限 15 秒；构造搜索软预算 3 秒，编辑目录软预算 0.5 秒，模型等待期间的对照改路预算 0.25 秒，保留模型动作后的路径改进预算 0.25 秒。首解保护、完整回放和单次路径计算可跨过软优化截止，因此这些预算不是整个请求的硬上限。最终以平台从请求到收齐响应的计时为准。

## 复现测试

在项目根目录执行。宿主机使用 `vendor/` 中未经修改的官方 SDK；基础容器里可去掉 `PYTHONPATH=vendor:.`。

```bash
PYTHONPATH=vendor:. python3 -m unittest discover -s tests -v
```

仅测试算法，不调用模型：

```bash
python3 -m src.cli benchmark --offline --budget 3 --output-dir artifacts/offline
```

真实调用官方模型，通过官方 SDK 原始 ASGI 应用完成 14 个公开大例、2 个小例和 1 个容量变化用例。此命令消耗模型配额：

```bash
PYTHONPATH=vendor:. python3 -m tests.runtime_benchmark \
  --include-mutation --single-cpu --memory-mib 1024 \
  --output-dir artifacts/runtime-v1.2-retest
```

测试使用 SDK 的实际请求解析、`/solve` 回调、超时和遥测逻辑，通过进程内 HTTP 传输避免端口占用；这不是替代服务。结果在 `artifacts/runtime-v1.2-retest/summary.json`，各完整响应包含模型请求、候选及验证证据。

比较归档 1.1 和当前代码，交替执行顺序，使用同一接口与输入；另在相同冻结基线和编辑目录上对照真实模型与确定性数量搜索。这两条命令也消耗模型配额：

```bash
PYTHONPATH=vendor:. python3 -m tests.paired_runtime_v12 \
  --all-public --repeat 1 --output-dir artifacts/v1.2/paired-retest
PYTHONPATH=vendor:. python3 -m tests.live_edit_ablation \
  --output-dir artifacts/v1.2/ablation-retest
```

完整流程的 42 个输入扰动回归使用明确标记的模拟模型响应，不联网：

```bash
PYTHONPATH=vendor:. python3 -m tests.mutation_benchmark_v12 \
  --output artifacts/mutations-v1.2-retest.json
```

验证保存的方案：

```bash
python3 -m src.cli replay \
  --task materials/spec/examples/256-rank/cloudmatrix/ag/task.json \
  --topology materials/spec/examples/256-rank/cloudmatrix/ag/topology.json \
  --solution artifacts/runtime-v1.2-retest/cloudmatrix_ag.json
```

## 手动构建与提交

以下命令由用户执行，开发过程中没有执行任何 Docker 命令。基于已经导入的官方镜像构建新的参赛镜像，不直接提交基础镜像或依赖宿主机挂载的开发容器。

```bash
cd /home/czy/agentcc
docker build --platform linux/amd64 -t agentcc-team:1.2 .
docker run --rm --name agentcc-check \
  --cpus=1 --memory=1024m --pids-limit=256 \
  --env-file /home/czy/agentcc/.env \
  -p 8080:8080 agentcc-team:1.2
```

若现有开发容器占用了 8080，测试时改用 `-p 8081:8080`。`.env` 不打包，平台运行时也必须提供模型地址、Key 和模型名这三项配置；实际平台是否自动注入及注入哪些字段，当前材料没有明确说明。不能把“镜像成功构建”当作“平台已获得模型配置”。

验证完后导出。镜像归档统一保存在 `/home/czy/agentcc-images`，后续版本也使用此目录：

```bash
docker save -o /home/czy/agentcc-images/agentcc-team-1.2.tar agentcc-team:1.2
```

Dockerfile 只复制 `src/`，继承官方依赖，使用非 root 的 `agent` 用户，默认执行 `python -m src.server`。测试输出、样例、SDK 副本和 Key 均不会进入镜像。

## 验证边界

本地 Replay 实现公开规则，使用公开完整小例和非法调度测试校验。公开 14 个大例不是正式冻结题库；材料未附官方 Judge 可执行程序及正式 `L/U/t_good`，因此本地通过不等于官方通过，也不能换算正式得分。

最终宿主机测试将测试进程绑定到一个允许的 CPU，并将地址空间限制为 1024 MiB；这与 Docker 的 CPU 配额、内存 cgroup 和 256 pids 限制并不完全相同。未执行 Docker，提交前可用上述命令在容器中复测。当前路径输出使用规范允许的显式 `path`；通信时间在 Replay 报告中为微秒，在 solution.metrics 中为秒。

完整对照结果、异常调查和验证范围见 [VALIDATION.md](VALIDATION.md)。历史 1.0、1.1 代码分别保存在 `artifacts/baseline-1.0/` 和 `artifacts/baseline-1.1/`。
