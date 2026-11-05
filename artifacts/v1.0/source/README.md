# AgentCC V22

读取每次请求中的任务与有向拓扑，调用官方 LLM 提出参数，构造候选调度，按公开规则独立回放后返回最好有效方案。当前实现面向 `agentcc-256-20261101-v22`，不按实例名称套用答案。

## 项目结构

- `src/server.py`：正式入口，执行官方 `AgentRuntime(MyAgent()).run()`。
- `src/my_agent.py`、`src/service.py`：官方 SDK 适配和求解流程。
- `src/config.py`、`src/llm.py`：模型配置、调用、策略校验与审计证据。
- `src/topology.py`：方向、重复边、共享资源解析及合法路径搜索。
- `src/solver.py`：AllGather / AllReduce 调度构造和有限候选搜索。
- `src/replay.py`：独立的符号状态与通信成本校验。
- `src/cli.py`：本地求解、回放和公开样例批量测试。
- `tests/`：正确性、输入变更、官方运行时及端到端测试。
- `materials/`：用户提供的原始资料；`materials/spec/` 为公开规范和样例。
- `vendor/`：从官方基础镜像只读提取的 SDK，仅供宿主机测试，不修改、不打包。
- `artifacts/`：测试响应、模型证据和汇总，不打包。

## 模型如何参与决策

每次请求重新解析全部边、方向、带宽、资源成员和任务约束。程序向模型发送当前任务、拓扑派生摘要（方向统计、边类型、邻接分组、瓶颈边及资源成员样本等），而不是仅发送拓扑名称。完整输入的 SHA-256 关联本次请求，摘要明确标记采样范围。

模型选择算法、切分数、共享资源路径权重和 rank 排列。模型建议会首先构造并回放，之后用其参数生成改进候选，与通用基线比较。非法建议不会作为代码或路径执行；过大的 AllGather 切分可能为控制符号输出体积而投影到任务允许的较小值，调整理由保留在证据中。最后选取本地 Replay 最快的有效候选。

响应的 `output.answer` 只包含 solution 字段；`output.diagnostics` 记录实际模型请求、响应策略、输入摘要、各候选参数、约束校验、时间和最终选择。SDK 的 LLM 追踪记录真实调用次数和返回的 Token 用量。

模型失败会明确返回失败，不把纯本地结果伪装成成功的 LLM Agent。`--offline` 只用于显式的算法测试。

任务结构、切分对齐和 rank 集合等在调用模型前校验。模型首候选最多占用约一半构造预算；若构造超时或无效，会尝试保留模型算法、rank 排列和资源权重的较小切分候选，并记录恢复原因，给有效方案留出时间。

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

`task` 与 `topology` 必须放完整实际内容。SDK 未收到 `timeout_ms` 时默认 30 秒。程序默认模型请求上限 15 秒、候选搜索预算 8 秒；搜索预算不含输入解析、模型调用、最终校验和响应序列化。最终以平台从请求到收齐响应的计时为准。

## 复现测试

在项目根目录执行。宿主机使用 `vendor/` 中未经修改的官方 SDK；基础容器里可去掉 `PYTHONPATH=vendor:.`。

```bash
PYTHONPATH=vendor:. python3 -m unittest discover -s tests -v
```

仅测试算法，不调用模型：

```bash
python3 -m src.cli benchmark --offline --output-dir artifacts/offline
```

真实调用官方模型，通过官方 SDK 原始 ASGI 应用完成 14 个公开大例、2 个小例和 1 个容量变化用例。此命令消耗模型配额：

```bash
PYTHONPATH=vendor:. python3 -m tests.runtime_benchmark --include-mutation
```

测试使用 SDK 的实际请求解析、`/solve` 回调、超时和遥测逻辑，通过进程内 HTTP 传输避免端口占用；这不是替代服务。结果在 `artifacts/runtime/summary.json`，各完整响应包含模型请求、候选及验证证据。

验证保存的方案：

```bash
python3 -m src.cli replay \
  --task materials/spec/examples/256-rank/cloudmatrix/ag/task.json \
  --topology materials/spec/examples/256-rank/cloudmatrix/ag/topology.json \
  --solution artifacts/runtime/cloudmatrix_ag.json
```

## 手动构建与提交

以下命令由用户执行，开发过程中没有执行任何 Docker 命令。基于已经导入的官方镜像构建新的参赛镜像，不直接提交基础镜像或依赖宿主机挂载的开发容器。

```bash
cd /home/czy/agentcc
docker build --platform linux/amd64 -t agentcc-team:1.0 .
docker run --rm --name agentcc-check \
  --cpus=1 --memory=1024m --pids-limit=256 \
  --env-file /home/czy/agentcc/.env \
  -p 8080:8080 agentcc-team:1.0
```

若现有开发容器占用了 8080，测试时改用 `-p 8081:8080`。`.env` 不打包，平台运行时也必须提供模型地址、Key 和模型名这三项配置；实际平台是否自动注入及注入哪些字段，当前材料没有明确说明。不能把“镜像成功构建”当作“平台已获得模型配置”。

验证完后导出。镜像归档统一保存在 `/home/czy/agentcc-images`，后续版本也使用此目录：

```bash
docker save -o /home/czy/agentcc-images/agentcc-team-1.0.tar agentcc-team:1.0
```

Dockerfile 只复制 `src/`，继承官方依赖，使用非 root 的 `agent` 用户，默认执行 `python -m src.server`。测试输出、样例、SDK 副本和 Key 均不会进入镜像。

## 验证边界

本地 Replay 实现公开规则，使用公开完整小例和非法调度测试校验。公开 14 个大例不是正式冻结题库；材料未附官方 Judge 可执行程序及正式 `L/U/t_good`，因此本地通过不等于官方通过，也不能换算正式得分。

本次宿主机测试未通过 Docker 强制设置 1 CPU / 1024 MiB 限额，实测时间和峰值内存只反映测试环境；提交前可用上述限额命令复测。当前路径输出使用规范允许的显式 `path`；通信时间在 Replay 报告中为微秒，在 solution.metrics 中为秒。
