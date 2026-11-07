# AgentCC 1.11 · 保持 1.6 调度流程的加速对照

本版继承 1.9（1.6 调度流程 + C++ 容量内核），再将最短路径计算迁入 C++。保留原候选顺序、路径平局规则、模型协议和独立 Replay；不是 1.10 的通信结构搜索版本。

- [发布源码](source/) · [源码包](agentcc-source-1.11.tar.gz)
- [固定回答完整请求](runs/path-fixed/summary.json)：56/56 有效，所有提示与最终调度一致，平均请求 2.011 → 1.906 秒。
- [真实模型完整请求](runs/path-live/summary.json)：1.9 与 1.11，每版 14 例各两次；原始响应与阶段时间均保留。
- [1.9 导出镜像审计](analysis/images/image-1.9/audit.json)：同目录还复核了 1.6、1.10 的实际镜像，而非只检查开发目录。
- [同源码后端隔离](runs/backend-live/summary.json)：1.6 Python、1.9 Python、1.9 C++，84 次真实请求。
- [官方遥测刷新实验](analysis/telemetry-timing.json)：未修改 SDK，演示完整响应计时与 SDK 内部时长的差异；不代表已经证明平台延迟来源。
- [本轮综合验证](../../VALIDATION.md) · [1.12 结构优化版](../v1.12/README.md)

1.9 同源码对照中，C++ 的 CPU 和完整请求均较 Python 少，未复现平台十秒级退步。平台没有提供阶段日志，不能认定 C++ 或服务器必然是根因。1.11 主要作为保持旧流程的可核验对照；旧模型参数协议对新过程评分的适配，弱于 1.12 的代码生成流程。

[完整快照回归](reports/unit-tests-release.log)：430 项中 427 项通过、3 项历史证据检查明确跳过。

[实际导出镜像核验](../v1.12/runs/image-sdk-1.11/summary.json)：镜像内应用和测试源码各跑14例，28/28有效；提示、最终调度均一致，均使用C++。使用已记录模型回答，无新联网调用。

## 构建与导出

```bash
docker build --platform linux/amd64 -t agentcc-team:1.11 /home/czy/agentcc/artifacts/v1.11/source
docker save -o /home/czy/agentcc-images/agentcc-team-1.11.tar agentcc-team:1.11
```

入口仍是官方 Python AgentRuntime。扩展在基础镜像中编译，运行时要求 C++ 成功加载。配置和真实 Key 不进入源码包或镜像。是否已完成导出，以 `reports/image.json` 为准。

最终四版同场对照：每版84次，1.11完整请求算术平均3.271秒，1.6为3.432秒；逐例中位数平均3.069秒和3.252秒。实际模型回答允许变化，固定回答结果另列。[完整验证](../../VALIDATION.md)与[对照看板](../v1.12/reports/release-comparison/report.html)同时包含退化项及慢请求。
