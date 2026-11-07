# AgentCC 1.12 · 扩大通信结构并减少重复计算

本版在 1.10 的通信结构空间上优化执行成本。入口仍为官方 Python AgentRuntime，数值容量和最短路径由 C++ 执行。与 1.11 的对照属于同语言、不同调度设计；与 1.10 的对照用于确认本轮执行优化。

## 主要变化

1. 保留 1.10 的宽阶段、整轮路由平衡、完整原子分流及独立 reduce/copy 阶段，让模型生成求解表达式，在当前实例容量表上执行。
2. 用容量预览代替提前构造大量备用分片；只对可能入选的候选展开完整调度，最终仍需独立 Replay。
3. 在模型返回前验证已达到有限搜索空间下界的备用方案，减少串行等待。这个下界只作审计，不能代替模型生成路径。
4. C++ 请求内容量图一次构建，复用边的系数，保留路径顺序、共享资源重复占用次数和数值计算顺序。AllReduce 只展开当前选中阶段需要的数据。
5. 明确模型表达式语法限制，保留最多一次有反馈的修正；模型输出不能直接修改工程或跳过验证。

## 对照与验证

- [完整验证说明](../../VALIDATION.md) · [最终逐例对照](reports/release-comparison/report.html)
- [最终真实 SDK 请求](runs/final-sdk/summary.json)：1.6、1.11、1.10、1.12，两批共336/336有效，14个公开输入，每版每例六次；[第二批](runs/repeat-sdk/summary.json)使用相同冻结源码。
- [固定回答的执行对照](runs/numeric-graph-fixed/summary.json)：28/28 有效，1.10 与 1.12 的 14 个最终调度完全相同；平均 CPU 1.872 → 1.610 秒。模型人工立即返回，这不是实际模型等待时间。
- [1.9 / 1.10 问题排查](../v1.11/analysis/diagnosis.md)
- [记录程序回放](reports/fixed-release-audit.json) · [拓扑变化回放](reports/mutation-release-audit.json) · [真实变化输入](runs/changed-final/summary.json)
- [官方 SDK 回归测试](reports/unit-tests-final.log)：457 项全部通过。
- [实际导出镜像核验](runs/image-sdk-1.12/summary.json)：提取镜像内的应用和扩展，与测试源码各跑14例，28/28有效、提示与最终调度全部一致，均使用C++；重放原始模型回答，无新联网调用。

### 未纳入生产的探索

下面是额外实验，不与最终发布结果混算。所有候选都检查正确性，效果不佳的方案保留原始证据。

| 方向 | 记录 | 不采用的原因 |
|---|---|---|
| 分阶段循环转发 | [7 例](analysis/cyclic-stages.json) | 多数没有改善；少数约 1% 收益伴随额外构造开销 |
| AllGather 更细切分 | [21 例](analysis/split-probe.json) | 通信改善通常不到 1%，CPU 增加，部分响应接近翻倍 |
| 放开单原子结构搜索 | [28 例](analysis/single-atom-probe.json) | 11 例没有改善，少数改善伴随约 0.2–0.4 秒开销 |

此前容量预览及等待重叠阶段的结果保存在 `runs/`，对应当时源码，不能与最终源码重复数据合并。`iterations/01/` 是这次继续优化前的研究快照，不是另一个可提交版本。

## 发布

[源码](source/) · [源码包](agentcc-source-1.12.tar.gz) · [镜像记录](reports/image.json)

```bash
docker build --platform linux/amd64 -t agentcc-team:1.12 /home/czy/agentcc/artifacts/v1.12/source
docker save -o /home/czy/agentcc-images/agentcc-team-1.12.tar agentcc-team:1.12
```

不含真实 Key、实例答案、SDK、训练模型或材料。没有代为上传官网。是否已经导出，以镜像记录为准；本地结果不等于官方得分。
