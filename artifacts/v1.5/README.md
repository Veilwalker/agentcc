> 历史版本：平台评测 94.2 分、14/14 有效、8,512 Token。已交付源码在本目录 `source/`，当前开发转至 [1.6](../v1.6/README.md)。下面保留 1.5 当时的研究记录；根目录文档现在对应 1.6。

# 1.5 · revision-05

主基准为已发布的 **1.4**，当前源码在项目[src](../../src/)，审阅快照在[source](source/)。先读[验证报告](../../VALIDATION.md)、[设计原理](../../THEORY.md)和[逐例交互对照](reports/revision-05/comparison/report.html)。

84次真实官方SDK请求全部有效。完整请求时间中位数的样例平均下降17.80%，CPU下降2.28%，每14例8,181～8,189 Token。通信12例改善、1例持平、Cloud AG回退0.86%；响应体增大，不能宣称逐项全面改进或保证平台涨分。

本轮将模型可编辑窗口从三个复制轮扩大到五个，明确半量/全量的容量收益；同时优化路径末跳查询和提示准备，减少重复计算。模型决定具体提前比例，完整程序经独立Replay改善后才采用。没有训练神经网络。

## 当前证据

| 入口 | 内容 |
| --- | --- |
| [source/](source/) | 当前源码、测试、文档与哈希清单 |
| [runs/revision-05/verified-sdk/](runs/revision-05/verified-sdk/) | 14例 × 1.4/当前1.5 × 三次真实请求 |
| [reports/revision-05/comparison/](reports/revision-05/comparison/) | 通信、完整请求、CPU、Token、输出大小逐例报告 |
| [reports/revision-05/validation.json](reports/revision-05/validation.json) | 发布检查与源码一致性 |
| [reports/revision-05/unit-candidate.log](reports/revision-05/unit-candidate.log) | 406项自动测试 |
| [analysis/revision-05/fixed-replies.json](analysis/revision-05/fixed-replies.json) | 真实回复因果审计与有限目录243组合 |
| [analysis/revision-05/stability.json](analysis/revision-05/stability.json) | 真实模型回复、调度哈希及延迟分解 |
| [analysis/revision-05/mutations.json](analysis/revision-05/mutations.json) | 42次合成回复拓扑扰动，明确不是实际模型测试 |
| [runs/revision-05/live-mutations/](runs/revision-05/live-mutations/) | 4次实际模型变更拓扑补测 |
| [analysis/revision-05/structural-sweep.json](analysis/revision-05/structural-sweep.json) | 476个不同窗口、提前距离及混合基数候选 |
| [analysis/revision-05/prototypes/](analysis/revision-05/prototypes/) | 未采用的多方案及联合成本原型，生产不导入 |
| [iterations/04/source/](iterations/04/source/) | 前一轮1.5原字节备份 |
| [iterations/03/source/](iterations/03/source/) | 更早1.5备份 |

## 研究记录怎么读

主对照之外的baseline-v14、wider-pilot、portfolio-pilot、two-plans-pilot、lazy-gain-pilot、compact-pilot与pair-cost-pilot均是探索，未混入84次发布统计。三方案实验中出现过模型格式错误和截断，即使回退后的最终方案有效，也不视为模型协议成功。

阶段合并对Clos有潜力，对Mesh/NV/Sparse常明显退化。双/三方案输出增加等待；邻接成本提示没有稳定收益。这些方向保留证据，未按拓扑名称编排最佳离线答案。有限目录审计显示模型尚未达到目录最优，详见验证报告。

历史实验保留原路径：[revision-04](reports/revision-04/comparison/report.html)、[revision-03](reports/revision-03-comparison/report.html)。此前较详细的容量优化研究导航保留于[iterations/04/research-navigation.md](iterations/04/research-navigation.md)，其中链接已按归档位置调整；最新入口以本页为准。

镜像版本仍为agentcc-team:1.5；新导出文件用agentcc-team-1.5-r05.tar。1.5.5仅是本轮联合成本原型的对照标签，不是另一个发布版本。官方运行时、非root及配置用法见[项目README](../../README.md)。本次没有执行Docker或导出镜像。
