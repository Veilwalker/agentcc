# AgentCC 1.4 档案

返回 [版本看板](../index.html) · [目录规范](../README.md)

1.4 是当前本地候选，**暂无官方成绩**。正式对照入口为 [public-narrow 原始汇总](runs/public-narrow/summary.json) 和 [交互报告](reports/comparison/report.html)：14 个公开样例、两个版本、每例三次，共 84 次真实 SDK 请求有效。完整结果包含通信与请求时间退化项，不据此预测平台分数。

- [源码快照](source/README.md)：已冻结78个源码、测试及文档文件，含逐文件SHA清单。
- [当前验证报告](../../VALIDATION.md)：补充测试的完成范围和复现证据。
- [设计与理论边界](../../THEORY.md)：容量模型、模型具体编辑、约束校验及候选组合。
- [最终格式固定提示审计](analysis/stability/llm-final-fixed-input.json)：4 份 v3 提示、两种温度、各三次，共 24 次直接模型请求全部协议有效。温度 0 仍出现真实数量变化；原始回答全部保留，不等于 SDK 求解或 Replay 验证。
- [独立时延补测](reports/latency-followup/report.html)：20 次 SDK 请求中，ring AR 与 clos AG 的 1.4 请求中位数仍较慢，分别 +18.61% 和 +25.52%；不替换主 14 例 × 每版三次对照。

[comparison-wide](reports/comparison-wide/report.html) 和 [public-wide](runs/public-wide/summary.json) 属于**未采用的宽窗口探索**，原样保留，不与最终 public-narrow 合并。[早期 v2 固定提示实验](analysis/stability/llm-fixed-input.json) 也仅作为探索记录。`analysis/` 中其他早期实验应结合其源码和输入记录解读。

`runs/` 保存输入、完整响应与汇总；`analysis/` 保存专项分析；`reports/` 保存测试日志、报告与哈希；`source/` 保存交付时的源码和文档。已评测的 1.3 历史版本见 [上一版档案](../v1.3/README.md)。

最终交付检查见[validation.json](reports/validation.json)。[固定原始回复审计](runs/fixed-response-final/audit.json)核对21次离线完整服务运行及21次冻结输入重复；这些不计作真实模型调用。
