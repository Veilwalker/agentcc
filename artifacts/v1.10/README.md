# AgentCC 1.10 · 容量约束下的通信结构搜索

以 1.9（1.6 调度流程 + C++）为对照，扩大阶段组合、完整原子分流及合法持有者选择，并让模型生成可执行的组合搜索片段。官方运行时仍为 Python AgentRuntime，数值核心为 C++；没有训练或部署神经网络。

## 交付入口

- [固定源码](source/) · [源码压缩包](agentcc-source-1.10.tar.gz)
- [完整验证说明](source/VALIDATION.md) · [理论与限制](source/THEORY.md)
- [逐例可视化对照](reports/release-comparison/report.html)
- [镜像校验信息](reports/image.json)：`/home/czy/agentcc-images/agentcc-team-1.10.tar`
- [1.9 回退版本](../v1.9/README.md)

本地公开样例不等于平台冻结输入；通信耗时改善不直接换算为官方分数。以验证报告的完整请求及逐例退化项为准，不只看数值内核耗时。生成表达式的作用与工程算法的作用分别记录，不将全部收益归因于模型。

最终84次真实SDK请求全部有效。通信时间几何平均降低16.30%，12例改善、1例持平、1例回退0.71%；完整请求中位数的样例平均2.965→3.053秒，增加0.088秒。每14例平均4,113 Token。仍有10例请求中位数增加，尚未达到逐例不退化目标，详见完整验证说明。

## 实验索引

| 记录 | 范围 |
|---|---|
| [平台历史回退](reports/platform-regression.json) | 用户截图中的 1.6 / 1.7 / 1.8，非受控同环境比较 |
| [完整请求](runs/bounded-output-sdk/summary.json) | 最终候选与1.9，14例、每版三次、真实模型与官方SDK |
| [回放复用后、128输出上限](runs/payload-cache-sdk/summary.json) | 上一候选84次请求，保留一次偏题回复及反馈修复的完整开销 |
| [上一候选完整请求](runs/release-sdk/summary.json) | 分片校验复用之前的 84 次请求；未与最终版混合 |
| [固定模型代码](reports/fixed-release-audit.json) | 最终14例，运行记录中的原始程序，核对状态、编译与成本 |
| [拓扑扰动](reports/mutation-release-audit.json) | 最终42例，改变实际边、方向和共享资源；不调用模型 |
| [回归测试](reports/unit-tests-final.log) | 官方Python/SDK环境446项通过 |
| [真实变更输入](runs/changed-sdk/summary.json) | 4 次实际调用，含紧轮数约束的完整代码生成 |
| [内核消融](analysis/routing-ablation.json) | 140 个合法候选，原路由、分流、整轮重选等 |
| [结构探索](analysis/structure-probe.json) | 140 个候选，更宽阶段及结构变体 |
| [持有者探索](analysis/holder-probe.json) | 14 例，更换已有数据的实际发送者 |
| [互补顺序](analysis/complementary-probe.json) | 42 例，主要为负面结果 |
| [布局探索](analysis/layout-probe.json) | 36 例，收益不足以覆盖准备成本 |
| [归约根探索](analysis/root-probe.json) | 60 例，小收益伴随额外计算，未启用 |
| [思考参数](analysis/thinking-probe.json) | 12 次真实调用；无稳定加速，也无法确认网关采纳参数 |
| [输出上限](analysis/output-limit-probe.json) | 12 次真实调用，128 与48交错；无稳定优势 |
| [短提示](analysis/compact-probe.json) · [JSON提示](analysis/json-probe.json) | 保留错误输出证据，未将错误递推纳入生产 |

早期失败、解析错误与被淘汰的探索均保留在 `analysis/`，不是当前发布方案。离线消融的数学最优值不是模型输出，不能计为真实调用或模型贡献。所有生产方案仍需完整 Replay。

## 构建与导出

```bash
docker build --platform linux/amd64 -t agentcc-team:1.10 /home/czy/agentcc/artifacts/v1.10/source
docker save -o /home/czy/agentcc-images/agentcc-team-1.10.tar agentcc-team:1.10
```

模型 URL、Key、模型名沿用平台配置；真实 `.env` 不打入镜像。构建时在官方基础环境编译 C++，最终镜像以非 root 用户运行并要求扩展成功加载。只执行用户授权的 build/save，不操作其他容器。

已完成构建与导出，镜像为557,793,792字节。SHA256：

```text
c05e5093def1f314f8f4ed39492822098bf061a24e705ce50a1093e271f63056
```

读取导出归档核对了linux/amd64、非root用户、官方入口、强制C++配置及全部生产Python文件哈希；镜像源码与最终测试快照一致。没有额外启动容器或代为上传平台。可直接将上述tar文件下载送测。
