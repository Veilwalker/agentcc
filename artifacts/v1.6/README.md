# 1.6 · 通信结构迭代

交付源码在项目根目录 `src/`，独立快照在 [source/](source/)。主基准为已发布的 **1.5 revision-05**，不覆盖历史源码。当前没有 1.6 官方成绩。[源码压缩包](agentcc-source-1.6.tar.gz)仅包含代码、测试和说明，不是可提交的容器镜像。

先看 [完整验证](../../VALIDATION.md)、[设计原理](../../THEORY.md)和[交互式对照报告](reports/comparison/report.html)。本版把实际多通道路由与多路通信阶段引入生产，LLM 输出具体阶段编辑，完整 Replay 决定是否采用。正常每实例一次模型请求。

## 发布证据

| 记录 | 用途 |
| --- | --- |
| [release-sdk](runs/release-sdk/summary.json) | 最终源码，14 例 × 两版 × 各三次，完整官方 SDK / 真实模型对照 |
| [固定回复与消融](analysis/release-fixed-replies.json) | 同回复可重复、零编辑、错误编辑、离线有限目录最优；不联网 |
| [三次真实重复的稳定性](analysis/release-stability.json) | 输入/提示哈希与模型回复、调度差异 |
| [四个真实拓扑扰动](runs/live-mutations/summary.json) | 修改输入后重新请求模型并完整回放；独立检查，不作为加速统计 |
| [42 个输入扰动](analysis/mutations.json) | 实例名字不变，修改实际边、方向、带宽、重叠共享池；合成回复 |
| [最终代码回归](reports/unit-cached.log) | 421 项回归测试 |
| [源码清单](source/manifest.json) | 独立源码快照及逐文件 SHA-256 |

`release-sdk` 是发布统计来源。`verified-sdk` 是路径缓存优化前的完整 84 次对照，单独保留，不混合来源哈希不同的数据。缓存前后使用同一组真实回复的最终调度一致，见 `cached-fixed-replies.json`。

## 结构探索与未采用方向

| 试验 | 发现与取舍 |
| --- | --- |
| [并行路由](reports/fabric-routing.json) | 固定轮结构下也能显著减轻 CloudMatrix 共享通道拥塞；证明收益不只来自少轮数 |
| [84 个结构组合](analysis/structure-ablation.json) | 输入顺序、容量亲和顺序、压力顺序及第二跳分支的组合；压力顺序未稳定获益，保留成本比较 |
| [98 个宽扇出方案](analysis/fanout-ablation.json) | 阶段宽度 2～8；过宽会放大同步拥塞和输出。生产上限为三维、八路 |
| [全部阶段 DP](reports/stage-space.json) | 离线有限目录参考，不将它伪装为模型提案；不保证全局最优 |
| [分包比较](reports/stage-coalesced.json) | 包条数、真实路径包时延和容量时间的折衷 |
| [亲和顺序](reports/stage-coalesced-locality.json) | 对 Uneven 有帮助，对 Mesh 可明显变差；没有统一强制应用 |

## 模型接口探索与失败记录

`runs/` 里按实验目的命名，避免把多个中间版本都叫“final”：

- `stage-pilot`：自由阶段宽度，回复格式与质量不够稳定。
- `fusion-pilot`：可冲突的合并编号，模型可能返回互相重叠的编辑。
- `band-pilot`、`single-band-pilot`：互不重叠的区段编辑，减少冲突。
- `lazy-band-pilot`、`named-band-pilot`：延迟生成原子表达式、语义名称协议。
- `parser-regression`：裸 JSON 数组被旧对象解析器误拒绝的中断对照，**失败证据，不进入发布收益**。已修复并增加完整协议链测试。
- `verified-sdk`：修复协议后的第一组完整对照。
- `release-sdk`：容量编译和交换机路径缓存后，最终发布对照。

原型源码在 `analysis/prototypes/`，剖析在 `analysis/`，日志在 `reports/`；都不进入镜像。真实模型、已记录回复重放、合成回复、纯结构实验分别标注，不混计调用次数与 Token。
