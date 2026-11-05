# AgentCC 实验档案

Git 收录本目录的版本导航、各版 `README.md` 和 `v1.0` 至 `v1.6` 的 `source/` 发布快照。`runs/`、`analysis/`、`reports/`、中间迭代和压缩包保留在开发服务器，未上传；下文指向这些实验资料的链接需在服务器本地查看。历史源码内容保持原样，后续开发在项目根目录进行。

当前交付为 **1.6**，主基准是已发布的 **1.5 revision-05**。从[本轮逐例对照](v1.6/reports/comparison/report.html)、[1.6实验目录](v1.6/README.md)和[完整验证](../VALIDATION.md)开始。1.6扩大通信结构搜索空间，重点处理并行通道与阶段合并；公开测试不能换算为平台得分。

最终发布测量是 `v1.6/runs/release-sdk`。历史源码在各版本 `source/` 中原样保留；`v1.6/analysis/prototypes/` 是研究代码。1.5已收到平台结果：94.2分、14/14有效、8,512 Token、14次模型调用。1.6待平台验证。

先打开 [离线版本看板](index.html)，查看平台成绩、逐例变化和主要证据。它不连接网络、不读取配置，也不把本地测试换算成官方得分。

## 统一目录

```text
artifacts/
├── index.html                  # 可视化导航与历史平台成绩
├── README.md                   # 本说明
├── comparisons/                # 跨版本对照
├── v1.0/
├── v1.1/
├── v1.2/
├── v1.3/                       # 平台 93.9，14/14 有效
├── v1.4/                       # 平台93.0，历史最好仍为1.3的93.9
│   ├── source/                 # 发布时冻结的源码、测试和文档
│   ├── runs/<实验名>/           # 输入、完整响应、summary.json、console.log
│   ├── analysis/               # 专项消融、扰动回归、剖析
│   └── reports/                # 验证清单、测试日志与源码哈希
├── v1.5/                       # 平台94.2；冻结交付与历史实验
└── v1.6/                       # 当前源码、结构探索与完整请求验证
```

版本目录统一用 `v1.x`，不再混用 `v11`、`baseline-1.1` 和无版本散文件。实验目录用含义明确的小写连字符名称，例如 `runs/paired-final/`。代码文件保留 Python 的下划线命名。

## 版本入口

| 版本 | 官方结果 | 源码 | 主要测试与证据 |
| --- | --- | --- | --- |
| 1.0 | 73.4，12/14 有效 | [源码归档](v1.0/source/README.md) | [验证清单](v1.0/reports/validation.json) · [原始 SDK 测试](v1.0/runs/runtime-final/summary.json) |
| 1.1 | 92.1，14/14 有效 | [源码归档](v1.1/source/README.md) | [验证清单](v1.1/reports/validation.json) · [原始 SDK 测试](v1.1/runs/runtime-final/summary.json) |
| 1.2 | 92.1，14/14 有效 | [源码归档](v1.2/source/README.md) | [验证清单](v1.2/reports/validation.json) · [配对测试](v1.2/runs/paired-final/summary.json) · [同目录消融](v1.2/runs/live-ablation-final/summary.json) |
| 1.3 | 93.9，14/14 有效 | [源码归档](v1.3/source/README.md) | [交互对照报告](v1.3/reports/comparison/report.html) · [原始统计](v1.3/reports/comparison/report.json) · [模型调度编辑证据](v1.3/reports/comparison/model-evidence.json) |
| 1.4 | 93.0，14/14 有效 | [源码快照](v1.4/source/README.md) | [最终本地对照](v1.4/reports/comparison/report.html) · [public-narrow 原始汇总](v1.4/runs/public-narrow/summary.json) · [发布时验证](v1.4/source/VALIDATION.md) |
| 1.5 | 94.2，14/14 有效 | [交付快照](v1.5/source/README.md) | [历史实验](v1.5/README.md) |
| 1.6 | 待平台验证 | [交付快照](v1.6/source/README.md) | [完整请求对照与结构探索](v1.6/README.md) |

平台原始录入数据在 `materials/platform-result-1.x.json`，来自用户截图。界面显示的一位小数与逐例三位小数均保留，逐例平均值只是截图近似还原值。最新对照见 [平台 1.2 → 1.3](comparisons/platform-v1.2-v1.3.json)；历史对照见 [平台 1.1 → 1.2](comparisons/platform-v1.1-v1.2.json) 和 [本地 1.0 → 1.1](comparisons/local-v1.0-v1.1.json)，不要把两种口径混用。

## 平台结果

[1.3 官方结果录入](../materials/platform-result-1.3.json) 完成于 2026-11-04 01:03（Asia/Shanghai）：93.9 分，14/14 有效，168 轮通信、14 次模型调用、17,503 Token；用户截图当时排名第 6。与 1.2 相比，逐例展示值的平均总分增加 1.816，其中求解时间分增加 1.704、通信分增加 0.112；平均请求耗时从 8.663 降至 4.901 秒。通信 10 例改善、2 例持平、2 例回退，详见 [逐例对照](comparisons/platform-v1.2-v1.3.json)。

用户随后提供的1.4结果为93.0分、14/14有效、14次模型调用、9,794 Token，通信45.0/50、求解33.0/35。榜单按历史最好成绩仍显示1.3的93.9。

1.3和1.4的冻结快照原样保留。下面分开列出 1.4 与 1.3 历史本地测量，不把不同版本、不同实验的数据混合计算。

## 1.4 历史本地对照

采用 [public-narrow](v1.4/runs/public-narrow/summary.json)：14 个公开样例 × 1.3/1.4 两版 × 每例 3 次，共 84 次真实 SDK 请求全部通过独立 Replay，源码 SHA 前后一致。交互结果见 [最终对照报告](v1.4/reports/comparison/report.html)，数据见 [report.json](v1.4/reports/comparison/report.json)。逐例先取三次中位数，再对 14 例等权统计：

| 指标 | 1.3 | 1.4 | 变化 |
| --- | ---: | ---: | ---: |
| Token 中位数的样例平均 | 1,260.357 | 648.071 | -48.58% |
| 请求时间中位数的样例平均，秒 | 3.351818 | 3.250144 | -3.03% |
| 通信几何平均加速比 | — | 1.009117× | 4 例改善、8 例持平、2 例退化 |

请求时间的 4 个退化项为 `directed_ring/ar` **+34.02%**、`fullmesh_clos/ag` **+8.48%**、`fullmesh_clos/ar` **+1.68%**、`nvlink_like/ar` **+0.91%**；通信退化为 `cloudmatrix/ag` **+0.1732%**、`cloudmatrix/ar` **+0.0817%**。完整原始观测保留，不用后续补测覆盖，也不据此推测单一原因或官方分数。

[295 项自动测试](v1.4/reports/unit-tests-narrow.log) 已通过。84 次主对照、20 次独立补测和 7 次真实补充请求全部通过本地 Replay；42 次离线扰动、21 次固定回复运行及冻结输入重复也通过。[固定原始回复审计](v1.4/runs/fixed-response-final/audit.json)和[交付清单](v1.4/reports/validation.json)记录最终范围；当时口径见[发布时验证报告](v1.4/source/VALIDATION.md)。设计和适用边界见 [THEORY.md](v1.4/source/THEORY.md)；源码使用统一的 [v1.4/source](v1.4/source/README.md) 快照入口。

[最终格式固定提示审计](v1.4/analysis/stability/llm-final-fixed-input.json) 核对了 4 份 `dense_tables_v3` 提示 × 温度 0/0.2 × 各 3 次，共 24 次直接模型请求、15,740 Token，原始回答全部保留。温度 0 下，`nvlink_like/ar` 出现 `[2,2]` 和 `[1,1]`，`directed_ring/ar` 出现 `[8,8]`、`[2,2]`、`[1,1]`，存在真实数量决策变化，不能称为确定性。这些只通过输出协议检查，未运行 SDK 求解或 Replay；三次重复不足以估计 P95，也不能把求解波动全部归因于服务等待。

[独立时延补测报告](v1.4/reports/latency-followup/report.html) 包含 2 例 × 两版 × 各 5 次，共 20 次有效 SDK 请求。1.4 的请求中位数仍较慢：`directed_ring/ar` 为 2.954 → 3.504 秒（+18.61%），`fullmesh_clos/ag` 为 3.004 → 3.771 秒（+25.52%）。补测不替换或并入上方 14 例 × 每版 3 次的主对照，不据此推断单一原因。

[早期 v2 固定提示实验](v1.4/analysis/stability/llm-fixed-input.json) 仅保留为探索；[平台重跑 5 例对照](v1.4/analysis/stability/platform-partial-comparison.json) 缺少输入哈希和完整结果，不能证明是同输入重复。

[comparison-wide](v1.4/reports/comparison-wide/report.html) 与 [public-wide](v1.4/runs/public-wide/summary.json) 是**未采用的宽窗口探索**，不与最终 public-narrow 合并。历史探索保留用于解释取舍。

## 1.3 历史本地对照（2026-11-03）

1.3 已完成本地验证，源码冻结在 [v1.3/source](v1.3/source/README.md)。除下述对照外，231 项自动测试、7 次真实补充请求和 42 项离线扰动也已通过，详见 [1.3 验证快照](v1.3/source/VALIDATION.md)。

[1.3 对照报告](v1.3/reports/comparison/report.html) 合并 [public-01](v1.3/runs/public-01/summary.json) 和 [public-02](v1.3/runs/public-02/summary.json)：14 个公开样例、3 个版本、每例 3 次，共 126 次真实 SDK 请求全部通过独立 Replay。合并前核对每版本源码 SHA、资源限制和计时定义一致；按每例中位数等权比较。

相对 1.2，1.3 的逐例中位请求时间平均值从 4.668 秒降至 3.440 秒，Token 从 2,440 降至 1,258；通信几何平均加速比为 1.010 倍。报告同时保留 5 个通信退化项，最大为 `nvlink_like/ag` 的 +2.90%。相对 1.1，1.3 的 Token 仍较多。1.3 的 42 次请求中，4 次具体模型编辑胜过当前强对照并进入最终方案，详见 [可核验编辑证据](v1.3/reports/comparison/model-evidence.json)。这些是本地公开样例测量，不是官方成绩预测。

## 保留与迁移

本次只移动旧文件，没有删除历史证据，也没有改写归档源码、旧 JSON 或旧日志的内容。迁移前后的文件字节已核对，完整映射在 [迁移清单](v1.3/reports/artifact-migration.json)。1.2 的 `source/` 在整理前已单独冻结。

历史源码和实验报告可能包含当时的旧路径。为保持哈希与原始证据，它们按历史原样保留；需要查找时使用迁移清单。项目根目录当前文档和 `tests/` 工具已改用新路径。`analysis/prototypes/` 和 `v1.1/analysis/collectives-prototype/` 是历史探索代码，不是容器入口。

`source/` 不包含 `.env`、模型 Key、依赖副本或实验输出。发布镜像仅复制项目根目录的 `src/`；整个 `artifacts/` 都不进入镜像。

## 新实验约定

1. 运行前明确目标、源码版本、输入来源与是否真实调用模型。
2. 在当前研究版本的 `runs/<实验名>/` 新建目录；保留完整响应和汇总，避免覆盖已完成实验。现阶段为 `v1.6`，不再向历史发布目录混入新实验。
3. 专项对照写到对应版本的 `analysis/`，单元测试日志和最后验证清单写到 `reports/`。
4. 最终报告分别展示有效率、通信时间、请求时间和 Token。未收到平台成绩时，不填官方分数。
5. `final` 只用于冻结源码后完成的测试。探索中的失败或中间结果保留，并明确其阶段。
