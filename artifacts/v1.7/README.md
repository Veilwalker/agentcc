# AgentCC 1.7 · 发布与实验目录

> 此版已冻结。下面保留1.7发布时的测量与探索，当前开发为[1.8](../v1.8/README.md)；复现1.7请使用本目录的[source快照](source/SNAPSHOT.md)。

本版重点是通信结构和真实代码生成：扩大阶段组合、完整原子多路径分流，模型编写搜索代码，在实际容量表上执行，必要时反馈修正。**通信改善伴随更慢的完整请求；尚无官方成绩。** 根目录 `src/` 为当前代码，`source/` 为独立发布快照，旧 1.6 快照不修改。

## 交付入口

- [发布时项目说明与构建命令](source/README.md)
- [发布时设计理论与能力边界](source/THEORY.md)
- [发布时完整验证与退化项](source/VALIDATION.md)
- [逐例交互报告](reports/comparison/report.html)：通信、完整请求、CPU、Token、响应体
- [源码快照](source/) · [源码包](agentcc-source-1.7.tar.gz) · [发布校验](reports/validation.json)

Git 保存版本导航和源码快照；原始实验、官方材料、SDK、压缩包保留在开发服务器。浏览 GitHub 时下列本地记录链接可能不可用，这不表示它们进入了提交镜像。

## 最终发布证据

| 记录 | 范围 | 结果 |
| --- | --- | --- |
| [release-sdk](runs/release-sdk/summary.json) | 14 例 × 两版 × 三次真实官方 SDK 请求 | 84/84 有效，生产源码前后一致 |
| [comparison](reports/comparison/report.json) | 逐例中位数、样例等权 | 通信 11 改善 / 2 持平 / 1 回退；几何平均 1.157 倍 |
| [unit-final](reports/unit-final.log) | 完整自动回归 | 432 项通过 |
| [release-fixed-code](analysis/release-fixed-code.json) | 14 例真实代码离线重放、重复编译、模块消融 | 预览与独立 Replay 一致；不计真实模型调用 |
| [release-stability](analysis/release-stability.json) | 三次真实调用的源码、调度与耗时 | 新版 14 例调度不变，9 例生成代码写法有变化 |
| [mutations](analysis/mutations.json) | 保留名称，修改实际边/方向/容量/资源 | 42/42 合法，36 例执行代码、6 例明确缺路回退 |
| [live-changes](runs/live-changes/summary.json) | 两个真实拓扑扰动、两个紧轮数实例 | 4/4 有效，4 调用、1,856 Token |
| [model-code-evidence](reports/model-code-evidence.json) | 一个被采用的真实生成程序 | 原始交互、实际表、代码、执行、比较可核验 |

每 14 例实际平均 5,910 Token，主对照每请求一次模型调用；修复最多额外一次。27/42 次新请求采用模型候选，其余保留控制。大部分 AR 提升中的确定性保底贡献已单独列出，不将所有收益归给模型。

请求中位数样例平均 **3.685 → 5.620 秒**，CPU **2.533 → 4.152 秒**，平均响应体约 **0.964 → 1.318 MB**。Directed Ring AG 通信回退 **0.713%**。公开样例与平台输入不同，不预测自动榜净提分。

发布后的 [C++ 容量核心对照](analysis/cpp-speed/README.md) 单独记录 4 例 × 两种实现 × 两种模拟模型延迟 × 三次运行，共 48 次请求，调度完全一致。模型立即返回时完整请求均值减少 26.67%；模拟等待 5 秒时减少 2.45%。这是未进入提交代码的加速原型，不计入上面的发布结果，也没有新增真实模型调用。

## 通信结构探索

| 数据与原型 | 内容 | 处理 |
| --- | --- | --- |
| [stripes.json](analysis/stripes.json) · [代码](analysis/stripe_experiment.py) | 56 个普通/四路原子分流配置 | 逐阶段取较好者，进入生产 |
| [widths.json](analysis/widths.json) · [代码](analysis/width_experiment.py) | 28 个三维/四维阶段配置 | 计算增加明显，生产仍至多三维 |
| [independent-ar.json](analysis/independent-ar.json) · [代码](analysis/ar_experiment.py) | 14 个 AR 归约/广播独立划分配置 | 支持正确语义；本组未见额外收益 |
| [splits.json](analysis/splits.json) · [代码](analysis/split_experiment.py) | 21 个 AG 切分配置 | 代价与收益混合，不采用全面增加切分 |
| [roots.json](analysis/roots.json) · [代码](analysis/root_experiment.py) | 49 个 AR 根映射配置 | 不采用在线遍历 |
| [dimensions.json](analysis/dimensions.json) · [代码](analysis/dimension_experiment.py) | 35 个 AG 维度顺序配置 | 在亲和布局基础上未胜原顺序 |
| [ring-stages.json](analysis/ring-stages.json) · [代码](analysis/ring_stage_experiment.py) | 12 个分组环形流水线配置 | 最好约 0.26%，未加入默认搜索 |

这些试验主要比较通信和构造成本，不替代 `release-sdk` 完整请求对照。只有最终生产流程进入主发布统计。环形原型使用实际合法路径、轮初数据和完整 Replay；没有仅凭名称假定邻接关系。

## 模型协议探索与失败保留

`runs/pilot-sdk`、`protocol-pilot-2`、`compact-code-pilot`、`recurrence-pilot` 和 `analysis/*probe*` 是早期协议试验，包含旧源码、失败、截断和部分未完成运行，不能与最终发布合并。裸循环输出误解前驱含义、强行限制行数等都没有进入最终方案；当前将一般协议与紧轮数协议分开，并验证完整返回代码。

`analysis/mutations-harness-failure.json` 保留首次扰动审计中止记录。原因是测试工具把缺路回退时没有阶段表误当作输入未变化，生产返回方案仍有效；修正后重新完整执行 42 例。最终报告分别记录输入哈希、模型上下文哈希、阶段表哈希和回退原因，不把空执行列表称为“模型代码有效”。

目前真实主对照没有出现需要修复的代码。错误后反馈迭代由带模拟坏代码的回归测试覆盖；后续平台真实失败时会记录原始反馈，不伪造成功修复案例。

## 复现

使用项目内官方 SDK 和已配置 `.env`，以下命令只操作本项目；新的输出目录必须不存在。

```bash
PYTHONPATH=vendor:. python3 -m tests.compare_versions --versions 1.6 current --all-public --repeat 3 --output-dir artifacts/v1.7/runs/my-comparison
PYTHONPATH=vendor:. python3 -m tests.summarize_comparison artifacts/v1.7/runs/my-comparison/summary.json --output-dir artifacts/v1.7/reports/my-comparison --target 1.7
PYTHONPATH=vendor:. python3 -m tests.generated_audit fixed --run artifacts/v1.7/runs/release-sdk --output artifacts/v1.7/analysis/my-fixed-code.json
PYTHONPATH=vendor:. python3 -m tests.generated_audit mutations --run artifacts/v1.7/runs/release-sdk --output artifacts/v1.7/analysis/my-mutations.json
```

前两条真实访问模型；后两条重放已记录源码、零模型调用。计时对照期间不要并行运行其他 CPU 实验。官方 Judge、容器运行和平台得分仍由用户送测验证；本轮没有执行 Docker。
