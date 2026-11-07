# AgentCC 1.8 · C++ 容量计算

本版将容量计算中的高频循环迁移至 C++，保留 1.7 的模型代码生成、搜索空间、候选顺序和独立 Replay。它是 **Python 官方运行时 + C++ 数值核心**，不是整个项目重写。1.7 的源码快照和压缩包不修改。

## 交付入口

- [源码快照](source/) · [源码包](agentcc-source-1.8.tar.gz)
- [完整验证与实测数据](../../VALIDATION.md)
- [三版逐例对照报告](reports/comparison/report.html)：1.6 / 1.7 / 1.8，可切换通信、请求、CPU、Token
- [固定模型回复对照](reports/fixed-model.json)：隔离远端模型波动
- [设计与保证边界](../../THEORY.md)

Git 保存源码快照和版本说明。原始运行、官方环境、报告与压缩包保留在开发服务器，未随 Git 上传。下面的测试是本地公开样例验证，不是平台得分。

## 改动范围

| 模块 | 本版处理 |
| --- | --- |
| 路径容量编译 | C++ 累计每条边和共享资源的系数，保留占用次数与顺序 |
| 整轮容量分配 | C++ 维护负载，按原评分与平局规则选路 |
| 搜索、LLM、Replay | 保留 Python 实现及 1.7 语义 |
| 构建 | 在官方基础环境中编译扩展，最终镜像要求成功加载 C++ |
| 可核验性 | 响应记录后端；基准对 C++ 源文件和扩展记录哈希 |

不使用快速数学变换或仅开发机支持的指令集。每轮分配状态独立，无跨请求共享负载。开发时保留 Python 参考实现，正式镜像设置 `AGENTCC_CAPACITY_BACKEND=cpp`，不能悄悄退回 Python 后仍宣称使用 C++。

## 真实接口结论

14 个公开样例 × 1.6 / 1.7 / 1.8 × 各三次，共 **126/126 有效请求**。统一官方 Python 3.11 和 SDK、单 CPU、1 GiB 地址空间，轮换版本顺序。先逐例取中位数，再对样例等权统计：

| 指标 | 1.6 | 1.7 | 1.8 |
| --- | ---: | ---: | ---: |
| 完整请求 | 3.378 s | 5.920 s | 6.021 s |
| 请求 CPU | 2.194 s | 3.559 s | 2.589 s |
| 模型时长 | 2.251 s | 5.516 s | 5.625 s |
| 实际 Token 折合每14例 | 5,146 | 5,917 | 5,955 |

**CPU 相对1.7减少27.26%，但这轮完整请求增加1.71%，没有证明端到端加速。** 14例CPU都下降，完整请求7例更快、7例更慢。真实调用下1.7和1.8的所有最终调度也完全一致；没有新增模型请求，Token差异来自实际回复变化。

相对本轮1.6，通信12例改善、1例持平、1例回退，几何平均1.148倍；这继承自1.7的算法，不能归因于C++。1.8仍有更慢请求和略高CPU，不能称为全面优于1.6。模型与计算重叠，不能用CPU降幅预测总延迟；三次重复也不足以将小幅总延迟差异完全归因于服务器。

[主测量](runs/release-sdk/summary.json) · [统计和稳定性](reports/release-analysis.json) · [逐例报告](reports/comparison/report.html)。全部源码哈希前后一致。主测试每次只调用一次模型，1.8的27/42次模型候选被采用。

## 固定回复对照

使用 1.7 真实调用保存的原始回复；网络禁用，实际模型调用和 Token 均为零。14 个样例 × 两版 × 模拟等待 0/5 秒，各一次，共 **56 次完整官方 SDK 请求**。全部有效，所有提示词及最终调度哈希相同，生产源码前后不变。

| 模拟模型等待 | 指标 | 1.7 | 1.8 | 变化 |
| --- | --- | ---: | ---: | ---: |
| 0 秒 | 完整请求平均 | 3.557 s | 2.573 s | -27.66% |
| 0 秒 | 请求 CPU 平均 | 3.550 s | 2.567 s | -27.68% |
| 0 秒 | 容量准备平均 | 1.854 s | 0.868 s | -53.21% |
| 5 秒 | 完整请求平均 | 5.417 s | 5.403 s | -0.26% |
| 5 秒 | 请求 CPU 平均 | 3.574 s | 2.591 s | -27.51% |

以上说明 C++ 确实减少计算，但模型等待主导时总时间改善很小。每个条件只有一次观察，不据此估计尾延迟或统计显著性。原始记录见 [fixed-model](runs/fixed-model/summary.json)。

## 兼容性与测试记录

- [开发机完整回归](reports/unit-host.log)：439 项通过。
- [官方 Python 3.11 完整回归](reports/unit-image-python-final.log)：439 项通过。
- C++ 差分测试包含 6,000 次选路与平局比较、共享资源重复占用、状态隔离、非法参数和强制加载失败处理。
- [拓扑扰动回归](analysis/mutations.json)：42/42有效，36例执行已记录模型程序、6例因阶段缺路明确回退；不计真实模型调用。
- 官方 `agenthack/base:1.1.1` 的 Python 3.11.16、SDK 与动态库从归档读取，放在本项目 `materials/runtime-1.1.1/`；使用其解释器编译扩展并运行测试，不启动容器、不安装系统软件。
- [首次官方环境测试](reports/unit-image-python.log)保留了测试工具子进程未指定官方动态加载器导致的失败。修正测试工具后完整回归通过，未改动调度算法。

这不是完整镜像构建验证。实际 Docker 构建和平台评测仍由用户执行；构建文件会在官方基础环境内再次编译并检查扩展导入。

## 复现

已有本项目提取的官方运行时后，可执行：

```bash
python3 -m tests.image_runtime -- native/build.py
AGENTCC_CAPACITY_BACKEND=cpp python3 -m tests.image_runtime -- -m unittest discover -s tests -v
AGENTCC_CAPACITY_BACKEND=cpp python3 -m tests.image_runtime -- -m tests.native_comparison --output-dir artifacts/v1.8/runs/my-fixed-model
AGENTCC_CAPACITY_BACKEND=cpp python3 -m tests.image_runtime -- -m tests.compare_versions --versions 1.6 1.7 current --all-public --repeat 3 --output-dir artifacts/v1.8/runs/my-release-sdk
```

最后一条访问已配置的官方模型接口；固定回复对照不联网。输出目录必须是新目录。计时对照期间不并行运行其他 CPU 实验。

镜像构建、运行和导出命令见 [项目说明](../../README.md)。使用 `source/` 作为构建上下文可避免后续根目录迭代影响这一版。1.8 未获得平台成绩，不把本地差值换算成预期加分。
