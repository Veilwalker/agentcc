# AgentCC 1.10 源码快照

此快照对应最终 bounded-output-sdk 84 次官方 SDK 完整请求测试的生产源码。
数值核心为 C++，官方 AgentRuntime、编排、模型代码解释和独立 Replay 为 Python。

构建命令：

```bash
docker build --platform linux/amd64 -t agentcc-team:1.10 /home/czy/agentcc/artifacts/v1.10/source
docker save -o /home/czy/agentcc-images/agentcc-team-1.10.tar agentcc-team:1.10
```

不含真实 .env、Key、官方 SDK、比赛材料、实验响应或本机编译扩展。
构建时在官方基础环境重新编译 C++。最终镜像只复制生产代码和编译扩展。

README、THEORY、VALIDATION 是发布时根文档的副本；指向 artifacts/、materials/ 的链接相对原项目根目录解释。
验证原始记录保留在父项目 artifacts/v1.10/runs/ 与 reports/；独立源码包不重复打包这些大文件。
需要官方样例、SDK或历史对照的测试，应在父项目内按 README 运行；受控实验文件依赖本地材料。
SHA256清单覆盖本快照所有发布文件，清单自身除外。不以本地改善预测官方分数。
