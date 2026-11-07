# AgentCC 1.7 独立源码快照

此目录包含发布时的生产代码、测试、文档和 Dockerfile。`manifest.json` 逐文件记录 SHA-256；生产代码与最终 84 次 SDK 对照一致。它是源码，不是已导出的容器镜像。

解压源码包后，可在本目录构建：

```bash
docker build --platform linux/amd64 -t agentcc-team:1.7 .
```

本机需已有官方 `agenthack/base:1.1.1`。真实 Key 不在归档中，通过环境变量提供。新建 `.env` 时参照 `.env.example`；镜像只复制 `src/`。

源码包不包含旧版本、公开样例、官方 SDK、论文或原始实验。完整对照工具需要原开发项目的 `materials/`、`vendor/` 和历史快照。README、THEORY、VALIDATION 原样保留发布时内容，其中实验链接应在完整开发项目根目录查看，不能在独立源码包中找到。

验证结论：通信有改善，完整请求更慢，平台得分尚未验证。不要将本地公开样例结果视为官方成绩。
