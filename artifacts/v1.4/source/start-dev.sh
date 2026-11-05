#!/usr/bin/env bash
set -euo pipefail

# 创建并在后台运行基础开发容器。
docker run -dit \
    --name agentcc-dev \
    -v /home/czy/agentcc:/app \
    -w /app \
    -p 8080:8080 \
    agenthack/base:1.1.1 \
    /bin/bash

# 进入容器：sudo docker exec -it agentcc-dev /bin/bash
# 已有容器停止后再次启动：sudo docker start agentcc-dev
