FROM agenthack/base:1.1.1 AS native-build

USER root
RUN apt-get update && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY native/ /build/native/
RUN python native/build.py --output-dir /build/extension

FROM agenthack/base:1.1.1

WORKDIR /app
COPY --chown=agent:agent src/ /app/src/
COPY --from=native-build --chown=agent:agent /build/extension/ /app/src/

USER agent
ENV AGENTCC_CAPACITY_BACKEND=cpp
RUN python -c "from src.capacity_backend import BACKEND; assert BACKEND == 'cpp'"
CMD ["python", "-m", "src.server"]
