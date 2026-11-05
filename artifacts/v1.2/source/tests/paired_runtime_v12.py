"""Contemporaneous, alternating official-SDK runs of archived 1.1 and 1.2.

Explicit real-network benchmark. Both versions use the same .env, current public
input, single CPU, memory cap and gateway. No container commands are executed.
"""

import argparse
import asyncio
from contextlib import AsyncExitStack
import gc
import importlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

import httpx
from agenthack import AgentRuntime

from src.cli import read, write
from src.config import read_env_file
from src.my_agent import MyAgent
from src.replay import replay


def archived_agent():
    root = Path("artifacts/baseline-1.1/src").resolve()
    spec = importlib.util.spec_from_file_location("baseline_v11", root / "__init__.py",
                                                submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return importlib.import_module("baseline_v11.my_agent").MyAgent()


async def run(output, samples, repeat):
    records = []
    async with AsyncExitStack() as stack:
        clients = {}
        for version, agent in [("1.1", archived_agent()), ("1.2", MyAgent())]:
            runtime = AgentRuntime(agent)
            await stack.enter_async_context(runtime.app.router.lifespan_context(runtime.app))
            clients[version] = await stack.enter_async_context(httpx.AsyncClient(
                transport=httpx.ASGITransport(app=runtime.app), base_url="http://official-sdk-test"))
        for rep in range(repeat):
            for index, label in enumerate(samples):
                folder = Path("materials/spec/examples/256-rank") / label
                task, topology = read(folder / "task.json"), read(folder / "topology.json")
                request = {"task_id": task["task_id"], "task_type": "agentcc",
                           "input": {"task": task, "topology": topology, "instance_id": task["instance_id"]},
                           "constraints": {"timeout_ms": 180000}}
                order = ["1.1", "1.2"] if (index + rep) % 2 == 0 else ["1.2", "1.1"]
                for version in order:
                    gc.collect()
                    start = time.monotonic()
                    response = await clients[version].post("/solve", json=request)
                    elapsed = time.monotonic() - start
                    data = response.json()
                    row = {"sample": label, "repeat": rep, "version": version,
                           "order": order, "request_s": elapsed, "status": data["status"],
                           "calls": data.get("metadata", {}).get("llm_calls"),
                           "tokens": data.get("metadata", {}).get("total_tokens")}
                    if data["status"] == "success":
                        checked = replay(task, topology, data)
                        diagnostics = data["output"]["diagnostics"]
                        row.update(valid=checked["valid"], time_us=checked["total_time_us"],
                                   llm_s=diagnostics["llm"]["duration_ms"] / 1000,
                                   model_selected=diagnostics["search"].get("model_edit", {}).get("selected"))
                    else:
                        row.update(valid=False)
                    path = Path(output) / f"{label.replace('/', '_')}-v{version}-r{rep}.json"
                    write(path, data)
                    records.append(row)
                    write(Path(output) / "summary.json", records)
                    print(json.dumps(row), flush=True)
    return 0 if all(row["valid"] and row["calls"] == 1 for row in records) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", default=["mesh2d/ag", "mesh2d/ar"])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--all-public", action="store_true", help="all 14 public 256-rank inputs")
    parser.add_argument("--output-dir", default="artifacts/v1.2/paired-runtime")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("repeat must be positive")
    if args.all_public:
        args.samples = [str(path.parent.relative_to("materials/spec/examples/256-rank"))
                        for path in sorted(Path("materials/spec/examples/256-rank").glob("*/*/task.json"))]
    # The archived package resolves its own project root. Supply the same
    # existing configuration to both versions through this process only; no
    # credential is copied into the archived source or benchmark artifacts.
    for name, value in read_env_file(Path(".env")).items():
        if name in {"PLATFORM_LLM_URL", "PLATFORM_API_KEY", "PLATFORM_LLM_MODEL", "AGENTCC_LLM_TIMEOUT_S"}:
            os.environ.setdefault(name, value)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (1024 ** 3, 1024 ** 3))
    raise SystemExit(asyncio.run(run(args.output_dir, args.samples, args.repeat)))
