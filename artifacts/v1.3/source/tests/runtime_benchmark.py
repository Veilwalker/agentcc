"""Real LLM + unmodified official SDK + public input integration suite.

This is an explicit network test, not part of unittest discovery. It invokes
the SDK's ASGI application in process; it does not implement replacement routes.
"""

import argparse
import asyncio
import copy
import json
import os
from pathlib import Path
import resource
import time

import httpx
from agenthack import AgentRuntime

from src.cli import read, write
from src.my_agent import MyAgent
from src.replay import replay


async def run(output_dir, include_mutation, large_mutations_only=False, labels=None):
    runtime = AgentRuntime(MyAgent())
    records = []
    samples = []
    for path in sorted(Path("materials/spec/examples/256-rank").glob("*/*/task.json")):
        samples.append((path.parent.parent.name + "/" + path.parent.name, read(path), read(path.with_name("topology.json"))))
    small = Path("materials/spec/examples/shared-resource")
    for name in ("all-gather", "all-reduce"):
        samples.append(("shared/" + name, read(small / (name + "-task.json")), read(small / "topology.json")))
    if include_mutation:
        task = read(small / "all-gather-task.json")
        topology = copy.deepcopy(read(small / "topology.json"))
        topology["resources"][0]["capacity"] = "0.25GB/s"
        samples.append(("shared/mutated-capacity", task, topology))
    if large_mutations_only:
        from tests.test_regressions_v11 import perturb_topology
        originals = [sample for sample in samples if sample[0].split("/")[0] in ("cloudmatrix", "sparse_fabric")]
        samples = []
        for label, task, original in originals:
            changed, changes = perturb_topology(original, seed=23, preserve_rank_connectivity=True)
            label += "-mutated-23"
            write(Path(output_dir) / "inputs" / (label.replace("/", "_") + ".json"),
                  {"task": task, "topology": changed, "mutations": changes})
            samples.append((label, task, changed))
    if labels:
        samples = [sample for sample in samples if sample[0] in labels]
        missing = set(labels) - {sample[0] for sample in samples}
        if missing:
            raise ValueError(f"unknown sample labels: {sorted(missing)}")
    async with runtime.app.router.lifespan_context(runtime.app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=runtime.app), base_url="http://official-sdk-test") as client:
            health = await client.get("/health")
            assert health.status_code == 200
            for label, task, topology in samples:
                start = time.monotonic()
                record = {"sample": label}
                response = await client.post("/solve", json={"task_id": task["task_id"], "task_type": "agentcc",
                    "input": {"challenge": "agentcc", "schema_version": "agentcc.joint-test.v1",
                              "instance_id": task["instance_id"], "task": task, "topology": topology},
                    "constraints": {"timeout_ms": 180000}})
                request_time = time.monotonic() - start
                data = response.json()
                record.update(status=data["status"], http_status=response.status_code,
                              request_time_s=round(request_time, 6),
                              sdk_llm_calls=data.get("metadata", {}).get("llm_calls"),
                              sdk_total_tokens=data.get("metadata", {}).get("total_tokens"))
                if data["status"] == "success":
                    report = replay(task, topology, data)
                    search = data["output"]["diagnostics"]["search"]
                    llm = data["output"]["diagnostics"]["llm"]
                    record.update(valid=report["valid"], total_time_us=report["total_time_us"],
                                  rounds=report["nonempty_rounds"], split=report["split"],
                                  model_requested=search["model_requested"],
                                  selected=next(c["actual"] for c in search["candidates"] if c["selected"]),
                                  source_input_sha256=llm.get("source_input_sha256"),
                                  seed_valid=search["candidates"][0]["valid"],
                                  llm_duration_s=round(llm.get("duration_ms", 0) / 1000, 6),
                                  route_refinement=search.get("route_refinement"))
                    edit = search.get("model_edit")
                    if edit is not None:
                        record.update(model_edit=edit, model_proposal=llm.get("response_proposal"),
                                      phases_s=data["output"]["diagnostics"].get("phases_s"))
                else:
                    record.update(valid=False, error=data.get("output", {}).get("error"))
                write(Path(output_dir) / (label.replace("/", "_") + ".json"), data)
                record.update(wall_time_s=round(time.monotonic() - start, 6), response_bytes=len(response.content),
                              max_rss_mib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2))
                records.append(record)
                write(Path(output_dir) / "summary.json", records)
                print(json.dumps(record, ensure_ascii=False), flush=True)
    return 0 if all(r["valid"] and r["sdk_llm_calls"] == 1 for r in records) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="artifacts/v1.3/runs/runtime")
    parser.add_argument("--include-mutation", action="store_true")
    parser.add_argument("--large-mutations-only", action="store_true", help="four changed 256-rank inputs, preserving task IDs")
    parser.add_argument("--samples", nargs="+", help="run only these family/operator labels")
    parser.add_argument("--single-cpu", action="store_true", help="restrict only this benchmark process to one allowed CPU")
    parser.add_argument("--memory-mib", type=int, help="address-space cap for this benchmark process; not a container memory cgroup")
    args = parser.parse_args()
    if args.single_cpu:
        permitted = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {min(permitted)})
    if args.memory_mib is not None:
        if args.memory_mib <= 0:
            parser.error("--memory-mib must be positive")
        resource.setrlimit(resource.RLIMIT_AS, (args.memory_mib * 1024**2, args.memory_mib * 1024**2))
    raise SystemExit(asyncio.run(run(args.output_dir, args.include_mutation, args.large_mutations_only, args.samples)))
