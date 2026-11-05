"""Official SDK protocol tests. Run with PYTHONPATH=vendor:. locally."""

import asyncio
import json
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch

import httpx

try:
    from agenthack import AgentRuntime
except ImportError:
    AgentRuntime = None


@unittest.skipIf(AgentRuntime is None, "Official SDK is not on the import path")
class RuntimeTests(unittest.TestCase):
    def test_entrypoint_calls_official_run(self):
        with patch("agenthack.AgentRuntime.run") as run:
            runpy.run_module("src.server", run_name="__main__")
            run.assert_called_once_with()

    def test_official_protocol_and_model_evidence(self):
        from src.my_agent import MyAgent
        root = Path(__file__).resolve().parents[1] / "materials/spec/examples/shared-resource"
        task = json.loads((root / "all-gather-task.json").read_text())
        topology = json.loads((root / "topology.json").read_text())
        from src.config import Settings
        settings = Settings("http://unused/v1/chat/completions", "test-only", "test-model")
        metadata = {"usage": {"prompt_tokens": 10, "completion_tokens": 4}, "called": True}

        def model(task, problem, **kwargs):
            return ({"bands": [0] * len(problem["bands"])} if "bands" in problem
                    else {"counts": [0] * len(problem["edits"])}), metadata
        runtime = AgentRuntime(MyAgent())

        async def check():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=runtime.app), base_url="http://test") as client:
                self.assertEqual((await client.get("/health")).status_code, 200)
                result = await client.post("/solve", json={"task_id": task["task_id"], "task_type": "agentcc",
                    "input": {"task": task, "topology": topology}, "constraints": {"timeout_ms": 180000}})
                data = result.json()
                self.assertEqual(data["status"], "success", data)
                self.assertEqual(data["metadata"]["llm_calls"], 1)
                self.assertEqual(data["metadata"]["total_tokens"], 14)
                self.assertTrue(data["output"]["diagnostics"]["local_replay"]["valid"])
                self.assertNotIn("strategy", data["output"]["answer"])
                evidence = data["output"]["diagnostics"]["search"]["model_edit"]
                self.assertFalse(evidence["selected"])
                self.assertFalse(evidence["accepted"])
                self.assertEqual((await client.post("/solve", json={"input": {}})).status_code, 422)

        with patch("src.synthesis_service.load_settings", return_value=settings), patch("src.synthesis_service.edit_stages", side_effect=model):
            asyncio.run(check())

    def test_model_failure_is_not_reported_as_success(self):
        from src.my_agent import MyAgent
        from src.llm import LLMError
        from src.config import Settings
        settings = Settings("http://unused/v1/chat/completions", "test-only", "test-model")
        runtime = AgentRuntime(MyAgent())
        root = Path(__file__).resolve().parents[1] / "materials/spec/examples/shared-resource"

        async def check():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=runtime.app), base_url="http://test") as client:
                result = await client.post("/solve", json={"task_id": "public-shared-ag", "input": {
                    "task": json.loads((root / "all-gather-task.json").read_text()),
                    "topology": json.loads((root / "topology.json").read_text())}})
                self.assertEqual(result.json()["status"], "error")

        with patch("src.synthesis_service.load_settings", return_value=settings), patch("src.synthesis_service.edit_stages", side_effect=LLMError("mock model failure")):
            asyncio.run(check())
