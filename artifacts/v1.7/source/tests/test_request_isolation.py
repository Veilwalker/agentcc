"""Official-runtime request isolation and cancellation without network calls."""

import asyncio
from copy import deepcopy
import threading
import unittest
from unittest.mock import patch

import httpx

from src.config import Settings
from src.llm import LLMError
from src.replay import replay
from src.solver import solve_schedule
from tests.test_solver import make_case
from tests.test_generated_solver import BODY_REFERENCE

try:
    from agenthack import AgentRuntime
except ImportError:
    AgentRuntime = None


SETTINGS = Settings("http://unused.test/v1/chat/completions", "synthetic-key", "test-model")


def request_case(label, operator="all-gather"):
    task, topology = make_case(8, operator)
    task["task_id"] = task["instance_id"] = label
    return {
        "task_id": label, "task_type": "agentcc",
        "input": {"task": task, "topology": topology, "instance_id": label},
        "constraints": {"timeout_ms": 3000},
    }


def short_search(*args, **kwargs):
    kwargs["time_budget_s"] = 0.05
    return solve_schedule(*args, **kwargs)


@unittest.skipIf(AgentRuntime is None, "Official SDK is not on the import path")
class RequestIsolationTests(unittest.TestCase):
    def test_concurrent_requests_preserve_model_counts_and_topologies(self):
        from src.my_agent import MyAgent
        first, second = request_case("first"), request_case("second", "all-reduce")
        for edge in second["input"]["topology"]["edges"]:
            edge["bw"] = "0.25GB/s"
        originals = deepcopy([first, second])
        rendezvous = threading.Barrier(2)

        def planner(system, content, settings, **kwargs):
            rendezvous.wait(2)
            tokens = 11 if '"op":"all-gather"' in content else 22
            return BODY_REFERENCE, {
                "called": True, "usage": {"prompt_tokens": tokens, "completion_tokens": 3},
            }

        async def check():
            runtime = AgentRuntime(MyAgent())
            async with runtime.app.router.lifespan_context(runtime.app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=runtime.app),
                                             base_url="http://in-process") as client:
                    responses = await asyncio.gather(client.post("/solve", json=first),
                                                     client.post("/solve", json=second))
            data = [response.json() for response in responses]
            for number, result in enumerate(data):
                self.assertEqual(result["status"], "success", result)
                self.assertEqual(result["task_id"], originals[number]["task_id"])
                self.assertEqual(result["metadata"]["llm_calls"], 1)
                self.assertEqual(result["metadata"]["total_tokens"], [14, 25][number])
                case = originals[number]["input"]
                self.assertTrue(replay(case["task"], case["topology"], result)["valid"])
                self.assertNotIn("synthetic-key", str(result))
            self.assertNotEqual(data[0]["metadata"]["session_id"], data[1]["metadata"]["session_id"])
            self.assertNotEqual(data[0]["output"]["diagnostics"]["search"]["model_edit"]["space"]["input_sha256"],
                                data[1]["output"]["diagnostics"]["search"]["model_edit"]["space"]["input_sha256"])

        with patch("src.llm._chat_completion", side_effect=AssertionError("unit test must not call network")), \
                patch("src.generated_service.load_settings", return_value=SETTINGS), \
                patch("src.code_policy._chat_completion", side_effect=planner), \
                patch("src.generated_service.solve_schedule", side_effect=short_search):
            asyncio.run(check())
        self.assertEqual([first, second], originals)

    def test_sdk_timeout_returns_while_model_waits_and_next_request_succeeds(self):
        from src.my_agent import MyAgent
        first, second = request_case("slow"), request_case("next")
        first["constraints"]["timeout_ms"] = 30
        second["input"]["task"]["data_size"]["per_rank_bytes"] = 65536
        model_waiting, release_model, model_finished = threading.Event(), threading.Event(), threading.Event()

        def planner(system, content, settings, **kwargs):
            if '"bytes":32768' in content:
                model_waiting.set()
                if not release_model.wait(3):
                    raise LLMError("synthetic test did not release its model")
                model_finished.set()
            return BODY_REFERENCE, {
                "called": True, "usage": {"prompt_tokens": 7, "completion_tokens": 2}}

        async def check():
            runtime = AgentRuntime(MyAgent())
            async with runtime.app.router.lifespan_context(runtime.app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=runtime.app),
                                             base_url="http://in-process") as client:
                    try:
                        timed_out = (await asyncio.wait_for(client.post("/solve", json=first), 1)).json()
                        self.assertTrue(model_waiting.is_set())
                        self.assertFalse(model_finished.is_set(), "the response must not wait for a blocked model")
                        self.assertEqual(timed_out["status"], "timeout")
                        self.assertEqual(timed_out["output"]["answer"], "")
                        successful = (await client.post("/solve", json=second)).json()
                        self.assertEqual(successful["status"], "success", successful)
                        self.assertEqual(successful["metadata"]["llm_calls"], 1)
                        self.assertEqual(successful["metadata"]["total_tokens"], 9)
                    finally:
                        release_model.set()
                        await asyncio.to_thread(model_finished.wait, 1)

        with patch("src.llm._chat_completion", side_effect=AssertionError("unit test must not call network")), \
                patch("src.generated_service.load_settings", return_value=SETTINGS), \
                patch("src.code_policy._chat_completion", side_effect=planner), \
                patch("src.generated_service.solve_schedule", side_effect=short_search):
            asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
