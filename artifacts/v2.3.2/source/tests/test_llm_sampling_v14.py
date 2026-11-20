"""Single-request sampling settings and evidence, without a live model call."""

import json
import unittest
from unittest.mock import patch

from src.llm import _chat_completion, edit_schedule, plan
from tests import test_llm_edits as edit_fixtures


class SamplingProtocolTests(unittest.TestCase):
    def setUp(self):
        fixture = edit_fixtures.EditModelTests()
        fixture.setUp()
        self.task, self.problem, self.settings = fixture.task, fixture.problem, fixture.settings

    @patch("src.llm.build_opener")
    def test_edit_uses_zero_temperature_in_exact_wire_and_evidence(self, factory):
        raw = '{"base":"8e38c7590c94abd2","counts":[1]}'
        edit_fixtures.EditModelTests.response(factory, raw)
        proposal, metadata = edit_schedule(self.task, self.problem, self.settings)
        payload = json.loads(factory.return_value.open.call_args.args[0].data)
        self.assertEqual(payload["temperature"], 0.0)
        self.assertNotIn("seed", payload)
        self.assertEqual(metadata["request_evidence"], payload)
        self.assertEqual(metadata["input_encoding"]["format"], "dense_tables_v4")
        self.assertEqual(json.loads(payload["messages"][1]["content"])["problem"]["groups"][0]["max_chunks"], 2)
        self.assertEqual(metadata["sampling"], {"temperature": 0.0, "guaranteed_deterministic": False})
        self.assertEqual(metadata["response_evidence"]["content"], raw)
        self.assertEqual(proposal["edits"], [["g0", 1]])
        self.assertEqual(factory.return_value.open.call_count, 1)

    @patch("src.llm.build_opener")
    def test_legacy_strategy_keeps_default_sampling(self, factory):
        edit_fixtures.EditModelTests.response(factory, '{"algorithm":"tree"}')
        strategy, metadata = plan(self.task, {}, self.settings)
        payload = json.loads(factory.return_value.open.call_args.args[0].data)
        self.assertEqual(payload["temperature"], 0.2)
        self.assertEqual(metadata["request_evidence"], payload)
        self.assertEqual(strategy, {"algorithm": "tree"})
        self.assertEqual(factory.return_value.open.call_count, 1)

    @patch("src.llm.build_opener")
    def test_explicit_temperature_is_not_ignored_or_misreported(self, factory):
        edit_fixtures.EditModelTests.response(factory, "recorded-test-content")
        content, metadata = _chat_completion("system", "user", self.settings, temperature=0.1)
        payload = json.loads(factory.return_value.open.call_args.args[0].data)
        self.assertEqual(payload["temperature"], 0.1)
        self.assertEqual(metadata["sampling"]["temperature"], 0.1)
        self.assertEqual(content, "recorded-test-content")

    @patch("src.llm.build_opener")
    def test_different_server_counts_are_never_replaced_for_stability(self, factory):
        for count in (0, 2):
            raw = json.dumps({"base": self.problem["base"], "counts": [count]})
            edit_fixtures.EditModelTests.response(factory, raw)
            proposal, metadata = edit_schedule(self.task, self.problem, self.settings)
            self.assertEqual(proposal["edits"], [["g0", count]])
            self.assertEqual(metadata["response_wire_proposal"]["counts"], [count])
            self.assertEqual(metadata["response_evidence"]["content"], raw)
            proposal["edits"][0][1] = 99
            self.assertEqual(metadata["response_proposal"]["edits"], [["g0", count]])
            self.assertEqual(metadata["response_wire_proposal"]["counts"], [count])
        self.assertEqual(factory.return_value.open.call_count, 2)  # one per user-side call


if __name__ == "__main__":
    unittest.main()
