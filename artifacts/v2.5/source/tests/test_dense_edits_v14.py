"""The dense input is a reversible encoding, never a model decision."""

from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from src.llm import (build_edit_request, compact_edit_problem,
                     dense_edit_problem, edit_schedule)
from src.schedule_edit import extract_problem
from src.collectives import build
from tests.prompt_compression import (dense_candidate, factored_candidate, unpack_candidate,
                                      restore_projection_as_problem, unpack_request)
from tests import test_llm_edits as edit_fixtures
from tests.test_route_balance import parallel_case


def rounded_projection(problem):
    def rounded(value):
        if isinstance(value, float):
            return float(f"{value:.4g}")
        if isinstance(value, list):
            return [rounded(item) for item in value]
        if isinstance(value, dict):
            return {key: rounded(item) for key, item in value.items()}
        return value
    return rounded(compact_edit_problem(problem))


class DenseEditProtocolTests(unittest.TestCase):
    def setUp(self):
        # Reuse only fixture data and its explicitly mocked HTTP transport.
        fixture = edit_fixtures.EditModelTests()
        fixture.setUp()
        self.task, self.problem, self.settings = fixture.task, fixture.problem, fixture.settings

    def test_direction_paths_and_capacity_are_exactly_reversible(self):
        compact = compact_edit_problem(self.problem)
        compact["constraints"] += [[["s0", "s1"], 7], [["s1", "s0"], 3]]
        compact["rounds"][0]["loads"] += [17.0, None]
        compact["groups"][0]["delta"] += [1.25, -0.125]
        compact["groups"][0]["id"] = "nonsequential-group-19"
        original = deepcopy(compact)
        wire, aliases = dense_edit_problem(compact)
        self.assertEqual(unpack_candidate(wire, aliases), original)
        self.assertEqual(compact, original)
        forward, reverse = wire["constraints"][-2:]
        self.assertEqual(forward[:2], list(reversed(reverse[:2])))
        self.assertEqual([forward[-1], reverse[-1]], [7, 3])
        self.assertIsNone(wire["rounds"][0][5][-1])
        self.assertEqual(aliases["group_ids"], ["nonsequential-group-19"])
        self.assertEqual(wire["groups"][0]["paths"][0][0], wire["groups"][0]["paths"][1][0])
        self.assertEqual(wire["groups"][0]["paths"][0][-1], wire["groups"][0]["paths"][1][-1])

    def test_identical_numeric_columns_do_not_merge_physical_pools(self):
        compact = compact_edit_problem(self.problem)
        compact["constraints"] = [["pool-a", 1], ["pool-b", 1], ["pool-c", 9]]
        compact["rounds"][0]["loads"] = [5, 5, 5]
        compact["groups"][0]["delta"] = [0, 0, 0]
        wire, aliases = dense_edit_problem(compact)
        self.assertEqual(len(wire["constraints"]), 3)
        self.assertEqual(len(aliases["pools"]), 3)
        self.assertEqual(unpack_candidate(wire, aliases), compact)

    def test_repeated_and_overlapping_pool_effects_keep_multiplicity(self):
        task, graph = parallel_case("all-reduce")
        baseline = build(task, graph, 4, kind="xor_tree", resource_weight=0)
        graph.resources["common"] = 12e9
        for edge in graph.edge_resources:
            graph.edge_resources[edge] += ("common",)
        problem, _ = extract_problem(task, graph, baseline, time_budget_s=1)
        compact = compact_edit_problem(problem)
        wire, aliases = dense_edit_problem(compact)
        self.assertEqual(unpack_candidate(wire, aliases), compact)
        common_index = next(i for i, row in enumerate(compact["constraints"]) if row[0] == "common")
        common_index = aliases["column_order"].index(common_index)
        self.assertTrue(compact["groups"])
        for row in wire["groups"]:
            self.assertEqual(row["delta"][common_index], 0)
        self.assertEqual(len(wire["groups"]), len(problem["groups"]))
        self.assertEqual(len(wire["rounds"]), len(problem["rounds"]))

    def test_production_matches_both_independent_prototype_roundtrips(self):
        compact = compact_edit_problem(self.problem)
        named, named_aliases = dense_edit_problem(compact)
        dense, dense_aliases = dense_candidate(compact)
        self.assertEqual(named_aliases, dense_aliases)
        self.assertEqual({key: value for key, value in named.items() if key != "groups"},
                         {key: value for key, value in dense.items() if key != "groups"})
        self.assertEqual(named["groups"], [dict(zip(("round", "max_chunks", "delta", "paths", "bounds"), row))
                                           for row in dense["groups"]])
        self.assertEqual(unpack_candidate(named, named_aliases), compact)
        self.assertEqual(unpack_candidate(dense, dense_aliases), compact)
        factored, aliases = factored_candidate(compact)
        self.assertEqual(unpack_candidate(factored, aliases, factored=True), compact)

    def test_column_permutations_produce_identical_wire_without_moving_groups(self):
        compact = compact_edit_problem(self.problem)
        compact["constraints"] += [[["s1", "s0"], 3], [["s0", "s1"], 7], ["p0", 2]]
        compact["rounds"][0]["loads"] += [None, 17, 4]
        compact["groups"][0]["delta"] += [-0.125, 1.25, 0]
        compact["groups"].append({**deepcopy(compact["groups"][0]), "id": "g91", "max_chunks": 9})
        reference, audit = dense_edit_problem(compact)
        self.assertEqual(len(reference["constraints"]), 5)  # p0 at capacities 1 and 2 stay separate.
        for permutation in ([4, 3, 2, 1, 0], [1, 4, 0, 2, 3], [2, 0, 4, 3, 1]):
            changed = deepcopy(compact)
            changed["constraints"] = [compact["constraints"][i] for i in permutation]
            for row, original in zip(changed["rounds"], compact["rounds"]):
                row["loads"] = [original["loads"][i] for i in permutation]
            for row, original in zip(changed["groups"], compact["groups"]):
                row["delta"] = [original["delta"][i] for i in permutation]
            actual, actual_audit = dense_edit_problem(changed)
            self.assertEqual(actual, reference)
            self.assertEqual(actual_audit["group_ids"], ["g0", "g91"])
            self.assertEqual(unpack_candidate(actual, actual_audit), changed)
            self.assertNotEqual(actual_audit["column_order"], audit["column_order"])

    def test_every_saved_v13_prompt_is_reversible_without_losing_groups(self):
        root = Path(__file__).resolve().parents[1] / "artifacts/v1.3/runs"
        evidence = sorted(root.glob("public-*/v1.3/*.json"))
        if not evidence:
            self.skipTest("historical v1.3 evidence is not bundled with source-only builds")
        for path in evidence:
            with self.subTest(path=path.name):
                model = json.loads(path.read_text())["output"]["diagnostics"]["llm"]
                compact = json.loads(model["request_evidence"]["messages"][1]["content"])["problem"]
                wire, aliases = dense_edit_problem(compact)
                self.assertEqual(unpack_candidate(wire, aliases), compact)
                self.assertEqual(len(wire["groups"]), len(compact["groups"]))
                self.assertEqual(len(wire["rounds"]), len(compact["rounds"]))

    def test_request_only_exposes_decisions_the_language_can_change(self):
        prompt, content, audit = build_edit_request(self.task, self.problem)
        value = json.loads(content)
        self.assertEqual(value["task"], {"operator": "all-reduce"})
        self.assertNotIn("base", value["problem"])
        self.assertNotIn("chunk_bytes", value["problem"])
        self.assertEqual(audit["chunk_bytes"], 32768)
        self.assertEqual(audit["base"], self.problem["base"])
        self.assertIn("exactly 1 integers", prompt)
        self.assertIn("whole atoms", prompt)
        self.assertIn("Replay", prompt)
        self.assertEqual(audit["format"], "dense_tables_v4")

    def test_named_wire_matches_saved_real_trial_without_changing_its_counts(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "artifacts/v1.4/analysis/prompt/named-groups-trial.json"
        if not path.exists():
            self.skipTest("real prompt-format trial is not bundled with source-only builds")
        for trial in json.loads(path.read_text()):
            with self.subTest(case=trial["case"]):
                saved = json.loads((root / trial["source_file"]).read_text())["output"]["diagnostics"]["llm"]
                compact = unpack_candidate(json.loads(saved["request_evidence"]["messages"][1]["content"])["problem"],
                                           saved["input_encoding"]["aliases"])
                wire, aliases = dense_edit_problem(compact)
                recorded = trial["metadata"]["request_evidence"]["messages"]
                self.assertEqual(wire, json.loads(recorded[1]["content"])["problem"])
                self.assertEqual(aliases, saved["input_encoding"]["aliases"])
                system, _, audit = build_edit_request({}, restore_projection_as_problem(compact))
                self.assertNotEqual(system, recorded[0]["content"])
                self.assertEqual(audit["format"], "dense_tables_v4")
                proposal = json.loads(trial["raw_response"])
                self.assertEqual(proposal["base"], wire["base"])
                self.assertEqual(len(proposal["counts"]), len(wire["groups"]))
                self.assertTrue(all(type(count) is int and 0 <= count <= group["max_chunks"]
                                    for count, group in zip(proposal["counts"], wire["groups"])))

    def test_opaque_name_changes_are_only_a_reversible_relabeling(self):
        compact = compact_edit_problem(self.problem)
        before, aliases_before = dense_edit_problem(compact)
        for row in compact["constraints"]:
            row[0] += "-arbitrary-name"
        for group in compact["groups"]:
            group["paths"] = [[name + "-renamed" for name in path] for path in group["paths"]]
        after, aliases_after = dense_edit_problem(compact)
        self.assertEqual(before, after)
        self.assertNotEqual(aliases_before, aliases_after)
        self.assertEqual(unpack_candidate(after, aliases_after), compact)

    def test_changed_capacity_and_direction_change_semantic_input(self):
        _, before, _ = build_edit_request(self.task, self.problem)
        self.problem["rounds"][0]["constraints"][0]["capacity_Bps"] *= 2
        _, changed, audit = build_edit_request(self.task, self.problem)
        self.assertNotEqual(before, changed)
        value = json.loads(changed)["problem"]
        restored = unpack_request(value, audit)
        self.assertEqual(restored["constraints"][0][1], 2)
        self.assertEqual(restored["rounds"][0]["loads"][0], 32.77)
        self.assertEqual(restored["groups"][0]["delta"][0], -32.77)

    @patch("src.llm.build_opener")
    def test_raw_counts_and_nonsequential_group_mapping_remain_exact(self, factory):
        self.problem["groups"][0]["id"] = "g93"
        self.problem["groups"].append({**deepcopy(self.problem["groups"][0]), "id": "g4", "max_chunks": 7})
        raw = '{"base":"8e38c7590c94abd2","counts":[1,3]}'
        edit_fixtures.EditModelTests.response(factory, raw)
        proposal, metadata = edit_schedule(self.task, self.problem, self.settings)
        self.assertEqual(proposal["edits"], [["g93", 1], ["g4", 3]])
        self.assertEqual(metadata["response_evidence"]["content"], raw)
        self.assertEqual(metadata["response_wire_proposal"], json.loads(raw))
        self.assertEqual(metadata["input_encoding"]["aliases"]["group_ids"], ["g93", "g4"])
        request = json.loads(factory.return_value.open.call_args.args[0].data)
        self.assertEqual(metadata["request_evidence"], request)
        self.assertEqual(unpack_request(json.loads(request["messages"][1]["content"])["problem"],
                                         metadata["input_encoding"]), rounded_projection(self.problem))
        self.assertNotIn("fake-secret", json.dumps(metadata))

    @patch("src.llm.build_opener")
    def test_legacy_explicit_edits_are_untouched(self, factory):
        raw = '{"base":"8e38c7590c94abd2","edits":[["g0",1]]}'
        edit_fixtures.EditModelTests.response(factory, raw)
        proposal, metadata = edit_schedule(self.task, self.problem, self.settings)
        self.assertEqual(proposal, json.loads(raw))
        self.assertEqual(metadata["proposal_encoding"], "explicit_edits_v1")

    def test_no_group_topology_summary_is_not_omitted_or_aliased_lossily(self):
        self.problem.update(groups=[], rounds=[], topology={
            "edge_examples": [["r0", "s0", 2, 1], ["s0", "r1", 1, 1]],
            "pools": [{"name": "shared", "members": [["r0", "s0"], ["s0", "r1"]], "GBps": 1}],
        })
        _, content, audit = build_edit_request(self.task, self.problem)
        wire = json.loads(content)["problem"]
        self.assertEqual(wire["topology"], self.problem["topology"])
        self.assertEqual(unpack_request(wire, audit), rounded_projection(self.problem))


if __name__ == "__main__":
    unittest.main()
