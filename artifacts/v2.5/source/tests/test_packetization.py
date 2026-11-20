"""Packetization preserves complete collective semantics and capacity loads."""

from collections import Counter, defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.replay import replay
from src.route_balance import _PathCosts
from src.topology import Topology
from tests.packetization_experiment import (
    PROJECT, _plan_round, repacketize, run, verify_atomic_preservation,
)
from tests.test_solver import make_case


def fixture(operator="all-gather", *, shared_capacity=None):
    """Two ranks, four slots, and a complete two-contributor AR expression."""
    task, topology = make_case(2, operator)
    if shared_capacity is not None:
        refs = [{"src": rank, "dst": "S0", "bidirectional": True} for rank in task["ranks"]]
        topology["resources"] = [
            {"id": "overlap-a", "capacity": shared_capacity, "edges": refs + refs[:1]},
            {"id": "overlap-b", "capacity": shared_capacity, "edges": deepcopy(refs)},
        ]
    graph = Topology.from_data(topology)
    owned = {rank: [f"{rank}_c{i}" for i in range(4)] for rank in task["ranks"]}
    chunks = [atom for values in owned.values() for atom in values]

    def transfer(action, src, dst, tokens):
        return {"action": action, "src": src, "dst": dst, "path": [src, "S0", dst],
                "chunk_ranges": list(tokens)}

    if operator == "all-gather":
        rounds = [{"round": 1, "transfers": [transfer("copy", "r0", "r1", owned["r0"]),
                                               transfer("copy", "r1", "r0", owned["r1"])]}]
        final = chunks.copy()
    else:
        rounds = [{"round": 1, "transfers": [transfer("reduce", "r0", "r1", owned["r0"])]},
                  {"round": 2, "transfers": [transfer("copy", "r1", "r0",
                                                        [f"r0_c{i}+r1_c{i}" for i in range(4)])]}]
        final = [f"c{i}_sum" for i in range(4)]
    answer = {"operator": operator, "split": 4, "chunks": chunks, "initial_possession": owned,
              "final_chunks": final, "round_count": len(rounds), "rounds": rounds}
    return task, topology, graph, answer


def atomic_rows(answer):
    # Independent test representation: every token keeps its entire expression.
    return Counter((row["round"], t["action"], t["src"], t["dst"], tuple(t["path"]), token)
                   for row in answer["rounds"] for t in row["transfers"] for token in t["chunk_ranges"])


def loads(answer, task, graph):
    chunk_bytes = task["data_size"]["per_rank_bytes"] // answer["split"]
    result = []
    for row in answer["rounds"]:
        edges, resources = defaultdict(int), defaultdict(int)
        for t in row["transfers"]:
            amount = len(t["chunk_ranges"]) * chunk_bytes
            for edge in zip(t["path"], t["path"][1:]):
                edges[edge] += amount
                for resource in graph.edge_resources[edge]:
                    resources[resource] += amount
        result.append((dict(edges), dict(resources)))
    return result


class PacketizationTests(unittest.TestCase):
    def checked(self, operator="all-gather", *, shared_capacity=None, budget=20):
        task, topology, graph, answer = fixture(operator, shared_capacity=shared_capacity)
        snapshots = deepcopy((task, topology, answer))
        baseline = replay(task, graph, answer)
        self.assertTrue(baseline["valid"], baseline["errors"])
        with patch("tests.packetization_experiment.replay", side_effect=AssertionError("planner invoked Replay")), \
                patch("tests.packetization_experiment.atoms", side_effect=AssertionError("planner scanned all atoms")):
            candidate, evidence = repacketize(task, graph, answer, baseline_report=baseline,
                                              max_extra_packets=budget)
        checked = replay(task, graph, candidate)
        self.assertTrue(checked["valid"], checked["errors"])
        self.assertFalse(evidence["replay_performed"])
        self.assertEqual((task, topology, answer), snapshots)
        self.assertEqual(atomic_rows(candidate), atomic_rows(answer))
        self.assertEqual(loads(candidate, task, graph), loads(answer, task, graph))
        self.assertAlmostEqual(evidence["before_us"], baseline["total_time_us"])
        self.assertAlmostEqual(evidence["predicted_after_us"], checked["total_time_us"])
        for predicted, actual in zip(evidence["predicted_round_times_us"], checked["round_times_us"]):
            self.assertAlmostEqual(predicted, actual)
        old_count = sum(len(r["transfers"]) for r in answer["rounds"])
        new_count = sum(len(r["transfers"]) for r in candidate["rounds"])
        self.assertEqual(evidence["before_transfer_count"], old_count)
        self.assertEqual(evidence["after_transfer_count"], new_count)
        self.assertEqual(evidence["actual_extra_packets"], new_count - old_count)
        self.assertEqual(evidence["extra_packets"], new_count - old_count)
        self.assertEqual(sum(p["extra_packets"] for p in evidence["plans"]), new_count - old_count)
        self.assertLessEqual(new_count - old_count, budget)
        return task, graph, answer, candidate, evidence, baseline, checked

    def test_allgather_full_replay_improves_path_cost(self):
        _, _, _, candidate, evidence, baseline, checked = self.checked()
        self.assertLess(checked["total_time_us"], baseline["total_time_us"])
        self.assertGreater(evidence["extra_packets"], 0)
        self.assertAlmostEqual(candidate["metrics"]["total_time"] * 1e6, checked["total_time_us"])
        self.assertEqual(candidate["metrics"]["round_times"], [t / 1e6 for t in checked["round_times_us"]])

    def test_allreduce_keeps_whole_multicontributor_expression(self):
        _, _, before, after, evidence, baseline, checked = self.checked("all-reduce")
        self.assertLess(checked["total_time_us"], baseline["total_time_us"])
        original_expressions = before["rounds"][1]["transfers"][0]["chunk_ranges"]
        sent_expressions = [token for t in after["rounds"][1]["transfers"] for token in t["chunk_ranges"]]
        self.assertEqual(sent_expressions, original_expressions)
        self.assertTrue(all("+" in token for token in sent_expressions))
        self.assertGreater(len(after["rounds"][1]["transfers"]), 1)
        self.assertEqual(evidence["actual_extra_packets"], 2)

    def test_overlapping_resources_and_repeated_bindings_keep_exact_loads(self):
        for operator in ("all-gather", "all-reduce"):
            with self.subTest(operator=operator):
                task, graph, before, after, _, baseline, checked = self.checked(operator, shared_capacity="100GB/s")
                self.assertLess(checked["total_time_us"], baseline["total_time_us"])
                for row, (_, resource_loads) in zip(before["rounds"], loads(after, task, graph)):
                    payload = sum(len(t["chunk_ranges"]) for t in row["transfers"]) * 8192
                    # Each path crosses two member edges; overlapping pools
                    # each receive both occurrences. A duplicate reference
                    # does not create a third binding.
                    self.assertEqual(resource_loads, {"overlap-a": 2 * payload, "overlap-b": 2 * payload})

    def test_resource_dominated_report_skips_load_reconstruction(self):
        task, _, graph, answer = fixture(shared_capacity="0.1GB/s")
        baseline = replay(task, graph, answer)
        with patch("tests.packetization_experiment._RoundLoad", side_effect=AssertionError("unnecessary load scan")):
            result, evidence = repacketize(task, graph, answer, baseline_report=baseline)
        self.assertIs(result, answer)
        self.assertEqual(evidence["rounds_skipped_by_replay"], 1)
        self.assertEqual(evidence["extra_packets"], 0)
        self.assertEqual(evidence["predicted_after_us"], baseline["total_time_us"])
        self.assertTrue(replay(task, graph, result)["valid"])

    def test_resource_dominated_without_report_is_unchanged(self):
        task, _, graph, answer = fixture(shared_capacity="0.1GB/s")
        result, evidence = repacketize(task, graph, answer)
        self.assertIs(result, answer)
        self.assertEqual(evidence["rounds_load_evaluated"], 1)
        self.assertAlmostEqual(evidence["predicted_after_us"], replay(task, graph, result)["total_time_us"])

    def test_budget_limits_and_zero_budget_skip_path_work(self):
        for budget in (0, 1, 2, 3, 20):
            with self.subTest(budget=budget):
                _, _, before, after, evidence, baseline, checked = self.checked(budget=budget)
                if budget < 2:
                    self.assertIs(after, before)
                    self.assertEqual(checked["total_time_us"], baseline["total_time_us"])
                else:
                    self.assertLess(checked["total_time_us"], baseline["total_time_us"])
        task, _, graph, answer = fixture()
        with patch("tests.packetization_experiment._PathCosts", side_effect=AssertionError("zero budget path work")):
            candidate, evidence = repacketize(task, graph, answer, max_extra_packets=0)
        self.assertIs(candidate, answer)
        self.assertIsNone(evidence["predicted_after_us"])

    def test_single_atoms_return_before_path_or_counter_work(self):
        task, _, graph, answer = fixture()
        for row in answer["rounds"]:
            row["transfers"] = [{**t, "chunk_ranges": [atom]}
                                for t in row["transfers"] for atom in t["chunk_ranges"]]
        baseline = replay(task, graph, answer)
        with patch("tests.packetization_experiment._PathCosts", side_effect=AssertionError("indivisible path work")):
            candidate, evidence = repacketize(task, graph, answer, baseline_report=baseline)
        self.assertIs(candidate, answer)
        self.assertEqual(evidence["reason"], "no_splittable_transfer")
        self.assertEqual(evidence["predicted_after_us"], baseline["total_time_us"])

    def test_prediction_uses_actual_resource_floor_not_target(self):
        task, _, graph, answer = fixture()
        t = answer["rounds"][0]["transfers"][0]
        costs = _PathCosts(graph)
        _, _, overhead, per_byte = costs.describe(t["path"])
        atom_time = per_byte * 8192
        target = overhead + 3.5 * atom_time
        resource_floor = overhead + 2 * atom_time
        plan = _plan_round([t], costs, 8192, target, overhead + 4 * atom_time, resource_floor)
        self.assertEqual(plan["extra_packets"], 1)
        self.assertAlmostEqual(plan["after_us"], resource_floor)
        self.assertLess(plan["after_us"], plan["target_us"])

    def test_existing_replay_times_preserve_empty_round_alignment(self):
        task, _, graph, answer = fixture()
        answer["rounds"][0]["round"] = 2
        answer["rounds"].insert(0, {"round": 1, "transfers": []})
        answer["round_count"] = 2
        original = deepcopy(answer)
        baseline = replay(task, graph, answer)
        self.assertTrue(baseline["valid"], baseline["errors"])
        candidate, evidence = repacketize(task, graph, answer, baseline_report=baseline)
        checked = replay(task, graph, candidate)
        self.assertTrue(checked["valid"], checked["errors"])
        self.assertEqual(evidence["predicted_round_times_us"][0], 0)
        self.assertAlmostEqual(evidence["predicted_round_times_us"][1], checked["round_times_us"][0])
        self.assertEqual(candidate["metrics"]["round_times"], [checked["round_times_us"][0] / 1e6])
        self.assertEqual(answer, original)

    def test_threshold_packet_count_is_minimal_and_matches_actual_slices(self):
        task, _, graph, answer = fixture()
        transfer = answer["rounds"][0]["transfers"][0]
        costs = _PathCosts(graph)
        _, _, overhead, per_byte = costs.describe(transfer["path"])
        for n in range(1, 33):
            for maximum in range(1, n + 1):
                t = {**transfer, "chunk_ranges": [f"atom-{i}" for i in range(n)]}
                target = overhead + per_byte * (maximum + 0.1)
                plan = _plan_round([t], costs, 1, target, overhead + n * per_byte, overhead)
                actual_packets = [t["chunk_ranges"][i:i + plan["sizes"][0]]
                                  for i in range(0, n, plan["sizes"][0])]
                count = plan["extra_packets"] + 1
                self.assertEqual(len(actual_packets), count)
                self.assertLessEqual(max(map(len, actual_packets)), maximum)
                self.assertLess((count - 1) * maximum, n)
                self.assertGreaterEqual(count * maximum, n)

    def test_optional_atomic_proof_detects_drops_and_preserves_inputs(self):
        task, topology, graph, answer = fixture("all-reduce")
        original = deepcopy((task, topology, answer))
        result, evidence = repacketize(task, graph, answer, verify_preservation=True)
        self.assertTrue(evidence["preservation_checked"])
        self.assertEqual((task, topology, answer), original)
        damaged = deepcopy(result)
        damaged["rounds"][0]["transfers"][0]["chunk_ranges"].pop()
        with self.assertRaisesRegex(ValueError, "data/path action"):
            verify_atomic_preservation(answer, damaged)

    def test_invalid_budgets_and_report_shape_are_rejected(self):
        task, _, graph, answer = fixture()
        for value in (-1, True, 1.5, math.inf, "2", None):
            with self.subTest(budget=value), self.assertRaises(ValueError):
                repacketize(task, graph, answer, max_extra_packets=value)
        for report in ({"valid": False}, {"valid": True, "round_times_us": [], "total_time_us": 0},
                       {"valid": True, "round_times_us": [math.nan], "total_time_us": math.nan}):
            with self.subTest(report=report), self.assertRaises(ValueError):
                repacketize(task, graph, answer, baseline_report=report)

    def test_offline_runner_records_hashes_and_external_replay_only(self):
        task, topology, _, answer = fixture()
        with tempfile.TemporaryDirectory(dir=PROJECT / "artifacts") as directory:
            archive = Path(directory)
            (archive / "inputs").mkdir()
            (archive / "v1.4").mkdir()
            request_file = archive / "inputs" / "synthetic-ag.json"
            response_file = archive / "v1.4" / "synthetic-ag-r01.json"
            request_file.write_text(json.dumps({"input": {"task": task, "topology": topology}}))
            response_file.write_text(json.dumps({"output": {"answer": answer}}))
            (archive / "summary.json").write_text(json.dumps({"rows": [
                {"version": "1.4", "sample": "synthetic/ag", "response_file": "v1.4/synthetic-ag-r01.json"}]}))
            original = {p: p.read_bytes() for p in (request_file, response_file, archive / "summary.json")}
            with redirect_stdout(io.StringIO()):
                run(archive, archive / "experiment.json", [0, 2])
            report = json.loads((archive / "experiment.json").read_text())
            self.assertEqual(report["baseline_replay_calls"], 1)
            self.assertEqual(report["candidate_replay_calls"], 2)
            self.assertTrue(report["source_unchanged"] and report["tool_unchanged"] and report["archive_files_unchanged"])
            tool = PROJECT / "tests/packetization_experiment.py"
            self.assertEqual(report["tool_sha256"]["tests/packetization_experiment.py"], hashlib.sha256(tool.read_bytes()).hexdigest())
            for row in report["rows"]:
                self.assertTrue(row["valid"] and row["replay_performed"] and row["inputs_unchanged"] and row["answer_unchanged"])
                self.assertEqual(row["extra_packets"], row["after_transfer_count"] - row["before_transfer_count"])
                self.assertAlmostEqual(row["predicted_after_us"], row["after_us"])
                self.assertEqual(row["source_input_file_sha256"], hashlib.sha256(original[request_file]).hexdigest())
                self.assertEqual(row["source_response_file_sha256"], hashlib.sha256(original[response_file]).hexdigest())
                self.assertEqual(len(row["after_answer_sha256"]), 64)
            self.assertEqual(original, {p: p.read_bytes() for p in original})


if __name__ == "__main__":
    unittest.main()
