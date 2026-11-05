"""Semantic and scale checks for constructed schedules, using local replay."""

import copy
import unittest
from unittest.mock import patch

from src.replay import replay
from src.solver import SolverError, solve_schedule, validate_task
from src import solver
from src.topology import Topology


def make_case(n=4, operator="all-gather", direct=False):
    ranks = [f"r{i}" for i in range(n)]
    task = {
        "task_id": "synthetic-test", "instance_id": "synthetic-test",
        "operator": operator, "ranks": ranks, "npu": ranks,
        "data_size": {"per_rank_bytes": 32768},
        "split_policy": {"mode": "search_allowed", "candidates": [1, 2, 4], "alignment_bytes": 4096},
        "split": 1, "max_nonempty_rounds": 64, "allow_copy": True,
        "communication_constraints": {
            "receive_unique_chunk_once": True, "send_only_possessed_chunks": True,
            "send_unique_chunk_pair_once": True, "switches_forward_same_round_without_storage": True,
        },
    }
    topology = {
        "ranks": ranks, "switches": [] if direct else ["S0"],
        "communication_startup_time_us": 1, "switch_processing_overhead_us": 0.3,
        "edges": [{"src": rank, "dst": "S0", "bw": "1GB/s", "bidirectional": True}
                  for rank in ranks] if not direct else [
                      {"src": ranks[i], "dst": ranks[i + 1], "bw": "1GB/s", "bidirectional": True}
                      for i in range(n - 1)],
        "resources": [],
    }
    return task, topology


class SolverTests(unittest.TestCase):
    def assert_valid(self, task, topology, strategy=None):
        answer = solve_schedule(task, topology, strategy, time_budget_s=15)
        report = replay(task, topology, answer)
        self.assertTrue(report["valid"], report["errors"])
        self.assertLessEqual(answer["round_count"], task["max_nonempty_rounds"])
        self.assertAlmostEqual(answer["metrics"]["total_time"], report["total_time_us"] / 1e6)
        return answer

    def test_both_operators_power_of_two(self):
        for operator in ["all-gather", "all-reduce"]:
            with self.subTest(operator=operator):
                self.assert_valid(*make_case(8, operator))

    def test_two_rank_allreduce_completes_all_contributions(self):
        answer = self.assert_valid(*make_case(2, "all-reduce"))
        self.assertEqual(answer["round_count"], 2)
        self.assertEqual(answer["rounds"][0]["transfers"][0]["action"], "reduce")
        self.assertEqual(answer["rounds"][1]["transfers"][0]["action"], "copy")

    def test_uneven_rank_counts(self):
        for n in [1, 3, 5, 9]:
            for operator in ["all-gather", "all-reduce"]:
                with self.subTest(n=n, operator=operator):
                    self.assert_valid(*make_case(n, operator), {"algorithm": "tree", "split": 4})

    def test_rank_relay_uses_separate_rounds(self):
        for operator in ["all-gather", "all-reduce"]:
            with self.subTest(operator=operator):
                task, topology = make_case(5, operator, direct=True)
                answer = self.assert_valid(task, topology)
                self.assertEqual(answer["strategy"]["algorithm"], "rank_relay")
                for round_ in answer["rounds"]:
                    for transfer in round_["transfers"]:
                        self.assertEqual(len(transfer["path"]), 2)

    def test_directed_cycle_relay(self):
        for operator in ["all-gather", "all-reduce"]:
            task, topology = make_case(4, operator, direct=True)
            topology["edges"] = [
                {"src": f"r{i}", "dst": f"r{(i + 1) % 4}", "bw": "1GB/s"}
                for i in range(4)]
            self.assert_valid(task, topology)

    def test_disconnected_graph_is_rejected(self):
        task, topology = make_case()
        topology["edges"] = topology["edges"][:2]
        with self.assertRaises(SolverError):
            solve_schedule(task, topology)

    def test_invalid_hint_cannot_change_task(self):
        task, topology = make_case()
        task_before, topology_before = copy.deepcopy(task), copy.deepcopy(topology)
        answer = self.assert_valid(task, topology, {"split": 123, "algorithm": "execute_code",
                                                   "resource_weight": -1, "path": ["made-up"]})
        self.assertIn(answer["split"], task["split_policy"]["candidates"])
        self.assertEqual(task, task_before)
        self.assertEqual(topology, topology_before)

    def test_model_candidate_is_constructed_and_replayed_first(self):
        task, topology = make_case(8, "all-reduce")
        hint = {"algorithm": "tree", "split": 4, "resource_weight": 2.0, "rank_order": "topology"}
        answer = self.assert_valid(task, topology, hint)
        evidence = answer["strategy"]
        self.assertEqual(evidence["model_requested"], hint)
        self.assertEqual(evidence["model_adopted"], hint)
        first = evidence["candidates"][0]
        self.assertEqual(first["source"], "model")
        self.assertEqual(first["actual"]["split"], 4)
        self.assertTrue(first["valid"])
        self.assertEqual(sum(record["selected"] for record in evidence["candidates"]), 1)

    def test_model_timeout_retries_compact_model_parameters(self):
        task, topology = make_case(8)
        hint = {"algorithm": "recursive_doubling", "split": 4,
                "resource_weight": 2.0, "rank_order": "topology"}
        real_builder = solver._allgather_doubling
        deadlines = []

        def timeout_once(*args, **kwargs):
            deadlines.append(args[-1])
            if len(deadlines) == 1:
                raise TimeoutError("simulated first model candidate timeout")
            return real_builder(*args, **kwargs)

        with patch.object(solver, "_allgather_doubling", side_effect=timeout_once):
            answer = self.assert_valid(task, topology, hint)
        records = answer["strategy"]["candidates"]
        self.assertEqual(records[0]["source"], "model")
        self.assertFalse(records[0]["valid"])
        compact = records[1]
        self.assertEqual(compact["source"], "model_compact")
        self.assertTrue(compact["valid"])
        self.assertEqual(compact["actual"], {**hint, "split": 1})
        self.assertIn("reason", compact)
        self.assertGreater(deadlines[1], deadlines[0])

    def test_alignment_and_rank_mismatch_rejected(self):
        task, topology = make_case()
        task["split_policy"]["candidates"] = [3]
        with self.assertRaises(SolverError):
            solve_schedule(task, topology)
        task, topology = make_case()
        task["ranks"] = task["ranks"][::-1]
        with self.assertRaises(SolverError):
            solve_schedule(task, topology)

    def test_task_validation_rejects_boolean_byte_counts_and_alignment(self):
        for field in ["per_rank_bytes", "alignment_bytes"]:
            with self.subTest(field=field):
                task, topology = make_case()
                container = task["data_size"] if field == "per_rank_bytes" else task["split_policy"]
                container[field] = True
                with self.assertRaises(SolverError):
                    validate_task(task, Topology.from_data(topology))

    def test_task_validation_rejects_malformed_candidates(self):
        for candidates in [[], None, "1,2", [True, 2], [0, 2], [-1, 2], [1.0, 2]]:
            with self.subTest(candidates=candidates):
                task, topology = make_case()
                task["split_policy"]["candidates"] = candidates
                with self.assertRaises(SolverError):
                    validate_task(task, Topology.from_data(topology))

    def test_task_validation_rejects_invalid_round_limits(self):
        for rounds in [True, False, -1, 1.5, "64", 0]:
            with self.subTest(rounds=rounds):
                task, topology = make_case()
                task["max_nonempty_rounds"] = rounds
                with self.assertRaises(SolverError):
                    validate_task(task, Topology.from_data(topology))
        task, topology = make_case(1)
        task["max_nonempty_rounds"] = 0
        self.assertEqual(validate_task(task, Topology.from_data(topology)), [1, 2, 4])

    def test_task_validation_rejects_inconsistent_rank_alias_and_flag_types(self):
        for field, value in [("npu", ["r0"]), ("ranks", "r0"), ("operator", "broadcast"),
                             ("allow_copy", "true"), ("allow_detour", 1),
                             ("communication_constraints", [])]:
            with self.subTest(field=field):
                task, topology = make_case()
                task[field] = value
                with self.assertRaises(SolverError):
                    validate_task(task, Topology.from_data(topology))

    def test_task_validation_filters_unaligned_candidates_without_mutation(self):
        task, topology = make_case()
        task["split_policy"]["candidates"] = [1, 3, 4]
        original = copy.deepcopy(task)
        self.assertEqual(validate_task(task, Topology.from_data(topology)), [1, 4])
        self.assertEqual(task, original)

    def test_fixed_split_must_be_permitted_and_aligned(self):
        task, topology = make_case()
        task["split_policy"]["mode"] = "fixed"
        task["split"] = 4
        graph = Topology.from_data(topology)
        self.assertEqual(validate_task(task, graph), [4])
        task["split"] = 3
        with self.assertRaises(SolverError):
            validate_task(task, graph)

    def test_256_rank_both_operators(self):
        for operator in ["all-gather", "all-reduce"]:
            with self.subTest(operator=operator):
                task, topology = make_case(256, operator)
                task["data_size"]["per_rank_bytes"] = 1048576 if operator == "all-gather" else 2097152
                task["split_policy"]["candidates"] = [1, 2, 4, 8, 16, 32]
                answer = self.assert_valid(task, topology)
                self.assertLessEqual(answer["round_count"], 16)
                if operator == "all-gather":
                    hierarchical = [record for record in answer["strategy"]["candidates"]
                                    if record["actual"]["algorithm"].startswith("hierarchical_")]
                    self.assertEqual(len(hierarchical), 2)
                    self.assertTrue(all(record["valid"] for record in hierarchical), hierarchical)


if __name__ == "__main__":
    unittest.main()
