"""Check legal whole-chunk transfer partitioning against published formulas.

Partitioning never changes the task's split, rounds, atomic chunks or routes.
It can reduce the path-serialization term, but cannot erase total edge/resource
load. These tests calculate expected times independently from the optimizer.
"""

from collections import Counter
from copy import deepcopy
import unittest

from src.replay import replay


def grouped_case(operator="all-gather", shared_resource=False):
    ranks, split, data_bytes = ["r0", "r1"], 4, 32768
    task = {
        "task_id": "transfer-partition", "instance_id": "transfer-partition",
        "operator": operator, "ranks": ranks,
        "data_size": {"per_rank_bytes": data_bytes},
        "split_policy": {"mode": "search_allowed", "candidates": [1, 2, 4], "alignment_bytes": 4096},
        "split": split, "max_nonempty_rounds": 4, "allow_copy": True,
    }
    topology = {
        "ranks": ranks, "switches": ["S"],
        "communication_startup_time_us": 1, "switch_processing_overhead_us": 0.3,
        "edges": [{"src": rank, "dst": "S", "bw": "1GB/s", "bidirectional": True} for rank in ranks],
        "resources": [],
    }
    if shared_resource:
        # Both directions of both edges consume this same pool.
        topology["resources"] = [{
            "id": "all-edges", "capacity": "1GB/s",
            "edges": [{"src": rank, "dst": "S", "bidirectional": True} for rank in ranks],
        }]
    own = {rank: [f"{rank}_c{slot}" for slot in range(split)] for rank in ranks}
    solution = {
        "operator": operator, "split": split,
        "chunks": [chunk for rank in ranks for chunk in own[rank]],
        "initial_possession": own,
        "final_chunks": [chunk for rank in ranks for chunk in own[rank]] if operator == "all-gather" else [f"c{slot}_sum" for slot in range(split)],
        "round_count": 1 if operator == "all-gather" else 2,
    }
    forward = {"action": "copy" if operator == "all-gather" else "reduce", "src": "r0", "dst": "r1", "path": ["r0", "S", "r1"], "chunk_ranges": own["r0"].copy()}
    backward = {"action": "copy", "src": "r1", "dst": "r0", "path": ["r1", "S", "r0"], "chunk_ranges": own["r1"].copy() if operator == "all-gather" else [f"c{slot}_sum" for slot in range(split)]}
    solution["rounds"] = [{"round": 1, "transfers": [forward, backward]}] if operator == "all-gather" else [
        {"round": 1, "transfers": [forward]}, {"round": 2, "transfers": [backward]},
    ]
    return task, topology, solution


def partition(solution, count):
    result = deepcopy(solution)
    for round_ in result["rounds"]:
        expanded = []
        for transfer in round_["transfers"]:
            for offset in range(count):
                tokens = transfer["chunk_ranges"][offset::count]
                if tokens:
                    expanded.append({**transfer, "chunk_ranges": tokens})
        round_["transfers"] = expanded
    return result


def directed_loads(solution, chunk_bytes):
    rounds = []
    for round_ in solution["rounds"]:
        load = Counter()
        for transfer in round_["transfers"]:
            amount = chunk_bytes * len(transfer["chunk_ranges"])
            for edge in zip(transfer["path"], transfer["path"][1:]):
                load[edge] += amount
        rounds.append(load)
    return rounds


class TransferPartitionTests(unittest.TestCase):
    def test_allgather_partition_reduces_path_term_but_preserves_edge_floor(self):
        task, topology, grouped = grouped_case()
        reports = []
        for count in [1, 2, 4]:
            candidate = partition(grouped, count)
            self.assertEqual(candidate["chunks"], grouped["chunks"])
            self.assertEqual(candidate["split"], grouped["split"])
            self.assertEqual(directed_loads(candidate, 8192), directed_loads(grouped, 8192))
            report = replay(task, topology, candidate)
            self.assertTrue(report["valid"], report)
            reports.append(report)
        # Grouped: 1 + .3 + 2 * 32768 / 1e9 * 1e6.
        self.assertAlmostEqual(reports[0]["total_time_us"], 66.836)
        # Two groups: path cost and aggregate edge cost meet at 34.068 us.
        self.assertAlmostEqual(reports[1]["total_time_us"], 34.068)
        # Four groups: the path term shrinks further, but edge load is unchanged.
        self.assertAlmostEqual(reports[2]["total_time_us"], 34.068)

    def test_allreduce_partition_keeps_whole_slot_expressions(self):
        task, topology, grouped = grouped_case("all-reduce")
        for count, expected in [(1, 133.672), (2, 68.136), (4, 68.136)]:
            with self.subTest(partitions=count):
                candidate = partition(grouped, count)
                self.assertEqual(directed_loads(candidate, 8192), directed_loads(grouped, 8192))
                report = replay(task, topology, candidate)
                self.assertTrue(report["valid"], report)
                self.assertAlmostEqual(report["total_time_us"], expected)

    def test_shared_resource_bottleneck_gives_no_partitioning_gain(self):
        for operator in ["all-gather", "all-reduce"]:
            task, topology, grouped = grouped_case(operator, shared_resource=True)
            base_report = replay(task, topology, grouped)
            for count in [2, 4]:
                with self.subTest(operator=operator, partitions=count):
                    report = replay(task, topology, partition(grouped, count))
                    self.assertTrue(report["valid"], report)
                    self.assertAlmostEqual(report["total_time_us"], base_report["total_time_us"])

    def test_partition_must_not_duplicate_a_chunk(self):
        for operator in ["all-gather", "all-reduce"]:
            task, topology, grouped = grouped_case(operator)
            candidate = partition(grouped, 4)
            candidate["rounds"][0]["transfers"].append(deepcopy(candidate["rounds"][0]["transfers"][0]))
            self.assertFalse(replay(task, topology, candidate)["valid"])

    def test_partition_must_not_invent_subchunk_ids(self):
        task, topology, grouped = grouped_case()
        candidate = partition(grouped, 4)
        candidate["rounds"][0]["transfers"][0]["chunk_ranges"] = ["r0_c0_part0", "r0_c0_part1"]
        report = replay(task, topology, candidate)
        self.assertFalse(report["valid"])
        self.assertIn("unknown AllGather chunk", report["errors"][0])

    def test_partition_must_not_split_a_reduced_contributor_expression(self):
        task, topology, grouped = grouped_case("all-reduce")
        # After reduction r1 holds {r0,r1} for slot 0. Neither original
        # contribution alone identifies the full value held there anymore.
        grouped["rounds"][1]["transfers"][0]["chunk_ranges"][0] = "r0_c0"
        report = replay(task, topology, grouped)
        self.assertFalse(report["valid"])
        self.assertIn("sender's start-of-round contributions", report["errors"][0])


if __name__ == "__main__":
    unittest.main()
