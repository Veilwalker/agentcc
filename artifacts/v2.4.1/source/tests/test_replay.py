"""Examples and adversarial checks for the independent symbolic replay."""

from copy import deepcopy
import unittest

from src.replay import replay


def shared_fixture(operator="all-gather"):
    task = {
        "task_id": "shared-demo", "instance_id": "shared-demo",
        "operator": operator, "ranks": ["r0", "r1"],
        "data_size": {"per_rank_bytes": 32768},
        "split_policy": {"candidates": [1, 2], "alignment_bytes": 4096},
        "split": 1, "max_nonempty_rounds": 4, "allow_copy": True,
    }
    topology = {
        "ranks": ["r0", "r1"], "switches": ["S0"],
        "communication_startup_time_us": 1,
        "switch_processing_overhead_us": 0.3,
        "edges": [
            {"src": "r0", "dst": "S0", "bw": "1GB/s", "bidirectional": True},
            {"src": "r1", "dst": "S0", "bw": "2GB/s", "bidirectional": True},
        ],
        "resources": [
            {"id": "ingress", "capacity": "1GB/s", "edges": [{"src": "r0", "dst": "S0"}, {"src": "r1", "dst": "S0"}]},
            {"id": "egress", "capacity": "1GB/s", "edges": [{"src": "S0", "dst": "r0"}, {"src": "S0", "dst": "r1"}]},
        ],
    }
    solution = {
        "operator": operator,
        "chunks": ["r0_c0", "r1_c0"],
        "initial_possession": {"r0": ["r0_c0"], "r1": ["r1_c0"]},
        "final_chunks": ["r0_c0", "r1_c0"] if operator == "all-gather" else ["c0_sum"],
        "round_count": 1 if operator == "all-gather" else 2,
        "rounds": [{"round": 1, "transfers": [{
            "action": "copy" if operator == "all-gather" else "reduce",
            "src": "r0", "dst": "r1", "chunk_ranges": ["r0_c0"],
            "path": ["r0", "S0", "r1"],
        }]}],
    }
    backward = {
        "action": "copy", "src": "r1", "dst": "r0",
        "chunk_ranges": ["r1_c0" if operator == "all-gather" else "r0_c0+r1_c0"],
        "path": ["r1", "S0", "r0"],
    }
    if operator == "all-gather":
        solution["rounds"][0]["transfers"].append(backward)
    else:
        solution["rounds"].append({"round": 2, "transfers": [backward]})
    return task, topology, solution


def three_rank_fixture():
    task, topology, solution = shared_fixture("all-reduce")
    task["ranks"].append("r2")
    topology["ranks"].append("r2")
    topology["edges"].append({"src": "r2", "dst": "S0", "bw": "1GB/s", "bidirectional": True})
    topology["resources"] = []
    solution["chunks"].append("r2_c0")
    solution["initial_possession"]["r2"] = ["r2_c0"]
    solution["rounds"] = [
        {"round": 1, "transfers": [
            {"action": "reduce", "src": rank, "dst": "r2", "chunk_ranges": [f"{rank}_c0"], "path": [rank, "S0", "r2"]}
            for rank in ("r0", "r1")
        ]},
        {"round": 2, "transfers": [
            {"action": "copy", "src": "r2", "dst": rank, "chunk_ranges": ["c0_sum"], "path": ["r2", "S0", rank]}
            for rank in ("r0", "r1")
        ]},
    ]
    return task, topology, solution


class ReplayTests(unittest.TestCase):
    def assertInvalid(self, task, topology, solution, fragment):
        report = replay(task, topology, solution)
        self.assertFalse(report["valid"], report)
        self.assertIn(fragment, report["errors"][0])

    def test_official_small_shared_resource_example(self):
        report = replay(*shared_fixture())
        self.assertTrue(report["valid"], report)
        self.assertAlmostEqual(report["total_time_us"], 66.836, places=6)

    def test_reduce_then_copy_complete_result_overwrites_partial(self):
        report = replay(*shared_fixture("all-reduce"))
        self.assertTrue(report["valid"], report)
        self.assertAlmostEqual(report["total_time_us"], 100.904, places=6)

    def test_complete_sum_alias(self):
        task, topology, solution = shared_fixture("all-reduce")
        solution["rounds"][1]["transfers"][0]["chunk_ranges"] = ["c0_sum"]
        self.assertTrue(replay(task, topology, solution)["valid"])

    def test_allreduce_cannot_forward_same_round_result(self):
        task, topology, solution = shared_fixture("all-reduce")
        solution["rounds"][0]["transfers"] += solution["rounds"].pop()["transfers"]
        solution["round_count"] = 1
        self.assertInvalid(task, topology, solution, "start-of-round")

    def test_partial_copy_is_not_valid_allreduce(self):
        task, topology, solution = shared_fixture("all-reduce")
        solution["rounds"][0]["transfers"][0]["action"] = "copy"
        self.assertInvalid(task, topology, solution, "complete reduction")

    def test_duplicate_receive_in_same_round(self):
        task, topology, solution = shared_fixture()
        solution["rounds"][0]["transfers"].append(deepcopy(solution["rounds"][0]["transfers"][0]))
        self.assertInvalid(task, topology, solution, "duplicated chunk")

    def test_undeclared_reverse_direction(self):
        task, topology, solution = shared_fixture()
        topology["edges"][0]["bidirectional"] = False
        topology["resources"] = []
        self.assertInvalid(task, topology, solution, "undeclared directed edge")

    def test_path_cannot_revisit_switch(self):
        task, topology, solution = shared_fixture()
        solution["rounds"][0]["transfers"][0]["path"] = ["r0", "S0", "S0", "r1"]
        self.assertInvalid(task, topology, solution, "repeats a node")

    def test_path_cannot_use_third_rank_as_interior(self):
        task, topology, solution = shared_fixture()
        task["ranks"].append("r2")
        topology["ranks"].append("r2")
        solution["chunks"].append("r2_c0")
        solution["initial_possession"]["r2"] = ["r2_c0"]
        solution["final_chunks"].append("r2_c0")
        solution["rounds"][0]["transfers"][0]["path"] = ["r0", "r2", "r1"]
        self.assertInvalid(task, topology, solution, "only declared switches")

    def test_resource_counts_each_bound_edge_on_path(self):
        task, topology, solution = shared_fixture()
        topology["resources"] = [{
            "id": "shared", "capacity": "1GB/s",
            "edges": [{"src": "r0", "dst": "S0", "bidirectional": True}, {"src": "r1", "dst": "S0", "bidirectional": True}],
        }]
        report = replay(task, topology, solution)
        self.assertTrue(report["valid"], report)
        self.assertAlmostEqual(report["total_time_us"], 132.372, places=6)

    def test_resource_overlap_is_not_load_splitting(self):
        task, topology, solution = shared_fixture()
        extra = deepcopy(topology["resources"][0])
        extra.update(id="slow-overlap", capacity="0.5GB/s")
        topology["resources"].append(extra)
        report = replay(task, topology, solution)
        self.assertTrue(report["valid"], report)
        self.assertAlmostEqual(report["total_time_us"], 132.372, places=6)

    def test_alignment_is_enforced(self):
        task, topology, solution = shared_fixture()
        task["data_size"]["per_rank_bytes"] = 32769
        self.assertInvalid(task, topology, solution, "alignment")

    def test_incomplete_final_state_rejected(self):
        task, topology, solution = shared_fixture()
        solution["rounds"][0]["transfers"].pop()
        self.assertInvalid(task, topology, solution, "final state is incomplete")

    def test_nonempty_round_limit_and_empty_round_removal(self):
        task, topology, solution = shared_fixture()
        task["max_nonempty_rounds"] = 1
        solution["rounds"].insert(0, {"round": 1, "transfers": []})
        solution["rounds"][1]["round"] = 2
        solution["round_count"] = 2
        self.assertTrue(replay(task, topology, solution)["valid"])
        task["max_nonempty_rounds"] = 0
        self.assertInvalid(task, topology, solution, "too many nonempty")

    def test_response_task_binding(self):
        task, topology, solution = shared_fixture()
        response = {"task_id": "wrong", "status": "success", "output": {"answer": solution}}
        self.assertInvalid(task, topology, response, "task_id")

    def test_reduce_expression_cannot_repeat_contributor(self):
        task, topology, solution = shared_fixture("all-reduce")
        solution["rounds"][1]["transfers"][0]["chunk_ranges"] = ["r0_c0+r1_c0+r0_c0"]
        self.assertInvalid(task, topology, solution, "repeats a contributor")

    def test_multiple_disjoint_reduces_to_same_slot_in_one_round(self):
        report = replay(*three_rank_fixture())
        self.assertTrue(report["valid"], report)
        # The complete three-contributor result is still one chunk per transfer.
        self.assertAlmostEqual(report["total_time_us"], 133.672, places=6)

    def test_overlapping_same_round_reduces_rejected(self):
        task, topology, solution = three_rank_fixture()
        solution["rounds"][0]["transfers"] = [{
            "action": "reduce", "src": "r0", "dst": "r1", "chunk_ranges": ["r0_c0"], "path": ["r0", "S0", "r1"],
        }]
        solution["rounds"][1]["transfers"] = [
            {"action": "reduce", "src": "r1", "dst": "r2", "chunk_ranges": ["r0_c0+r1_c0"], "path": ["r1", "S0", "r2"]},
            {"action": "reduce", "src": "r0", "dst": "r2", "chunk_ranges": ["r0_c0"], "path": ["r0", "S0", "r2"]},
        ]
        self.assertInvalid(task, topology, solution, "overlapping contributions")

    def test_allgather_cannot_forward_same_round_result(self):
        task, topology, solution = three_rank_fixture()
        task["operator"] = solution["operator"] = "all-gather"
        solution["final_chunks"] = solution["chunks"].copy()
        solution["round_count"] = 1
        solution["rounds"] = [{"round": 1, "transfers": [
            {"action": "copy", "src": "r0", "dst": "r1", "chunk_ranges": ["r0_c0"], "path": ["r0", "S0", "r1"]},
            {"action": "copy", "src": "r1", "dst": "r2", "chunk_ranges": ["r0_c0"], "path": ["r1", "S0", "r2"]},
        ]}]
        self.assertInvalid(task, topology, solution, "start of the round")

    def test_multiple_slots_charge_the_actual_chunk_size(self):
        task, topology, solution = shared_fixture("all-reduce")
        solution["split"] = 2
        solution["chunks"] = [f"r{rank}_c{slot}" for rank in range(2) for slot in range(2)]
        solution["initial_possession"] = {f"r{rank}": [f"r{rank}_c{slot}" for slot in range(2)] for rank in range(2)}
        solution["final_chunks"] = ["c0_sum", "c1_sum"]
        solution["rounds"][0]["transfers"][0]["chunk_ranges"] = ["r0_c0", "r0_c1"]
        solution["rounds"][1]["transfers"][0]["chunk_ranges"] = ["c0_sum", "c1_sum"]
        report = replay(task, topology, solution)
        self.assertTrue(report["valid"], report)
        self.assertAlmostEqual(report["total_time_us"], 100.904, places=6)
        task["split_policy"]["mode"] = "fixed"
        self.assertInvalid(task, topology, solution, "fixed split")


if __name__ == "__main__":
    unittest.main()
