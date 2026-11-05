"""Regression tests against cached answers or topology-name assumptions.

All variants deliberately retain the same task and topology identifiers. The
connections, directions and capacity fields must determine each new answer.
"""

from copy import deepcopy
import unittest

from src.replay import replay
from src.solver import solve_schedule
from src.topology import Topology


def two_plane_case():
    task = {
        "task_id": "same-task", "instance_id": "same-instance",
        "operator": "all-gather", "base_topology": "unchanged-family",
        "ranks": ["r0", "r1"], "data_size": {"per_rank_bytes": 32768},
        "split_policy": {"mode": "search_allowed", "candidates": [1], "alignment_bytes": 4096},
        "split": 1, "max_nonempty_rounds": 4, "allow_copy": True,
    }
    topology = {
        "description": "unchanged-family", "ranks": ["r0", "r1"],
        "switches": ["fast", "slow"],
        "communication_startup_time_us": 1, "switch_processing_overhead_us": 0.3,
        "edges": [
            {"src": rank, "dst": plane, "bw": bandwidth, "bidirectional": True}
            for plane, bandwidth in [("fast", "10GB/s"), ("slow", "1GB/s")]
            for rank in ("r0", "r1")
        ],
        "resources": [],
    }
    return task, topology


def paths(answer):
    return [transfer["path"] for round_ in answer["rounds"] for transfer in round_["transfers"]]


def pool(plane, capacity="0.01GB/s"):
    return {
        "id": "same-pool", "capacity": capacity,
        "edges": [{"src": rank, "dst": plane, "bidirectional": True} for rank in ("r0", "r1")],
    }


class InputChangeTests(unittest.TestCase):
    def solve_valid(self, task, topology):
        answer = solve_schedule(task, topology, time_budget_s=5)
        result = replay(task, topology, answer)
        self.assertTrue(result["valid"], result)
        return answer, result

    def test_same_name_bandwidth_changes_recompute_routes(self):
        task, original = two_plane_case()
        old, _ = self.solve_valid(task, original)
        self.assertTrue(all("fast" in path for path in paths(old)))
        changed = deepcopy(original)
        for edge in changed["edges"]:
            if edge["dst"] == "fast":
                edge["bw"] = "0.1GB/s"
        new, _ = self.solve_valid(task, changed)
        self.assertTrue(all("slow" in path for path in paths(new)))
        self.assertNotEqual(paths(old), paths(new))

    def test_same_name_resource_capacity_changes_recompute_routes(self):
        task, original = two_plane_case()
        original["resources"] = [pool("fast")]
        old, _ = self.solve_valid(task, original)
        self.assertTrue(all("slow" in path for path in paths(old)))
        changed = deepcopy(original)
        changed["resources"][0]["capacity"] = "100GB/s"
        new, _ = self.solve_valid(task, changed)
        self.assertTrue(all("fast" in path for path in paths(new)))

    def test_same_resource_id_new_membership_recomputes_routes(self):
        task, original = two_plane_case()
        original["resources"] = [pool("fast")]
        old, _ = self.solve_valid(task, original)
        self.assertTrue(all("slow" in path for path in paths(old)))
        changed = deepcopy(original)
        changed["resources"] = [pool("slow")]
        new, _ = self.solve_valid(task, changed)
        self.assertTrue(all("fast" in path for path in paths(new)))

    def test_deleted_edge_invalidates_old_solution(self):
        task, original = two_plane_case()
        old, _ = self.solve_valid(task, original)
        changed = deepcopy(original)
        changed["edges"] = [edge for edge in changed["edges"] if edge["dst"] != "fast"]
        stale_report = replay(task, changed, old)
        self.assertFalse(stale_report["valid"])
        self.assertIn("undeclared directed edge", stale_report["errors"][0])
        new, _ = self.solve_valid(task, changed)
        self.assertTrue(all("slow" in path for path in paths(new)))

    def test_added_edges_can_be_used_without_a_new_topology_name(self):
        task, original = two_plane_case()
        old, old_report = self.solve_valid(task, original)
        changed = deepcopy(original)
        changed["switches"].append("new-plane")
        changed["edges"].extend(
            {"src": rank, "dst": "new-plane", "bw": "100GB/s", "bidirectional": True}
            for rank in ("r0", "r1")
        )
        new, new_report = self.solve_valid(task, changed)
        self.assertTrue(all("new-plane" in path for path in paths(new)))
        self.assertLess(new_report["total_time_us"], old_report["total_time_us"])
        self.assertNotEqual(paths(old), paths(new))

    def test_direction_changes_invalidate_only_the_affected_routes(self):
        task, original = two_plane_case()
        old, _ = self.solve_valid(task, original)
        changed = deepcopy(original)
        for edge in changed["edges"]:
            if edge["src"] == "r1" and edge["dst"] == "fast":
                edge["bidirectional"] = False
        self.assertFalse(replay(task, changed, old)["valid"])
        new, _ = self.solve_valid(task, changed)
        for path in paths(new):
            if path[-1] == "r1":
                self.assertIn("slow", path)
        graph = Topology.from_data(changed)
        self.assertIn(("r1", "fast"), graph.edges)
        self.assertNotIn(("fast", "r1"), graph.edges)

    def test_shared_capacity_affects_time_without_changing_reachability(self):
        task, topology = two_plane_case()
        answer, before = self.solve_valid(task, topology)
        topology["resources"] = [pool("fast", "0.0001GB/s")]
        after = replay(task, topology, answer)
        self.assertTrue(after["valid"], after)
        self.assertGreater(after["total_time_us"], before["total_time_us"] * 100)

    def test_resource_reference_does_not_resurrect_deleted_edge(self):
        _, topology = two_plane_case()
        topology["resources"] = [pool("fast")]
        topology["edges"] = [edge for edge in topology["edges"] if edge["dst"] != "fast"]
        with self.assertRaisesRegex(ValueError, "unknown directed edge"):
            Topology.from_data(topology)


if __name__ == "__main__":
    unittest.main()
