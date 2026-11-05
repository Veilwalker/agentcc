"""Failure regressions and V22-like input perturbations.

These are synthetic regressions, not a reconstruction of the unavailable
official inputs. They deliberately keep task IDs while changing graph values.
"""

from copy import deepcopy
import importlib
import importlib.util
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src import solver
from src.replay import replay
from src.topology import Topology, bandwidth
from tests.test_solver import make_case


PROJECT = Path(__file__).resolve().parents[1]
PUBLIC = PROJECT / "materials/spec/examples/256-rank"
BASELINE = PROJECT / "artifacts/baseline-1.0/src"


def load_baseline_solver():
    """Import the preserved implementation under an isolated module name."""
    name = "agentcc_baseline_v10_for_tests"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, BASELINE / "__init__.py", submodule_search_locations=[str(BASELINE)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return importlib.import_module(f"{name}.solver")


def perturb_topology(original: dict, seed: int, *, preserve_rank_connectivity: bool = False) -> tuple[dict, dict]:
    """Change capacity, direction, connectivity and overlapping resources.

    Resource references are reconciled with existing edges, as required for a
    valid input. Rank attachment edges are retained; redundant switch edges are
    removed in only one direction. No topology or resource name carries meaning.
    """
    rng = random.Random(seed)
    data = deepcopy(original)
    switches = set(data["switches"])
    directed = []
    for edge in data["edges"]:
        forward = {key: value for key, value in edge.items() if key != "bidirectional"}
        forward["bw"] = bandwidth(edge["bw"]) * rng.choice([0.4, 0.7, 1.3, 1.8])
        directed.append(forward)
        if edge.get("bidirectional", False):
            reverse = deepcopy(forward)
            reverse["src"], reverse["dst"] = edge["dst"], edge["src"]
            reverse["bw"] *= rng.choice([0.6, 1.0, 1.4])
            directed.append(reverse)
    switch_pairs = sorted({(edge["src"], edge["dst"]) for edge in directed
                           if edge["src"] in switches and edge["dst"] in switches})
    removed = set(rng.sample(switch_pairs, min(12, len(switch_pairs) // 20)))
    data["edges"] = [edge for edge in directed if (edge["src"], edge["dst"]) not in removed]
    existing = {(edge["src"], edge["dst"]) for edge in data["edges"]}
    added = []
    for _ in range(100):
        if len(added) >= 4 or len(switches) < 2:
            break
        src, dst = rng.sample(sorted(switches), 2)
        if (src, dst) in existing:
            continue
        data["edges"].append({"src": src, "dst": dst, "bw": "17GB/s"})
        existing.add((src, dst))
        added.append((src, dst))
    for resource in data.get("resources", []):
        resource["capacity"] = bandwidth(resource.get("capacity", resource.get("bw"))) * rng.choice([0.3, 0.6, 1.2])
        refs = set()
        for ref in resource.get("edges", []):
            pairs = [(ref["src"], ref["dst"])]
            if ref.get("bidirectional", False):
                pairs.append((ref["dst"], ref["src"]))
            refs.update(pair for pair in pairs if pair in existing)
        resource["edges"] = [{"src": src, "dst": dst} for src, dst in sorted(refs)]
    sampled = rng.sample(sorted(existing), min(50, len(existing)))
    for number, members in enumerate([sampled[:35], sampled[15:]]):
        data.setdefault("resources", []).append({
            "id": f"perturbation-{seed}-{number}", "capacity": "19GB/s",
            "edges": [{"src": src, "dst": dst} for src, dst in members],
        })
    restored = set()
    if preserve_rank_connectivity:
        # A sparse directed fabric may lose an entire domain after random edge
        # removal. Restore only enough removed directions to keep the test
        # feasible; a separate test below retains the disconnected example.
        def ranks_connected():
            adjacent, reverse = {}, {}
            for edge in data["edges"]:
                adjacent.setdefault(edge["src"], set()).add(edge["dst"])
                reverse.setdefault(edge["dst"], set()).add(edge["src"])
            for graph in (adjacent, reverse):
                seen, pending = {data["ranks"][0]}, [data["ranks"][0]]
                while pending:
                    for destination in graph.get(pending.pop(), ()):
                        if destination not in seen:
                            seen.add(destination)
                            pending.append(destination)
                if not set(data["ranks"]) <= seen:
                    return False
            return True

        for pair in sorted(removed):
            if ranks_connected():
                break
            data["edges"].extend(deepcopy(edge) for edge in directed if (edge["src"], edge["dst"]) == pair)
            restored.add(pair)
        if not ranks_connected():
            raise ValueError("original graph lacks strong rank reachability")
    return data, {"seed": seed, "removed_directed_edges": len(removed),
                  "added_directed_edges": len(added), "added_overlapping_resources": 2,
                  "restored_for_connectivity": len(restored)}


class SearchBudgetRegressionTests(unittest.TestCase):
    def run_expensive_model(self, implementation, operator):
        task, topology = make_case(8, operator)
        task["split_policy"]["candidates"] = [1, 2]
        clock = SimpleNamespace(now=1000.0)
        fake_time = SimpleNamespace(monotonic=lambda: clock.now)
        builder_name = "_allgather_doubling" if operator == "all-gather" else "_tree"
        original_builder = getattr(implementation, builder_name)
        model_attempts = []

        def expensive_model(task, graph, split, order, weight, deadline):
            if split == 2:
                model_attempts.append(split)
                # A legal model strategy may take longer than optimization time.
                clock.now += 9.0
                raise TimeoutError("injected expensive model candidate")
            return original_builder(task, graph, split, order, weight, deadline)

        hint = {"algorithm": "recursive_doubling" if operator == "all-gather" else "tree",
                "split": 2, "resource_weight": 2, "rank_order": "topology"}
        with patch.object(implementation, "time", fake_time), patch.object(implementation, builder_name, expensive_model):
            answer = implementation.solve_schedule(task, topology, hint, time_budget_s=8)
        self.assertTrue(model_attempts, "the model-selected candidate must actually be attempted")
        self.assertTrue(replay(task, topology, answer)["valid"])
        return answer

    def test_model_timeout_preserves_a_valid_allgather_answer(self):
        self.run_expensive_model(solver, "all-gather")

    def test_model_timeout_preserves_a_valid_allreduce_answer(self):
        self.run_expensive_model(solver, "all-reduce")

    @unittest.skipUnless(BASELINE.is_dir(), "preserved 1.0 implementation unavailable")
    def test_v10_reproduces_no_answer_when_first_model_candidate_exhausts_budget(self):
        old = load_baseline_solver()
        for operator in ("all-gather", "all-reduce"):
            with self.subTest(operator=operator):
                with self.assertRaisesRegex(old.SolverError, "no locally valid schedule"):
                    self.run_expensive_model(old, operator)


@unittest.skipUnless((PUBLIC / "cloudmatrix").is_dir(), "public 256-rank examples unavailable")
class MutatedPublicInputTests(unittest.TestCase):
    def test_cloudmatrix_changes_remain_solvable_for_both_operators(self):
        for operator, code in [("all-gather", "ag"), ("all-reduce", "ar")]:
            with self.subTest(operator=operator):
                base = PUBLIC / "cloudmatrix" / code
                task = json.loads((base / "task.json").read_text())
                original = json.loads((base / "topology.json").read_text())
                changed, info = perturb_topology(original, seed=11)
                self.assertEqual(changed["description"], original["description"])
                self.assertGreater(info["removed_directed_edges"], 0)
                graph = Topology.from_data(changed)
                self.assertTrue(any(len(ids) >= 2 for ids in graph.edge_resources.values()))
                hint = {"algorithm": "recursive_doubling" if code == "ag" else "tree",
                        "split": 32, "resource_weight": 2, "rank_order": "topology"}
                answer = solver.solve_schedule(task, changed, hint)
                report = replay(task, changed, answer)
                self.assertTrue(report["valid"], report)
                self.assertLessEqual(answer["round_count"], task["max_nonempty_rounds"])

    def test_disconnected_directed_mutation_is_rejected_instead_of_fabricating_paths(self):
        base = PUBLIC / "directed_ring" / "ag"
        task = json.loads((base / "task.json").read_text())
        original = json.loads((base / "topology.json").read_text())
        changed, _ = perturb_topology(original, seed=37)
        # Independent necessary reachability condition, permitting rank relays:
        # even this less restrictive raw graph cannot carry r0's data to r192.
        adjacency = {}
        for edge in changed["edges"]:
            adjacency.setdefault(edge["src"], set()).add(edge["dst"])
        seen, pending = {"r0"}, ["r0"]
        while pending:
            for destination in adjacency.get(pending.pop(), ()):
                if destination not in seen:
                    seen.add(destination)
                    pending.append(destination)
        self.assertNotIn("r192", seen)
        with self.assertRaisesRegex(solver.SolverError, "strongly connected"):
            solver.solve_schedule(task, changed)


class EquivalentRepresentationTests(unittest.TestCase):
    def test_equivalent_edge_resource_encodings_preserve_replay_time(self):
        for operator in ("all-gather", "all-reduce"):
            with self.subTest(operator=operator):
                task, topology = make_case(8, operator)
                topology["resources"] = [{
                    "id": "shared", "capacity": "1.5GB/s",
                    "edges": [{"src": rank, "dst": "S0", "bidirectional": True} for rank in task["ranks"][:4]],
                }]
                solution = solver.solve_schedule(task, topology, time_budget_s=0.1)
                original = replay(task, topology, solution)
                self.assertTrue(original["valid"], original)
                changed = deepcopy(topology)
                changed["edges"] = []
                for edge in topology["edges"]:
                    pairs = [(edge["src"], edge["dst"])]
                    if edge.get("bidirectional", False):
                        pairs.append((edge["dst"], edge["src"]))
                    # Two declarations with half capacity merge to one direction.
                    for src, dst in pairs:
                        for _ in range(2):
                            changed["edges"].append({"src": src, "dst": dst, "bw": bandwidth(edge["bw"]) / 2})
                for resource in changed["resources"]:
                    refs = []
                    for ref in resource["edges"]:
                        refs.extend([{"src": ref["src"], "dst": ref["dst"]},
                                     {"src": ref["dst"], "dst": ref["src"]}])
                    resource["edges"] = refs + deepcopy(refs)
                random.Random(19).shuffle(changed["edges"])
                random.Random(20).shuffle(changed["resources"][0]["edges"])
                equivalent = replay(task, changed, solution)
                self.assertTrue(equivalent["valid"], equivalent)
                self.assertAlmostEqual(equivalent["total_time_us"], original["total_time_us"], places=8)

    def test_same_round_transfer_order_does_not_change_state_or_time(self):
        for operator in ("all-gather", "all-reduce"):
            with self.subTest(operator=operator):
                task, topology = make_case(8, operator)
                solution = solver.solve_schedule(task, topology, time_budget_s=0.1)
                original = replay(task, topology, solution)
                shuffled = deepcopy(solution)
                rng = random.Random(21)
                for round_ in shuffled["rounds"]:
                    rng.shuffle(round_["transfers"])
                equivalent = replay(task, topology, shuffled)
                self.assertTrue(equivalent["valid"], equivalent)
                self.assertAlmostEqual(equivalent["total_time_us"], original["total_time_us"], places=8)


if __name__ == "__main__":
    unittest.main()
