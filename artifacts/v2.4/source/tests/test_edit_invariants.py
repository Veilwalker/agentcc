"""Independent adversarial checks for the bounded model-edit compiler.

These tests use synthetic proposals and never call a model or load credentials.
Correctness is checked against the symbolic Replay and conserved data actions,
not against the compiler's own projected resource deltas.
"""

from collections import Counter
from copy import deepcopy
from itertools import product
import unittest

from src.collectives import build
from src.replay import replay
from src.topology import Topology
from tests.test_route_balance import parallel_case
from tests.test_solver import make_case


def actions(solution):
    """Expand only whole atomic tokens; paths and packet grouping may change."""
    return Counter(
        (round_["round"], transfer["action"], transfer["src"], transfer["dst"], token)
        for round_ in solution["rounds"]
        for transfer in round_["transfers"]
        for token in transfer["chunk_ranges"]
    )


class EditInvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Deferred until execution so collecting unrelated tests does not import
        # or initialize any model adapter.
        from src.schedule_edit import apply_proposal, extract_problem
        cls.apply = staticmethod(apply_proposal)
        cls.extract = staticmethod(extract_problem)

    def fixture(self, operator="all-gather"):
        task, graph = parallel_case(operator)
        kind = "reverse_doubling" if operator == "all-gather" else "xor_tree"
        baseline = build(task, graph, 4, kind=kind, resource_weight=0, packets=2)
        report = replay(task, graph, baseline)
        self.assertTrue(report["valid"], report)
        baseline["metrics"] = {"total_time": report["total_time_us"] / 1e6,
                               "round_times": [value / 1e6 for value in report["round_times_us"]]}
        problem, context = self.extract(task, graph, baseline, time_budget_s=1,
                                        max_groups=8, max_rounds=8)
        self.assertTrue(problem["groups"], "fixture must expose a real route-edit opportunity")
        return task, graph, baseline, problem, context

    def proposal(self, problem, fraction=1):
        return {"base": problem["base"], "edits": [
            [group["id"], int(group["max_chunks"] * fraction)] for group in problem["groups"]]}

    def assert_safe_result(self, task, graph, baseline, result, evidence):
        before, after = replay(task, graph, baseline), replay(task, graph, result)
        self.assertTrue(after["valid"], after)
        self.assertEqual(actions(result), actions(baseline))
        for field in ("operator", "split", "chunks", "initial_possession", "final_chunks", "round_count"):
            self.assertEqual(result[field], baseline[field], field)
        self.assertLessEqual(after["total_time_us"], before["total_time_us"])
        if evidence["accepted"]:
            self.assertIsNot(result, baseline)
            self.assertLess(after["total_time_us"], before["total_time_us"])
            self.assertAlmostEqual(result["metrics"]["total_time"] * 1e6, after["total_time_us"])
            self.assertEqual(len(result["metrics"]["round_times"]), len(after["round_times_us"]))
            for metric, actual in zip(result["metrics"]["round_times"], after["round_times_us"]):
                self.assertAlmostEqual(metric * 1e6, actual)
        else:
            self.assertIs(result, baseline)

    def test_atomic_data_actions_and_ar_contributors_survive_partial_edits(self):
        for operator in ("all-gather", "all-reduce"):
            with self.subTest(operator=operator):
                task, graph, baseline, problem, context = self.fixture(operator)
                original = deepcopy(baseline)
                if operator == "all-reduce":
                    self.assertTrue(any("+" in item[-1] for item in actions(baseline)))
                accepted = False
                for fraction in (0.25, 0.5, 0.75, 1):
                    result, evidence = self.apply(task, graph, baseline, context, self.proposal(problem, fraction))
                    self.assert_safe_result(task, graph, baseline, result, evidence)
                    accepted |= evidence["accepted"]
                self.assertTrue(accepted, "a useful local edit must be executable on this fixture")
                self.assertEqual(baseline, original)

    def test_all_joint_prefix_choices_are_replayed_and_never_worsen(self):
        task, graph, baseline, problem, context = self.fixture("all-reduce")
        selected = problem["groups"][:3]
        choices = [sorted({0, group["max_chunks"] // 2, group["max_chunks"]}) for group in selected]
        for counts in product(*choices):
            with self.subTest(counts=counts):
                proposal = {"base": problem["base"], "edits": [
                    [group["id"], count] for group, count in zip(selected, counts)]}
                result, evidence = self.apply(task, graph, baseline, context, proposal)
                self.assert_safe_result(task, graph, baseline, result, evidence)

    def test_overlap_and_repeated_resource_membership_are_not_divided(self):
        task, graph, baseline, problem, context = self.fixture()
        # Each fast path uses two edges belonging to BOTH pools. Each pool must
        # therefore be charged twice the transfer payload, independently.
        round_ = baseline["rounds"][-1]
        chunk_bytes = task["data_size"]["per_rank_bytes"] // baseline["split"]
        total = sum(len(transfer["chunk_ranges"]) * chunk_bytes for transfer in round_["transfers"])
        expected = graph.startup_us + graph.switch_overhead_us + 1e6 * 2 * total / graph.resources["shared-a"]
        self.assertAlmostEqual(replay(task, graph, baseline)["round_times_us"][-1], expected)
        result, evidence = self.apply(task, graph, baseline, context, self.proposal(problem, 0.5))
        self.assert_safe_result(task, graph, baseline, result, evidence)

    def test_valid_but_overloaded_joint_edit_is_rejected(self):
        task, graph = parallel_case()
        # A lightly loaded alternate pool is useful for a small amount of data,
        # but moving every offered chunk there creates a worse bottleneck.
        graph.resources["alternate-pool"] = 0.5e9
        for edge in graph.edge_resources:
            if "alternate" in edge:
                graph.edge_resources[edge] = ("alternate-pool",)
        baseline = build(task, graph, 4, resource_weight=0, packets=2)
        before = replay(task, graph, baseline)
        problem, context = self.extract(task, graph, baseline, time_budget_s=1, max_rounds=8)
        self.assertTrue(problem["groups"])
        result, evidence = self.apply(task, graph, baseline, context, self.proposal(problem))
        self.assertIs(result, baseline)
        self.assertFalse(evidence["accepted"])
        self.assertTrue(evidence["replay_valid"], "the rejection must be about cost, not invalid data")
        self.assertGreater(evidence["proposal_time_us"], before["total_time_us"])
        self.assertEqual(evidence["after_us"], before["total_time_us"])

    def test_directed_alternative_never_invents_reverse_edges(self):
        task, raw = make_case(8)
        raw["switches"] = ["fast", "alternate"]
        raw["edges"] = [{"src": rank, "dst": "fast", "bw": "10GB/s", "bidirectional": True}
                        for rank in task["ranks"]]
        raw["edges"] += [{"src": rank, "dst": "alternate", "bw": "8GB/s"} for rank in task["ranks"][:4]]
        raw["edges"] += [{"src": "alternate", "dst": rank, "bw": "8GB/s"} for rank in task["ranks"][4:]]
        raw["resources"] = [{"id": "fast-pool", "capacity": "1GB/s", "edges": [
            {"src": rank, "dst": "fast", "bidirectional": True} for rank in task["ranks"]]}]
        graph = Topology.from_data(raw)
        baseline = build(task, graph, 4, resource_weight=0, packets=2)
        report = replay(task, graph, baseline)
        baseline["metrics"] = {"total_time": report["total_time_us"] / 1e6,
                               "round_times": [value / 1e6 for value in report["round_times_us"]]}
        problem, context = self.extract(task, graph, baseline, time_budget_s=1, max_rounds=8)
        self.assertTrue(problem["groups"])
        result, evidence = self.apply(task, graph, baseline, context, self.proposal(problem))
        self.assert_safe_result(task, graph, baseline, result, evidence)
        for round_ in result["rounds"]:
            for transfer in round_["transfers"]:
                self.assertTrue(all(pair in graph.edges for pair in zip(transfer["path"], transfer["path"][1:])))

    def test_stale_task_is_rejected(self):
        task, graph, baseline, problem, context = self.fixture()
        changed = deepcopy(task)
        changed["data_size"]["per_rank_bytes"] *= 2
        result, evidence = self.apply(changed, graph, baseline, context, self.proposal(problem))
        self.assertIs(result, baseline)
        self.assertFalse(evidence["accepted"])

    def test_graph_mutation_is_detected_even_when_cached_input_hash_is_unchanged(self):
        for change in ("edge_capacity", "resource_capacity", "resource_membership", "edge_direction"):
            with self.subTest(change=change):
                task, graph, baseline, problem, context = self.fixture()
                old_hash = graph.input_sha256
                edge = ("r0", "alternate")
                if change == "edge_capacity":
                    graph.edges[edge] /= 2
                elif change == "resource_capacity":
                    graph.resources["shared-a"] /= 2
                elif change == "resource_membership":
                    graph.edge_resources[edge] = ("shared-a",)
                else:
                    del graph.edges[edge]
                self.assertEqual(graph.input_sha256, old_hash)
                result, evidence = self.apply(task, graph, baseline, context, self.proposal(problem))
                self.assertIs(result, baseline)
                self.assertFalse(evidence["accepted"])

    def test_stale_baseline_is_rejected_without_overwriting_it(self):
        task, graph, baseline, problem, context = self.fixture()
        changed = deepcopy(baseline)
        changed["metrics"]["total_time"] += 1
        original = deepcopy(changed)
        result, evidence = self.apply(task, graph, changed, context, self.proposal(problem))
        self.assertIs(result, changed)
        self.assertFalse(evidence["accepted"])
        self.assertEqual(changed, original)

    def test_model_cannot_edit_paths_actions_or_other_unadvertised_fields(self):
        task, graph, baseline, problem, context = self.fixture()
        proposal = self.proposal(problem)
        for field, value in [("path", ["r0", "r1"]), ("action", "copy"), ("round", 1), ("split", 8)]:
            with self.subTest(field=field):
                result, evidence = self.apply(task, graph, baseline, context, {**proposal, field: value})
                self.assertIs(result, baseline)
                self.assertFalse(evidence["accepted"])

    def test_malformed_or_out_of_range_edits_fail_closed(self):
        task, graph, baseline, problem, context = self.fixture()
        group = problem["groups"][0]
        malformed = [None, [], {"base": problem["base"]},
                     {"base": "0" * 16, "edits": []},
                     {"base": problem["base"], "edits": [["unknown", 1]]},
                     {"base": problem["base"], "edits": [[group["id"], 1], [group["id"], 1]]}]
        malformed += [{"base": problem["base"], "edits": [[group["id"], value]]}
                      for value in (True, False, -1, 0.5, "1", None, group["max_chunks"] + 1)]
        original = deepcopy(baseline)
        for proposal in malformed:
            with self.subTest(proposal=proposal):
                result, evidence = self.apply(task, graph, baseline, context, proposal)
                self.assertIs(result, baseline)
                self.assertFalse(evidence["accepted"])
        self.assertEqual(baseline, original)

    def test_prompt_catalog_mutation_cannot_rewrite_private_edit_paths(self):
        task, graph, baseline, problem, context = self.fixture()
        proposal = self.proposal(problem, 0.5)
        for group in problem["groups"]:
            group["path_sample"]["new"][:] = ["invented-rank", "invented-switch"]
            group["max_chunks"] = 10**12
            group["effects"] = [["invented-resource", -10**30]]
        result, evidence = self.apply(task, graph, baseline, context, proposal)
        self.assert_safe_result(task, graph, baseline, result, evidence)

    def test_empty_edit_is_an_explicit_noop(self):
        task, graph, baseline, problem, context = self.fixture()
        result, evidence = self.apply(task, graph, baseline, context, {"base": problem["base"], "edits": []})
        self.assertIs(result, baseline)
        self.assertFalse(evidence["accepted"])

    def test_independent_ablation_cost_matches_full_replay_for_both_operators(self):
        from tests.edit_ablation import ExactEditObjective
        for operator in ("all-gather", "all-reduce"):
            task, graph, baseline, problem, context = self.fixture(operator)
            objective = ExactEditObjective(task, graph, baseline, context)
            for numerator in (0, 1, 3, 7, 8):
                with self.subTest(operator=operator, fraction=numerator / 8):
                    counts = [group.max_chunks * numerator // 8 for group in context.groups]
                    proposal = {"base": problem["base"], "edits": [
                        [group.id, amount] for group, amount in zip(context.groups, counts)]}
                    _, evidence = self.apply(task, graph, baseline, context, proposal)
                    self.assertAlmostEqual(objective(counts), evidence["after_us"])
                    if evidence["proposal_time_us"] is not None:
                        self.assertAlmostEqual(objective.proposal_time_us(counts), evidence["proposal_time_us"])

    def test_zero_budget_ablation_preserves_valid_baseline(self):
        from tests.edit_ablation import optimize_counts
        task, graph, baseline, problem, context = self.fixture()
        result, proposal, audit = optimize_counts(task, graph, baseline, problem, context, time_budget_s=0)
        self.assertIs(result, baseline)
        self.assertEqual(proposal["edits"], [])
        self.assertFalse(audit["accepted"])


if __name__ == "__main__":
    unittest.main()
