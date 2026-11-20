"""Executable design experiment for fair postprocessing and honest provenance.

This module is not imported by production. Internal route/packet proposers must
preserve complete round data actions. The pair wrapper validates each changed
final branch once, then compares reports; observation is not a selection rule.
"""

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import unittest
from unittest.mock import patch

from src.collectives import build
from src.replay import replay
from src.selection import improves, round_actions
from tests.test_route_balance import parallel_case


@dataclass(frozen=True)
class Intent:
    split: int
    initial_possession: dict
    final_chunks: list
    round_count: int
    # (zero-based round, action, src, dst, complete token) -> target path.
    targets: dict


def compiler_intent(baseline, edit_manifest):
    """Read only already accepted original prefixes, not full schedule Counters.

    The production compiler has already bound the baseline to the input/hash.
    This prototype is an internal hand-off, not an external validation API.
    """
    targets = {}
    for number, ti, count, before_path, after_path in edit_manifest:
        transfer = baseline["rounds"][number - 1]["transfers"][ti]
        if list(transfer["path"]) != before_path or type(count) is not int or not 0 < count <= len(transfer["chunk_ranges"]):
            raise ValueError("manifest does not match its bound baseline")
        if before_path == after_path:
            continue
        for token in transfer["chunk_ranges"][:count]:
            key = number - 1, transfer["action"], transfer["src"], transfer["dst"], token
            if key in targets:
                raise ValueError("ambiguous duplicate model atom")
            targets[key] = tuple(after_path)
    return Intent(baseline["split"], baseline["initial_possession"], baseline["final_chunks"],
                  len(baseline["rounds"]), targets)


def observe_final(intent, final):
    """Observe exact surviving targets once, without constraining any route.

    Coordinates are rebuilt from final packets. Partial survival does not alter
    the original model count; it is an explicitly separate outcome measure.
    """
    if (intent.split != final["split"] or intent.initial_possession != final["initial_possession"]
            or intent.final_chunks != final["final_chunks"] or intent.round_count != len(final["rounds"])):
        return {"available": False, "reason": "incompatible_layout", "all_targets_preserved": False}
    rounds = {key[0] for key in intent.targets}
    found = defaultdict(list)
    for ri in sorted(rounds):
        for ti, transfer in enumerate(final["rounds"][ri]["transfers"]):
            prefix = ri, transfer["action"], transfer["src"], transfer["dst"]
            for ci, token in enumerate(transfer["chunk_ranges"]):
                key = (*prefix, token)
                if key in intent.targets:
                    found[key].append((tuple(transfer["path"]), [ri, ti, ci]))
    preserved, overwritten, absent, ambiguous = [], [], [], []
    for key, target in intent.targets.items():
        matches = found[key]
        if not matches:
            absent.append(key)
        elif len(matches) != 1:
            ambiguous.append(key)
        elif matches[0][0] == target:
            preserved.append(matches[0][1])
        else:
            overwritten.append(key)
    return {"available": True, "requested_atomic_count": len(intent.targets),
            "preserved_atomic_count": len(preserved), "overwritten_atomic_count": len(overwritten),
            "absent_atomic_count": len(absent), "ambiguous_atomic_count": len(ambiguous),
            "all_targets_preserved": bool(intent.targets) and len(preserved) == len(intent.targets),
            "selected_atomic_positions": sorted(preserved), "preservation_scope": "observed_not_enforced"}


def postprocess_pair(task, graph, control, control_report, edited, edited_report,
                     intent, route_proposer, packet_proposer):
    """Same H=packet(route(.)) policy for both; one Replay per changed branch.

    Callback policy/work limits are shared. Existing research optimizer APIs
    replay internally, so they need propose/validate separation before being
    connected to this orchestration; those extra checks are not hidden here.
    """
    def apply_h(solution, report):
        try:
            candidate = packet_proposer(route_proposer(solution))
            if candidate is solution:
                return solution, report, {"replay_reused": True, "accepted": False}
            checked = replay(task, graph, candidate)
            if not checked["valid"] or checked["total_time_us"] > report["total_time_us"]:
                return solution, report, {"replay_reused": False, "accepted": False,
                                         "reason": "invalid_or_worse"}
            candidate = {**candidate, "metrics": {"total_time": checked["total_time_us"] / 1e6,
                         "round_times": [value / 1e6 for value in checked["round_times_us"]]}}
            return candidate, checked, {"replay_reused": False, "accepted": True}
        except (ValueError, KeyError, IndexError, TypeError):
            return solution, report, {"replay_reused": True, "accepted": False, "reason": "proposal_failed"}

    optimized_control, control_checked, control_evidence = apply_h(control, control_report)
    optimized_edited, edit_checked, edited_evidence = (optimized_control, control_checked, control_evidence)
    if edited is not control:
        optimized_edited, edit_checked, edited_evidence = apply_h(edited, edited_report)
    model_branch_won = improves(edit_checked["total_time_us"], control_checked["total_time_us"])
    winner, report = ((optimized_edited, edit_checked) if model_branch_won
                      else (optimized_control, control_checked))
    evidence = {"model_branch_won": model_branch_won, "control": control_evidence,
                "edited": edited_evidence, "postprocessed_control_time_us": control_checked["total_time_us"],
                "postprocessed_edited_time_us": edit_checked["total_time_us"],
                "final_model_observation": observe_final(intent, winner)}
    return winner, report, evidence


def split_packets(solution):
    """The simplest same-path packet pass for small semantic tests."""
    rounds = [{**row, "transfers": [{**transfer, "chunk_ranges": [token]}
              for transfer in row["transfers"] for token in transfer["chunk_ranges"]]}
              for row in solution["rounds"]]
    return {**solution, "rounds": rounds}


class PostprocessSelectionExperimentTests(unittest.TestCase):
    def fixture(self, operator="all-gather"):
        task, graph = parallel_case(operator)
        baseline = build(task, graph, 4, kind="xor_tree" if operator == "all-reduce" else "reverse_doubling",
                         resource_weight=0)
        ri, ti = 1, 0
        edited = deepcopy(baseline)
        transfer = edited["rounds"][ri]["transfers"][ti]
        target = [transfer["src"], "alternate", transfer["dst"]]
        transfer["path"] = target
        manifest = [[ri + 1, ti, len(transfer["chunk_ranges"]),
                     baseline["rounds"][ri]["transfers"][ti]["path"], target]]
        return task, graph, baseline, edited, compiler_intent(baseline, manifest)

    def checked(self, task, graph, value):
        result = replay(task, graph, value)
        self.assertTrue(result["valid"], result["errors"])
        return result

    def test_repacketization_keeps_targets_but_rebuilds_final_coordinates(self):
        for operator in ("all-gather", "all-reduce"):
            with self.subTest(operator=operator):
                task, graph, baseline, edited, intent = self.fixture(operator)
                result = split_packets(edited)
                self.checked(task, graph, result)
                observation = observe_final(intent, result)
                self.assertTrue(observation["all_targets_preserved"])
                self.assertEqual(observation["preserved_atomic_count"], len(intent.targets))
                self.assertTrue(all(ci == 0 for _, _, ci in observation["selected_atomic_positions"]))
                if len(intent.targets) > 1:
                    self.assertGreater(len({ti for _, ti, _ in observation["selected_atomic_positions"]}), 1)
                for ri, ti, ci in observation["selected_atomic_positions"]:
                    t = result["rounds"][ri]["transfers"][ti]
                    self.assertEqual(tuple(t["path"]), intent.targets[(ri, t["action"], t["src"], t["dst"], t["chunk_ranges"][ci])])
                for before, after in zip(edited["rounds"], result["rounds"]):
                    self.assertEqual(round_actions(before), round_actions(after))
                if operator == "all-reduce":
                    self.assertTrue(any("+" in token for *_, token in intent.targets))

    def test_partial_route_overwrite_cannot_claim_whole_count_preserved(self):
        task, graph, baseline, edited, intent = self.fixture()
        final = split_packets(edited)
        changed = next(iter(intent.targets))
        for transfer in final["rounds"][changed[0]]["transfers"]:
            if transfer["chunk_ranges"] == [changed[-1]] and transfer["src"] == changed[2] and transfer["dst"] == changed[3]:
                transfer["path"] = list(baseline["rounds"][changed[0]]["transfers"][0]["path"])
                break
        self.checked(task, graph, final)
        observation = observe_final(intent, final)
        self.assertFalse(observation["all_targets_preserved"])
        self.assertEqual(observation["overwritten_atomic_count"], 1)
        self.assertEqual(observation["preserved_atomic_count"], len(intent.targets) - 1)

    def test_shared_policy_tie_prefers_control_despite_matching_model_targets(self):
        task, graph, baseline, edited, intent = self.fixture()
        def route_all_alternate(solution):
            return {**solution, "rounds": [{**row, "transfers": [
                {**t, "path": [t["src"], "alternate", t["dst"]]} for t in row["transfers"]]}
                for row in solution["rounds"]]}
        seen = []
        def proposer(solution):
            seen.append(solution)
            return route_all_alternate(solution)
        br, er = self.checked(task, graph, baseline), self.checked(task, graph, edited)
        before = deepcopy((baseline, edited))
        result, report, evidence = postprocess_pair(task, graph, baseline, br, edited, er,
                                                   intent, proposer, split_packets)
        self.assertEqual(len(seen), 2)
        self.assertIs(seen[0], baseline)
        self.assertIs(seen[1], edited)
        self.assertFalse(evidence["model_branch_won"])
        self.assertTrue(evidence["final_model_observation"]["all_targets_preserved"],
                        "observed target equality is not proof of causal model contribution")
        self.assertLessEqual(report["total_time_us"], br["total_time_us"])
        self.assertEqual(report, self.checked(task, graph, result))
        self.assertEqual((baseline, edited), before)

    def test_invalid_postprocessing_does_not_discard_original_valid_branch(self):
        task, graph, baseline, edited, intent = self.fixture()
        br, er = self.checked(task, graph, baseline), self.checked(task, graph, edited)
        def drop_transfer(solution):
            bad = deepcopy(solution)
            bad["rounds"][0]["transfers"].pop()
            return bad
        result, report, evidence = postprocess_pair(task, graph, baseline, br, edited, er,
                                                   intent, drop_transfer, lambda x: x)
        self.assertEqual(evidence["control"]["reason"], "invalid_or_worse")
        self.assertEqual(evidence["edited"]["reason"], "invalid_or_worse")
        self.assertLessEqual(report["total_time_us"], br["total_time_us"])
        self.assertEqual(report, self.checked(task, graph, result))

    def test_better_unrestricted_paths_can_replace_all_model_atoms(self):
        for operator in ("all-gather", "all-reduce"):
            task, graph, baseline, edited, intent = self.fixture(operator)
            graph.switches.add("third")
            graph.nodes.add("third")
            for rank in graph.ranks:
                for edge in ((rank, "third"), ("third", rank)):
                    graph.edges[edge] = 1e12
                    graph.edge_resources[edge] = ()
            def route_third(solution):
                return {**solution, "rounds": [{**row, "transfers": [
                    {**t, "path": [t["src"], "third", t["dst"]]} for t in row["transfers"]]}
                    for row in solution["rounds"]]}
            br, er = self.checked(task, graph, baseline), self.checked(task, graph, edited)
            result, report, evidence = postprocess_pair(task, graph, baseline, br, edited, er,
                                                       intent, route_third, split_packets)
            self.assertLess(report["total_time_us"], min(br["total_time_us"], er["total_time_us"]))
            self.assertEqual(evidence["final_model_observation"]["preserved_atomic_count"], 0)
            self.assertEqual(evidence["final_model_observation"]["overwritten_atomic_count"], len(intent.targets))
            self.assertFalse(evidence["final_model_observation"]["all_targets_preserved"])
            self.assertEqual(result["metrics"]["total_time"], report["total_time_us"] / 1e6)

    def test_identity_postprocessing_reuses_report_and_same_branch_once(self):
        task, graph, baseline, _, intent = self.fixture()
        br = self.checked(task, graph, baseline)
        calls = []
        def identity(solution):
            calls.append(solution)
            return solution
        with patch(__name__ + ".replay", side_effect=AssertionError("unchanged validated object needs no replay")):
            result, report, evidence = postprocess_pair(task, graph, baseline, br, baseline, br,
                                                       intent, identity, lambda x: x)
        self.assertEqual(len(calls), 1)
        self.assertIs(result, baseline)
        self.assertIs(report, br)
        self.assertFalse(evidence["model_branch_won"])

    def test_valid_but_worse_postprocess_retains_strong_control_and_report(self):
        task, graph, baseline, _, intent = self.fixture()
        control = deepcopy(baseline)
        for row in control["rounds"]:
            for t in row["transfers"]:
                t["path"] = [t["src"], "alternate", t["dst"]]
        cr = self.checked(task, graph, control)
        worse = self.checked(task, graph, baseline)
        self.assertGreater(worse["total_time_us"], cr["total_time_us"])
        result, report, evidence = postprocess_pair(task, graph, control, cr, control, cr,
                                                   intent, lambda x: deepcopy(baseline), lambda x: x)
        self.assertIs(result, control)
        self.assertIs(report, cr)
        self.assertEqual(evidence["control"]["reason"], "invalid_or_worse")

    def test_changed_pair_replays_only_final_joint_and_packet_outputs(self):
        task, graph, baseline, edited, intent = self.fixture()
        br, er = self.checked(task, graph, baseline), self.checked(task, graph, edited)
        real = replay
        validated_objects = []
        def check(task, graph, candidate):
            self.assertTrue(all(len(t["chunk_ranges"]) == 1 for r in candidate["rounds"] for t in r["transfers"]))
            validated_objects.append(candidate)
            return real(task, graph, candidate)
        with patch(__name__ + ".replay", side_effect=check):
            result, report, _ = postprocess_pair(task, graph, baseline, br, edited, er,
                                                intent, lambda x: {**x}, split_packets)
        self.assertEqual(len(validated_objects), 2)
        self.assertTrue(any(result["rounds"] is value["rounds"] for value in validated_objects))
        self.assertEqual(report, self.checked(task, graph, result))

    def test_different_split_and_duplicate_atoms_are_not_false_survivors(self):
        task, graph, baseline, edited, intent = self.fixture()
        incompatible = build(task, graph, 1)
        self.assertFalse(observe_final(intent, incompatible)["available"])
        duplicate = deepcopy(edited)
        duplicate["rounds"][1]["transfers"].append(deepcopy(duplicate["rounds"][1]["transfers"][0]))
        observed = observe_final(intent, duplicate)
        self.assertEqual(observed["ambiguous_atomic_count"], len(intent.targets))
        self.assertFalse(observed["all_targets_preserved"])


if __name__ == "__main__":
    unittest.main()
