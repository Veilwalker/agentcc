"""Original model intent stays separate from postprocessing and selection."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import unittest
from unittest.mock import patch

from src.collectives import build
from src.provenance import build_model_ledger, observe_model_ledger
from src.replay import replay
from src.schedule_edit import apply_proposal, extract_problem
from tests.test_route_balance import parallel_case


def atomic_packets(solution):
    return {**solution, "rounds": [{**row, "transfers": [
        {**transfer, "chunk_ranges": [token], "path": list(transfer["path"])}
        for transfer in reversed(row["transfers"]) for token in reversed(transfer["chunk_ranges"])]}
        for row in solution["rounds"]]}


class ProvenanceTests(unittest.TestCase):
    def fixture(self, operator="all-gather"):
        task, graph = parallel_case(operator)
        baseline = build(task, graph, 4, kind="xor_tree" if operator == "all-reduce" else "reverse_doubling",
                         resource_weight=0, packets=1)
        problem, context = extract_problem(task, graph, baseline, time_budget_s=1,
                                           max_groups=16, max_rounds=8)
        proposal = {"base": problem["base"], "edits": [[g["id"], max(1, g["max_chunks"] // 2 - 1)]
                                                       for g in problem["groups"]]}
        edited, evidence = apply_proposal(task, graph, baseline, context, proposal)
        self.assertTrue(evidence["accepted"], evidence)
        ledger = build_model_ledger(baseline, evidence, context=context)
        self.assertTrue(ledger.available, ledger.reason)
        return task, graph, baseline, edited, evidence, context, ledger

    def test_actual_compiler_counts_and_ar_expressions_survive_packet_reordering(self):
        for operator in ("all-gather", "all-reduce"):
            with self.subTest(operator=operator):
                task, graph, baseline, edited, evidence, _, ledger = self.fixture(operator)
                before = deepcopy((baseline, edited, evidence))
                final = atomic_packets(edited)
                self.assertTrue(replay(task, graph, final)["valid"])
                observed = observe_model_ledger(ledger, final, control=baseline)
                self.assertTrue(observed["available"])
                self.assertTrue(observed["all_targets_preserved"])
                self.assertEqual(observed["original_requested_edits"], evidence["requested_edits"]["edits"])
                self.assertEqual(observed["original_applied_edits"], evidence["applied_edits"])
                self.assertEqual(observed["preserved_atomic_count"], sum(row[1] for row in evidence["applied_edits"]))
                self.assertEqual(observed["new_target_atoms_vs_control"], observed["preserved_atomic_count"])
                self.assertEqual(observed["already_control_target_atomic_count"], 0)
                self.assertTrue(all(row["preserved_count"] == row["original_applied_count"]
                                    for row in observed["group_outcomes"]))
                self.assertEqual(len(observed["selected_atomic_positions"]), len(ledger.atoms))
                self.assertEqual(observed["fully_preserved_edits"], evidence["applied_edits"])
                self.assertFalse(observed["partially_preserved_edits"])
                for item in observed["model_atomic_manifest"]:
                    packet = final["rounds"][item["round"] - 1]["transfers"][item["transfer_index"]]
                    self.assertEqual(packet["path"], item["model_path"])
                    self.assertEqual([packet["chunk_ranges"][ci] for ci in item["chunk_indices"]], item["chunks"])
                if operator == "all-reduce":
                    self.assertTrue(any("+" in atom.token for atom in ledger.atoms))
                self.assertEqual((baseline, edited, evidence), before)

    def test_only_original_model_targets_are_counted_after_new_deterministic_edits(self):
        _, _, baseline, edited, _, _, ledger = self.fixture()
        final = atomic_packets(edited)
        target_keys = {atom.key for atom in ledger.atoms}
        arbitrary_changes = 0
        for ri, row in enumerate(final["rounds"]):
            for transfer in row["transfers"]:
                key = ri, transfer["action"], transfer["src"], transfer["dst"], transfer["chunk_ranges"][0]
                if key not in target_keys:
                    transfer["path"] = [transfer["src"], "alternate", transfer["dst"]]
                    arbitrary_changes += 1
        self.assertGreater(arbitrary_changes, len(ledger.atoms))
        observed = observe_model_ledger(ledger, final, control=baseline)
        self.assertEqual(observed["accepted_atomic_count"], len(ledger.atoms))
        self.assertEqual(observed["preserved_atomic_count"], len(ledger.atoms))

    def test_partial_overwrite_absence_and_ambiguity_have_distinct_counts(self):
        _, _, baseline, edited, _, _, ledger = self.fixture()
        final = atomic_packets(edited)
        lookup = {atom.key: atom for atom in ledger.atoms[:3]}
        changed = 0
        for ri, row in enumerate(final["rounds"]):
            transfers = []
            for transfer in row["transfers"]:
                key = ri, transfer["action"], transfer["src"], transfer["dst"], transfer["chunk_ranges"][0]
                if key in lookup:
                    changed += 1
                    if changed == 1:
                        transfer["path"] = list(lookup[key].before_path)
                    elif changed == 2:
                        continue
                    else:
                        transfers.append(deepcopy(transfer))
                transfers.append(transfer)
            row["transfers"] = transfers
        observed = observe_model_ledger(ledger, final, control=baseline)
        self.assertTrue(observed["available"])
        self.assertEqual(observed["overwritten_atomic_count"], 1)
        self.assertEqual(observed["absent_atomic_count"], 1)
        self.assertEqual(observed["ambiguous_atomic_count"], 1)
        self.assertEqual(observed["preserved_atomic_count"], len(ledger.atoms) - 3)
        self.assertFalse(observed["all_targets_preserved"])
        applied = dict(observed["original_applied_edits"])
        for group, retained in observed["fully_preserved_edits"]:
            self.assertEqual(retained, applied[group])
        for group, retained in observed["partially_preserved_edits"]:
            self.assertGreater(retained, 0)
            self.assertLess(retained, applied[group])

    def test_all_targets_overwritten_remains_an_observation_not_a_candidate_failure(self):
        _, _, baseline, _, _, _, ledger = self.fixture()
        observed = observe_model_ledger(ledger, baseline, control=baseline)
        self.assertTrue(observed["available"])
        self.assertFalse(observed["all_targets_preserved"])
        self.assertEqual(observed["preserved_atomic_count"], 0)
        self.assertEqual(observed["overwritten_atomic_count"], len(ledger.atoms))

    def test_control_matching_targets_is_not_new_model_contribution_and_reuses_scan(self):
        _, _, _, edited, _, _, ledger = self.fixture()
        from src import provenance
        real = provenance._find_atoms
        with patch("src.provenance._find_atoms", wraps=real) as scans:
            observed = observe_model_ledger(ledger, edited, control={**edited})
        self.assertEqual(scans.call_count, 1)
        self.assertTrue(observed["all_targets_preserved"])
        self.assertEqual(observed["new_target_atoms_vs_control"], 0)
        self.assertEqual(observed["already_control_target_atomic_count"], len(ledger.atoms))

    def test_control_ambiguity_or_missing_actions_is_not_counted_as_new(self):
        _, _, baseline, edited, _, _, ledger = self.fixture()
        control = atomic_packets(baseline)
        first = ledger.atoms[0]
        for transfer in control["rounds"][first.round_index]["transfers"]:
            if (transfer["action"], transfer["src"], transfer["dst"], transfer["chunk_ranges"][0]) == first.key[1:]:
                control["rounds"][first.round_index]["transfers"].append(deepcopy(transfer))
                break
        observed = observe_model_ledger(ledger, edited, control=control)
        self.assertEqual(observed["control_unknown_atomic_count"], 1)
        self.assertEqual(observed["new_target_atoms_vs_control"], len(ledger.atoms) - 1)

    def test_layout_mismatch_does_not_throw_or_discard_any_winner(self):
        _, _, _, edited, _, _, ledger = self.fixture()
        for field in ("split", "initial_possession", "rounds"):
            final = deepcopy(edited)
            if field == "split":
                final[field] = 1
            elif field == "initial_possession":
                final[field] = {}
            else:
                final[field] = final[field][:-1]
            before = deepcopy(final)
            observed = observe_model_ledger(ledger, final)
            self.assertFalse(observed["available"])
            self.assertEqual(observed["reason"], "incompatible_final_layout")
            self.assertIsNone(observed["preserved_atomic_count"])
            self.assertEqual(final, before)
        observed = observe_model_ledger(ledger, edited, control={})
        self.assertTrue(observed["available"])
        self.assertFalse(observed["control_available"])
        self.assertIsNone(observed["new_target_atoms_vs_control"])

    def test_manifest_binding_errors_and_changed_counts_fail_closed(self):
        _, _, baseline, _, evidence, context, _ = self.fixture()
        bad = deepcopy(evidence)
        bad["edit_manifest"][0][2] += 1
        self.assertFalse(build_model_ledger(baseline, bad, context=context).available)
        self.assertFalse(build_model_ledger(baseline, evidence, context=replace(context, base="stale")).available)
        bad = deepcopy(evidence)
        bad["edit_manifest"].append(deepcopy(bad["edit_manifest"][0]))
        self.assertFalse(build_model_ledger(baseline, bad).available)
        bad = deepcopy(evidence)
        bad["applied_edits"][0][1] = True
        self.assertFalse(build_model_ledger(baseline, bad).available)

    def test_rejected_request_keeps_original_counts_without_invented_accepted_atoms(self):
        evidence = {"accepted": False, "requested_edits": {"base": "x", "edits": [["g0", 2], ["g1", 0]]}}
        ledger = build_model_ledger({}, evidence)
        observed = observe_model_ledger(ledger, {})
        self.assertFalse(observed["available"])
        self.assertEqual(observed["original_requested_edits"], [["g0", 2], ["g1", 0]])
        self.assertEqual(observed["accepted_atomic_count"], 0)

    def test_ledger_is_detached_and_has_no_dependency_on_final_metrics(self):
        _, _, baseline, edited, evidence, context, ledger = self.fixture()
        before = observe_model_ledger(ledger, edited, control=baseline)
        evidence["requested_edits"]["edits"][0][1] = 999
        context = replace(context, base="after-capture")
        final = {**edited, "metrics": {"total_time": -1}, "strategy": {"unrelated": True}}
        self.assertEqual(before, observe_model_ledger(ledger, final, control=baseline))
        with self.assertRaises(FrozenInstanceError):
            ledger.atoms = ()

    def test_context_is_optional_but_unknown_group_assignments_are_not_invented(self):
        _, _, baseline, edited, evidence, _, _ = self.fixture()
        ledger = build_model_ledger(baseline, evidence)
        self.assertTrue(ledger.available)
        observed = observe_model_ledger(ledger, edited, control=baseline)
        self.assertTrue(observed["all_targets_preserved"])
        self.assertFalse(observed["group_counts_available"])
        self.assertIsNone(observed["group_outcomes"])
        self.assertEqual(observed["fully_preserved_edits"], [])

    def test_same_path_prefix_is_not_reported_as_model_path_change(self):
        _, _, baseline, _, evidence, _, _ = self.fixture()
        entry = deepcopy(evidence["edit_manifest"][0])
        entry[4] = list(entry[3])
        counts = [["g0", entry[2]]]
        evidence = {**evidence, "edit_manifest": [entry], "accepted_rounds": [entry[0]],
                    "requested_edits": {"base": evidence["base"], "edits": counts}, "applied_edits": counts}
        ledger = build_model_ledger(baseline, evidence)
        observed = observe_model_ledger(ledger, baseline, control=baseline)
        self.assertTrue(observed["available"])
        self.assertEqual(observed["ignored_noop_atomic_count"], entry[2])
        self.assertEqual(observed["preserved_atomic_count"], 0)
        self.assertEqual(observed["new_target_atoms_vs_control"], 0)
        self.assertFalse(observed["all_targets_preserved"])


if __name__ == "__main__":
    unittest.main()
