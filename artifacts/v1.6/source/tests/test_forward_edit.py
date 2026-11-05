"""Forwarding rewrites must conserve reception and respect round-start state."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import unittest

from src.collectives import build
from src.forward_edit import (Context, Move, Rewrite, apply, compile_rewrites,
                              copy_states, _body, _digest)
from tests.forward_catalog import extract, rebase
from src.replay import replay
from src.topology import Topology
from tests.test_solver import make_case


def fixture(operator="all-gather"):
    task, data = make_case(4, operator)
    data["switches"] = []
    data["edges"] = [dict(src=a, dst=b, bw="0.1GB/s" if (a, b) == ("r0", "r1") else "1GB/s")
                     for a in task["ranks"] for b in task["ranks"] if a != b]
    graph = Topology.from_data(data)
    base = build(task, graph, 1 if operator == "all-gather" else 4,
                 kind="reverse_doubling" if operator == "all-gather" else "xor_tree",
                 order=graph.ranks)
    return task, graph, base


def receptions(solution):
    return Counter((t["action"], t["dst"], a) for row in solution["rounds"]
                   for t in row["transfers"] for a in t["chunk_ranges"])


class ForwardTests(unittest.TestCase):
    def test_advancing_ready_atoms_is_a_real_valid_change(self):
        task, graph, base = fixture()
        saved = deepcopy(base)
        problem, ctx, info = extract(task, graph, base, max_route_queries=0)
        result, ev = apply(task, graph, base, ctx, {"apply": [0]})
        self.assertTrue(ev['accepted'], ev)
        self.assertGreater(ev['advanced_atoms'], 0)
        self.assertEqual(ev['sender_changes'], 0)
        self.assertEqual(receptions(base), receptions(result))
        self.assertEqual(base, saved)
        self.assertTrue(replay(task, graph, result)['valid'])
        self.assertLess(ev['after_us'], ev['before_us'])
        self.assertEqual(info['route_queries'], 0)
        self.assertEqual(problem['groups'][0][0], 'advance')

    def test_sender_must_hold_all_atoms_at_round_start(self):
        task, graph, base = fixture()
        _, ctx, _ = extract(task, graph, base, max_route_queries=0)
        ti = next(i for i, t in enumerate(base['rounds'][1]['transfers']) if t['src'] == 'r0')
        tokens = tuple(base['rounds'][1]['transfers'][ti]['chunk_ranges'])
        good = Move(1, ti, tokens, 'r2', 1, ('r2', 'r1'))
        context = replace(ctx, rewrites=(Rewrite('sender', (good,)),))
        result, ev = apply(task, graph, base, context, {'apply': [0]})
        self.assertTrue(ev['accepted'], ev)
        self.assertEqual(ev['sender_changes'], 1)
        self.assertEqual(receptions(base), receptions(result))
        bad = replace(good, source='r3', path=('r3', 'r1'))
        result, ev = apply(task, graph, base, replace(context, rewrites=(Rewrite('sender', (bad,)),)), {'apply': [0]})
        self.assertIs(result, base)
        self.assertFalse(ev['accepted'])
        self.assertTrue(ev['errors'])

    def test_full_packet_cannot_be_advanced_if_only_subset_is_ready(self):
        task, graph, base = fixture()
        _, ctx, _ = extract(task, graph, base, max_route_queries=0)
        first = ctx.rewrites[0].moves[0]
        tokens = tuple(base['rounds'][first.round_index]['transfers'][first.transfer_index]['chunk_ranges'])
        invalid = replace(first, atoms=tokens)
        result, ev = apply(task, graph, base, replace(ctx, rewrites=(Rewrite('advance', (invalid,)),)), {'apply': [0]})
        self.assertIs(result, base)
        self.assertTrue(ev['errors'])

    def test_allreduce_edits_never_move_partial_reductions(self):
        task, graph, base = fixture('all-reduce')
        _, ctx, _ = extract(task, graph, base, max_route_queries=0)
        states, bit = copy_states(task, graph, base)
        for rw in ctx.rewrites:
            for m in rw.moves:
                original = base['rounds'][m.round_index]['transfers'][m.transfer_index]
                self.assertEqual(original['action'], 'copy')
                self.assertTrue(all(a.endswith('_sum') for a in m.atoms))
                self.assertTrue(all(states[m.target_round][graph.ranks.index(m.source)] & bit(a) for a in m.atoms))
        edited, ev = apply(task, graph, base, ctx, {'apply': list(range(len(ctx.rewrites)))})
        self.assertTrue(replay(task, graph, edited)['valid'])
        reductions = lambda x: [t for r in x['rounds'] for t in r['transfers'] if t['action'] == 'reduce']
        self.assertEqual(reductions(base), reductions(edited))

    def test_possession_does_not_depend_on_chunks_list_order(self):
        task, graph, base = fixture()
        before, _ = copy_states(task, graph, base)
        original = deepcopy(base)
        base['chunks'].reverse()
        self.assertTrue(replay(task, graph, base)['valid'])
        _, ctx, _ = extract(task, graph, base, max_route_queries=0)
        result, ev = apply(task, graph, base, ctx, {'apply': [0]})
        self.assertTrue(ev['accepted'])
        self.assertTrue(replay(task, graph, result)['valid'])
        self.assertEqual(receptions(original), receptions(result))

    def test_malformed_overlapping_and_stale_proposals_keep_baseline(self):
        task, graph, base = fixture()
        _, ctx, _ = extract(task, graph, base, max_route_queries=0)
        for proposal in (None, {}, {'apply':[True]}, {'apply':[0,0]}, {'apply':[-1]},
                         {'apply':[99]}, {'apply':[0.0]}, {'apply':[], 'other':1}):
            with self.subTest(proposal=proposal):
                result, ev = apply(task, graph, base, ctx, proposal)
                self.assertIs(result, base)
                self.assertTrue(ev['errors'])
        duplicate = replace(ctx, rewrites=(ctx.rewrites[0], ctx.rewrites[0]))
        self.assertTrue(apply(task, graph, base, duplicate, {'apply':[0,1]})[1]['errors'])
        stale = deepcopy(base); stale['chunks'].reverse()
        self.assertTrue(apply(task, graph, stale, ctx, {'apply':[0]})[1]['errors'])
        graph.edges['r0','r1'] *= 2
        self.assertTrue(apply(task, graph, base, ctx, {'apply':[0]})[1]['errors'])

    def test_rebase_preserves_atoms_across_packet_boundaries(self):
        task, graph, base = fixture()
        _, ctx, _ = extract(task, graph, base, max_route_queries=0)
        target = deepcopy(base)
        for row in target['rounds']:
            row['transfers'] = [{**t, 'chunk_ranges':[a]} for t in row['transfers'] for a in t['chunk_ranges']]
        bound = rebase(task, graph, base, target, ctx, proposal={'apply':[0]})
        self.assertEqual(sum(len(m.atoms) for m in ctx.rewrites[0].moves), sum(len(m.atoms) for m in bound.rewrites[0].moves))
        compiled = compile_rewrites(target, bound, [0])
        self.assertTrue(replay(task, graph, compiled)['valid'])
        self.assertEqual(receptions(base), receptions(compiled))

    def test_changed_actions_cannot_be_silently_dropped(self):
        task, graph, base = fixture()
        _, ctx, _ = extract(task, graph, base, max_route_queries=0)
        target = deepcopy(base)
        move = ctx.rewrites[0].moves[0]
        target['rounds'][move.round_index]['transfers'][move.transfer_index]['chunk_ranges'].remove(move.atoms[0])
        with self.assertRaises(ValueError):
            rebase(task, graph, base, target, ctx, proposal={'apply':[0]})

    def test_noop_is_not_counted_as_model_adoption(self):
        task, graph, base = fixture()
        _, ctx, _ = extract(task, graph, base, max_route_queries=0)
        result, ev = apply(task, graph, base, ctx, {'apply':[]})
        self.assertIs(result, base)
        self.assertFalse(ev['accepted'])
        self.assertFalse(ev['selected'])
