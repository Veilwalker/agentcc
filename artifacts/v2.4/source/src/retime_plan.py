"""Model-authored retiming instructions compiled against the latest incumbent.

A rule identifies a copy round, not a stale packet index. A count of 1 moves
half of each transfer's eligible atoms, 2 moves all, and 0 leaves it alone.
Eligibility always means the sender possessed that complete atom before the
preceding round. Rules are instance-bound and interpreted only once; the
compiler does not search alternative model counts or repair rejected choices.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

from .forward_edit import (Context, Move, Rewrite, _body, _digest, _round_loads,
                           apply, copy_states)
from .route_balance import _PathCosts, _RoundLoad
from .schedule_edit import _input_digest


@dataclass(frozen=True)
class PlanContext:
    input_hash: str
    rounds: tuple[int, ...]
    round_count: int
    seed_hash: str


def _moves(solution, graph, states, atom_bit, ri, amount):
    moves = []
    rank_index = {rank: i for i, rank in enumerate(graph.ranks)}
    for ti, transfer in enumerate(solution['rounds'][ri]['transfers']):
        if transfer['action'] != 'copy':
            continue
        mask = states[ri - 1][rank_index[transfer['src']]]
        ready = tuple(a for a in transfer['chunk_ranges'] if mask & atom_bit(a))
        # A proportion applies separately to each packet to avoid a prefix
        # concentrated on the first rank or resource. Every atom stays whole.
        chosen = ready[:(len(ready) * amount + 1) // 2]
        if chosen:
            moves.append(Move(ri, ti, chosen, transfer['src'], ri - 1, tuple(transfer['path'])))
    return tuple(moves)


def _preview_pair(seed, graph, states, atom_bit, ri, chunk_bytes, costs):
    """Cost half/full edits without constructing temporary schedule copies.

    Packet sizes, order, routes and startup charges match compile_rewrites.
    This describes an isolated edit only; execution still compiles and checks
    the complete program. Zero-byte packets are omitted, never charged.
    """
    previous = [_RoundLoad(costs), _RoundLoad(costs)]
    current = [_RoundLoad(costs), _RoundLoad(costs)]
    for transfer in seed['rounds'][ri - 1]['transfers']:
        for load in previous:
            load.add(transfer['path'], len(transfer['chunk_ranges']) * chunk_bytes)
    index = {rank: i for i, rank in enumerate(graph.ranks)}
    ready_atoms, representative = 0, None
    for transfer in seed['rounds'][ri]['transfers']:
        ready = 0
        if transfer['action'] == 'copy':
            mask = states[ri - 1][index[transfer['src']]]
            ready = sum(bool(mask & atom_bit(a)) for a in transfer['chunk_ranges'])
        if ready and representative is None:
            representative = transfer['path']
        ready_atoms += ready
        for i, moved in enumerate(((ready + 1) // 2, ready)):
            remaining = len(transfer['chunk_ranges']) - moved
            if remaining:
                current[i].add(transfer['path'], remaining * chunk_bytes)
            if moved:
                previous[i].add(transfer['path'], moved * chunk_bytes)
    return ([previous[0].time_us, current[0].time_us],
            [previous[1].time_us, current[1].time_us], ready_atoms, representative)


def prepare(task, graph, seed, report):
    """Describe a small retiming program using live seed capacity measurements.

    The seed is a proposal context, not a frozen target. Final route/search
    changes are declared in the protocol and all effects are measured again.
    ``report`` must be the trusted internal Replay of this exact seed.
    """
    started = time.monotonic()
    if not report['valid']:
        raise ValueError('invalid retiming seed')
    states, atom_bit = copy_states(task, graph, seed)
    copies = [ri for ri, row in enumerate(seed['rounds']) if ri > 0 and any(t['action'] == 'copy' for t in row['transfers'])]
    rounds = tuple(copies[-5:][::-1])
    bound = PlanContext(_input_digest(task, graph), rounds, len(seed['rounds']), _digest(_body(seed)))
    costs = _PathCosts(graph)
    chunk_bytes = task['data_size']['per_rank_bytes'] // seed['split']
    loads = _round_loads(graph, seed, chunk_bytes, costs)
    short = lambda x: float(f'{x:.4g}')
    rows, edges, pools = [], {}, {}
    for ri in rounds:
        half_times, all_times, ready, path = _preview_pair(
            seed, graph, states, atom_bit, ri, chunk_bytes, costs)
        old_total = loads[ri - 1].time_us + loads[ri].time_us
        rows.append({'round': ri + 1, 'ready': ready,
                     'before': [short(loads[ri - 1].time_us), short(loads[ri].time_us)],
                     'half': [short(value) for value in (*half_times, old_total - sum(half_times))],
                     'all': [short(value) for value in (*all_times, old_total - sum(all_times))]})
        if path:
            for edge in zip(path, path[1:]):
                edges[edge] = short(graph.edges[edge] / 1e9)
        for index in (ri - 1, ri):
            for resource, count in loads[index].resource_bytes.items():
                if count:
                    pools[resource] = max(pools.get(resource, 0), count * 1e6 / graph.resources[resource])
    busiest = sorted(pools, key=lambda r: (-pools[r], r))[:4]
    # Numeric aliases keep edge direction and identity while bounding tokens.
    aliases = {node: i for i, node in enumerate(sorted({n for edge in edges for n in edge}))}
    problem = {'op': 'AG' if task['operator'] == 'all-gather' else 'AR',
               'task': [len(graph.ranks), task['data_size']['per_rank_bytes'], seed['split'], task['max_nonempty_rounds']], 'edits': rows,
               'e': [[aliases[a], aliases[b], cap] for (a,b),cap in sorted(edges.items())],
               'p': [[short(graph.resources[r] / 1e9), short(pools[r])] for r in busiest]}
    evidence = {'elapsed_s': time.monotonic() - started, 'rounds': [r + 1 for r in rounds],
                'seed_sha256': bound.seed_hash, 'input_sha256': bound.input_hash,
                'edge_aliases': {str(v): k for k,v in aliases.items()}, 'pool_ids': busiest}
    return problem, bound, evidence


def execute(task, graph, control, report, bound, proposal):
    """Compile exactly the model's counts and accept only a full Replay gain."""
    started = time.monotonic()
    base_evidence = {'accepted': False, 'selected': False, 'requested': proposal,
                     'control_time_us': report['total_time_us'], 'final_time_us': report['total_time_us'],
                     'errors': [], 'semantics': 'ready_copy_retiming_v1'}
    try:
        if bound.input_hash != _input_digest(task, graph) or len(control['rounds']) != bound.round_count:
            raise ValueError('retiming program does not match input or round structure')
        if not isinstance(proposal, dict) or set(proposal) != {'counts'}:
            raise ValueError('reply must contain only counts')
        counts = proposal['counts']
        if (not isinstance(counts, list) or len(counts) != len(bound.rounds)
                or any(type(v) is not int or v not in (0, 1, 2) for v in counts)):
            raise ValueError('one integer in 0..2 is required for every offered round')
        if not any(counts):
            return control, {**base_evidence, 'reason': 'no_edit', 'elapsed_s': time.monotonic() - started}
        states, atom_bit = copy_states(task, graph, control)
        rewrites = tuple(Rewrite('advance', _moves(control, graph, states, atom_bit, ri, count))
                         for ri, count in zip(bound.rounds, counts) if count)
        context = Context(bound.input_hash, _digest(_body(control)), rewrites, report['total_time_us'])
        candidate, evidence = apply(task, graph, control, context,
            {'apply': list(range(len(rewrites)))}, compiled_here=True, validated_report=report)
        evidence.update(requested=proposal, compiled_counts=list(counts),
                        compiled_base_sha256=context.schedule_hash,
                        compiled_atomic_count=sum(len(m.atoms) for rw in rewrites for m in rw.moves),
                        control_time_us=report['total_time_us'], final_time_us=evidence['after_us'],
                        semantics='ready_copy_retiming_v1', elapsed_s=time.monotonic() - started)
        return candidate, evidence
    except (ValueError, TypeError, KeyError, IndexError) as error:
        return control, {**base_evidence, 'reason': 'rejected', 'errors': [str(error)],
                         'elapsed_s': time.monotonic() - started}
