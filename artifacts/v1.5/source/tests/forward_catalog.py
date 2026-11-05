"""Fixed-catalog forwarding research, retained to reproduce rejected architectures.

Production uses instance-bound retiming programs. This experimental extractor
is not imported by the runtime and has no network or configuration side effects.
"""
from collections import Counter, defaultdict
import math
import time
from src.forward_edit import (Context, Move, Rewrite, _digest, _body, copy_states,
    compile_rewrites, _round_loads)
from src.schedule_edit import _input_digest
from src.route_balance import _PathCosts, _RoundLoad
from src.replay import replay

def extract(task, graph, solution, *, max_groups=4, max_rounds=4,
            max_route_queries=768, max_holders=3, validated_report=None):
    """Build related sender/advance operations from live state, never case names.

    Grouping amplifies changes with a common resource footprint. Per-packet
    alternative holders are ranked by local physical connectivity, not by an
    assumed topology family. Full candidate scores are only heuristic evidence
    until the independent final Replay. This initial catalog is deliberately
    small; it does not claim to enumerate all forwarding schedules.
    """
    started = time.monotonic()
    report = validated_report if validated_report is not None else replay(task, graph, solution)
    if not report["valid"]:
        raise ValueError("forwarding base failed Replay")
    if type(max_groups) is not int or not 0 <= max_groups <= 8:
        raise ValueError("max_groups must be an integer in 0..8")
    chunk_bytes = task["data_size"]["per_rank_bytes"] // solution["split"]
    costs = _PathCosts(graph)
    loads = _round_loads(graph, solution, chunk_bytes, costs)
    snapshots, atom_bit = copy_states(task, graph, solution)
    ranks = graph.ranks
    index = {rank: i for i, rank in enumerate(ranks)}
    outgoing = {rank: {} for rank in ranks}
    incoming = {rank: {} for rank in ranks}
    for (src, dst), bw in graph.edges.items():
        if src in outgoing:
            outgoing[src][dst] = bw
        if dst in incoming:
            incoming[dst][src] = bw

    def affinity(src, dst):
        direct = outgoing[src].get(dst)
        if direct:
            return 1 / direct
        common = outgoing[src].keys() & incoming[dst].keys()
        return min((1 / outgoing[src][s] + 1 / incoming[dst][s] for s in common), default=math.inf)

    grouped = defaultdict(list)
    queries = 0
    copy_rounds = [ri for ri, row in enumerate(solution["rounds"])
                   if any(t["action"] == "copy" for t in row["transfers"])]
    target_rounds = sorted(copy_rounds, key=lambda ri: (-loads[ri].time_us, ri))[:max_rounds]
    for ri in target_rounds:
        rows = solution["rounds"][ri]["transfers"]
        for ti, transfer in enumerate(rows):
            if transfer["action"] != "copy":
                continue
            src, dst = transfer["src"], transfer["dst"]
            tokens = tuple(transfer["chunk_ranges"])
            mask = 0
            for token in tokens:
                mask |= atom_bit(token)
            path = tuple(transfer["path"])
            old_edges, old_resources, old_overhead, old_slope = costs.describe(path)
            footprint = tuple(sorted(old_resources.items()))
            if ri > 0:
                ready = tuple(token for token in tokens if snapshots[ri - 1][index[src]] & atom_bit(token))
                if ready:
                    grouped[("advance", ri, footprint)].append(Move(ri, ti, ready, src, ri - 1, path))
            if queries >= max_route_queries:
                continue
            holders = [rank for rank in ranks if rank not in (src, dst)
                       and snapshots[ri][index[rank]] & mask == mask]
            holders.sort(key=lambda rank: (affinity(rank, dst), index[rank]))
            alternatives = []
            for holder in holders[:max_holders]:
                if queries >= max_route_queries:
                    break
                queries += 1
                try:
                    alternate = tuple(graph.path(holder, dst, size_bytes=chunk_bytes, resource_weight=4.0))
                except ValueError:
                    continue
                edges, resources, overhead, slope = costs.describe(alternate)
                # This order proposes alternate holders; it does not accept an
                # operation or select the model's final combination.
                peak = max([1e6 * loads[ri].edge_bytes[e] / graph.edges[e] for e in edges]
                           + [1e6 * loads[ri].resource_bytes[r] / graph.resources[r] for r in resources] + [0])
                alternatives.append(((peak, overhead + slope * chunk_bytes * len(tokens), index[holder]), holder, alternate, resources))
            if alternatives:
                _, holder, alternate, resources = min(alternatives)
                delta = Counter(resources)
                delta.subtract(old_resources)
                key = ("sender", ri, footprint, tuple(sorted((r, v) for r, v in delta.items() if v)))
                grouped[key].append(Move(ri, ti, tokens, holder, ri, alternate))

    context_base = dict(input_hash=_input_digest(task, graph), schedule_hash=_digest(_body(solution)), before_us=report["total_time_us"])
    candidates = []
    for key, moves in grouped.items():
        rewrite = Rewrite(key[0], tuple(moves))
        local = Context(**context_base, rewrites=(rewrite,))
        candidate = compile_rewrites(solution, local, [0])
        affected = sorted({ri for move in moves for ri in (move.round_index, move.target_round)})
        trial_loads = {}
        for ri in affected:
            load = _RoundLoad(costs)
            for t in candidate["rounds"][ri]["transfers"]:
                load.add(t["path"], len(t["chunk_ranges"]) * chunk_bytes)
            trial_loads[ri] = load
        gain = sum(loads[ri].time_us - trial_loads[ri].time_us for ri in affected)
        candidates.append((gain, rewrite, trial_loads))
    candidates.sort(key=lambda row: (-row[0], row[1].kind, row[1].moves[0].round_index,
                                     row[1].moves[0].transfer_index))
    chosen = candidates[:max_groups]
    context = Context(**context_base, rewrites=tuple(row[1] for row in chosen))
    groups = []
    implicated = {ri for _, rewrite, _ in chosen for move in rewrite.moves for ri in (move.round_index, move.target_round)}
    # The busiest input-declared pools make coupling visible; unlisted edges
    # are still fully counted during compilation and Replay.
    pool_rows = sorted(((loads[ri].resource_bytes[r] * 1e6 / graph.resources[r], ri, r)
                        for ri in implicated for r in graph.resources if loads[ri].resource_bytes[r]), reverse=True)[:6]
    def short(value):
        value = float(f"{value:.4g}")
        return int(value) if value.is_integer() else value
    for gid, (gain, rewrite, trials) in enumerate(chosen):
        m = rewrite.moves[0]
        t = solution["rounds"][m.round_index]["transfers"][m.transfer_index]
        groups.append([rewrite.kind, m.round_index + 1, m.target_round + 1,
                       len(rewrite.moves), sum(len(x.atoms) for x in rewrite.moves),
                       [t["src"], m.source, t["dst"]], short(gain),
                       [short((trials[ri].resource_bytes[r] - loads[ri].resource_bytes[r]) * 1e6 / graph.resources[r])
                        if ri in trials else 0 for _, ri, r in pool_rows]])
    memberships = [{(m.round_index, m.transfer_index, atom) for m in rw.moves for atom in m.atoms}
                   for rw in context.rewrites]
    conflicts = [[a, b] for a in range(len(chosen)) for b in range(a + 1, len(chosen))
                 if memberships[a] & memberships[b]]
    edges = {}
    for _, rewrite, _ in chosen:
        move = rewrite.moves[0]
        old = solution["rounds"][move.round_index]["transfers"][move.transfer_index]["path"]
        for path in (old, move.path):
            for edge in zip(path, path[1:]):
                edges[edge] = short(graph.edges[edge] / 1e9)
    problem = {"operator": task["operator"], "round_us": [[ri + 1, short(loads[ri].time_us)] for ri in sorted(implicated)],
               "pools": [[ri + 1, r, short(graph.resources[r] / 1e9), short(busy)] for busy, ri, r in pool_rows],
               "edge_examples": [[src, dst, bw] for (src, dst), bw in sorted(edges.items())],
               "groups": groups, "conflicts": conflicts}
    info = {"elapsed_s": time.monotonic() - started, "route_queries": queries,
            "candidate_groups": len(candidates), "group_count": len(chosen),
            "source_input_sha256": context.input_hash, "baseline_sha256": context.schedule_hash,
            "predicted_gains_us": [g for g, _, _ in chosen]}
    return problem, context, info


def rebase(task, graph, baseline, target, context, *, proposal=None, validated_report=None):
    """Carry the same semantic edits over route changes / packet boundaries.

    Every original source, destination, atom and round must still match. An
    advance retains the target's optimized route because the model instruction
    changes timing only. A sender rewrite retains its explicit proposed path.
    No model action is silently dropped; incompatible catalogs are rejected.
    """
    if (context.input_hash != _input_digest(task, graph)
            or context.schedule_hash != _digest(_body(baseline))
            or baseline["split"] != target["split"]):
        raise ValueError("incompatible forwarding base")
    ids = list(range(len(context.rewrites))) if proposal is None else proposal.get("apply") if isinstance(proposal, dict) and set(proposal) == {"apply"} else None
    if (not isinstance(ids, list) or any(type(i) is not int or not 0 <= i < len(context.rewrites) for i in ids)
            or len(ids) != len(set(ids))):
        raise ValueError("invalid forwarding proposal")
    positions = {}
    for ri, row in enumerate(target["rounds"]):
        for ti, transfer in enumerate(row["transfers"]):
            if transfer["action"] == "copy":
                for atom in transfer["chunk_ranges"]:
                    positions[ri, transfer["src"], transfer["dst"], atom] = ti
    rewrites = []
    for group_id, rewrite in enumerate(context.rewrites):
        moves = []
        if group_id not in ids:
            rewrites.append(Rewrite(rewrite.kind, ()))
            continue
        for move in rewrite.moves:
            old = baseline["rounds"][move.round_index]["transfers"][move.transfer_index]
            by_transfer = defaultdict(list)
            for atom in move.atoms:
                key = move.round_index, old["src"], old["dst"], atom
                if key not in positions:
                    raise ValueError("forwarding actions changed during route refinement")
                by_transfer[positions[key]].append(atom)
            for ti, atoms in by_transfer.items():
                path = (tuple(target["rounds"][move.round_index]["transfers"][ti]["path"])
                        if rewrite.kind == "advance" else move.path)
                moves.append(Move(move.round_index, ti, tuple(atoms), move.source, move.target_round, path))
        rewrites.append(Rewrite(rewrite.kind, tuple(moves)))
    report = validated_report if validated_report is not None else replay(task, graph, target)
    if not report["valid"]:
        raise ValueError("forwarding target failed Replay")
    return Context(context.input_hash, _digest(_body(target)), tuple(rewrites), report["total_time_us"])


