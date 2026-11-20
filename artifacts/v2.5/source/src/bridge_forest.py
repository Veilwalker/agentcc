"""Build a two-hop local reduction forest covering irregular access domains."""
from collections import defaultdict

from .capacity_backend import native
from .collectives import _initial
from .reduction_parents import emit_batches, regenerate
from .replay import certify


class LocalCover:
    def __init__(self, forest, members):
        self.members = members
        graph = forest.graph
        # All child and parent arcs need an actual two-hop path. A port only
        # qualifies when both the ingress and egress rank edges are declared.
        ports = [{p for p in forest.ports[r] if (p, r) in graph.edges} for r in members]
        self.near = [sum(1 << j for j, other in enumerate(ports) if spoke & other)
                     for spoke in ports]
        self.all_members = (1 << len(members)) - 1
        self.layouts = {}

    def representatives(self, pivot):
        if pivot in self.layouts:
            return self.layouts[pivot]
        width = len(self.members)
        order = [(pivot + step) % width for step in range(1, width)]
        reachable = self.near[pivot]
        eligible = [i for i in order if reachable & (1 << i)]
        missing = self.all_members & ~reachable
        helpers = [pivot]
        while missing:
            if not eligible:
                return None
            bridge = max(eligible, key=lambda i: (missing & self.near[i]).bit_count())
            if not missing & self.near[bridge]:
                return None
            helpers.append(bridge)
            eligible.remove(bridge)
            missing &= ~self.near[bridge]
        # Give the second relay the least-covered ranks. Two representatives
        # make ingress balance possible without a large fan-in at the root.
        while len(helpers) < 3 and eligible:
            groups = defaultdict(int)
            for j in range(width):
                overlap = sum(bool(self.near[h] & (1 << j)) for h in helpers)
                groups[overlap] |= 1 << j

            def score(i):
                return sum((mask & self.near[i]).bit_count() / max(1, n)
                           for n, mask in groups.items())
            bridge = max(eligible, key=score)
            helpers.append(bridge)
            eligible.remove(bridge)
        parents = {}
        for j, rank in enumerate(self.members):
            if j not in helpers:
                parents[rank] = tuple(self.members[h] for h in helpers if self.near[h] & (1 << j))
        result = self.members[pivot], [self.members[h] for h in helpers[1:]], parents
        self.layouts[pivot] = result
        return result


def construct(task, forest, split, maximum_us=None):
    if native is None or not 3 <= len(forest.groups) <= 16 or task['max_nonempty_rounds'] < 6:
        return None
    if any(len(group) > 64 for group in forest.groups):
        return None
    covers = [LocalCover(forest, group) for group in forest.groups]
    # Record a group of possible arcs and price all slots with that same choice
    # in one native job. Reduction and broadcast can choose different parents.
    stages = [defaultdict(list) for _ in range(6)]
    for slot in range(split):
        roots = []
        for domain, cover in enumerate(covers):
            size = len(cover.members)
            pivot = (slot * size // split + domain) % size
            layout = cover.representatives(pivot)
            if layout is None:
                return None
            root, bridges, parents = layout
            roots.append(root)
            for leaf, choices in parents.items():
                stages[0][tuple((leaf, b) for b in choices)].append(slot)
                stages[5][tuple((b, leaf) for b in choices)].append(slot)
            for bridge in bridges:
                stages[1][((bridge, root),)].append(slot)
                stages[4][((root, bridge),)].append(slot)
        global_root = roots[slot % len(roots)]
        for root in roots:
            if root != global_root:
                stages[2][((root, global_root),)].append(slot)
                stages[3][((global_root, root),)].append(slot)
    rows = []
    projected = 0.
    for i, stage in enumerate(stages):
        jobs, entries = [], []
        for arcs, slots in stage.items():
            paths = tuple(p for src, dst in arcs for p in forest.paths(src, dst, i not in (2, 3)))
            if not 1 <= len(paths) <= 128:
                return None
            jobs.append((paths, len(slots)))
            entries.append((paths, slots))
        if not jobs:
            continue
        choices, _, duration, _ = native.optimize_groups(len(forest.costs.indices), jobs, 2,
                                                         forest.costs.round_cache, 4, forest.size, True)
        projected += duration
        if maximum_us is not None and projected >= maximum_us:
            return None
        selections = iter(choices)
        batches = defaultdict(list)
        for paths, slots in entries:
            lanes = min(4, len(paths), len(slots))
            for j in range(lanes):
                p = paths[next(selections)][3]
                batches[p, i >= 3].extend(slots[len(slots) * j // lanes:len(slots) * (j + 1) // lanes])
        moves = emit_batches(batches, duration, forest.size, forest.costs)
        rows.append(dict(round=len(rows) + 1, transfers=moves))
    candidate = _initial(task, forest.graph.ranks, split)
    candidate.update(rounds=regenerate(rows, forest.graph.ranks, split), round_count=len(rows))
    checked = certify(task, forest.graph, candidate)
    if checked['valid'] and abs(checked['total_time_us'] - projected) <= max(.0001, projected * 1e-9):
        return candidate, checked
    return None
