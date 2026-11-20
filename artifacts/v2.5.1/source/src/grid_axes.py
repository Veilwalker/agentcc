"""Recover orthogonal rank cliques from the directed physical links."""
from collections import defaultdict


def axes(graph):
    # Both phase construction and verified refinement use the same immutable
    # graph in one request, including when no direct grid was found.
    if not hasattr(graph, '_physical_grid'):
        graph._physical_grid = _recover(graph)
    return graph._physical_grid


def _recover(graph):
    ranks = graph.ranks
    if not 16 <= len(ranks) <= 512:
        return None
    positions = {rank: i for i, rank in enumerate(ranks)}
    neighbors = [set() for _ in ranks]
    for u, v in graph.edges:
        if u in positions and v in positions and (v, u) in graph.edges:
            neighbors[positions[u]].add(positions[v])
    if any(not 3 <= len(adjacent) <= 64 for adjacent in neighbors):
        return None

    pairs = ((a, b) for a, adjacent in enumerate(neighbors) for b in adjacent if a < b)
    edge_ids = {pair: i for i, pair in enumerate(pairs)}
    parents = list(range(len(edge_ids)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    # Two edges of the same triangle belong to one line of the local grid.
    for (a, b), edge_id in edge_ids.items():
        for c in neighbors[a] & neighbors[b]:
            if c > b:
                line = root(edge_id)
                parents[root(edge_ids[a, c])] = line
                parents[root(edge_ids[b, c])] = line
    lines_by_id = defaultdict(set)
    for pair, edge_id in edge_ids.items():
        lines_by_id[root(edge_id)].update(pair)
    lines = [nodes for nodes in lines_by_id.values() if len(nodes) >= 3]
    membership = defaultdict(list)
    for i, line in enumerate(lines):
        for node in line:
            membership[node].append(i)
    if any(len(membership[i]) != 2 for i in range(len(ranks))):
        return None

    # Rows and columns in a valid component form a bipartite intersection
    # graph, with exactly one physical rank at each row/column intersection.
    touching = defaultdict(set)
    for a, b in membership.values():
        touching[a].add(b)
        touching[b].add(a)

    def median_bandwidth(family):
        values = sorted(graph.edges[ranks[a], ranks[b]]
                        for line in family for a in line for b in line
                        if b in neighbors[a] and a != b)
        return values[len(values) // 2]

    remaining = set(range(len(lines)))
    blocks = []
    while remaining:
        head = min(remaining)
        colors = {head: 0}
        queue = [head]
        while queue:
            line = queue.pop()
            for other in touching[line]:
                if other not in colors:
                    colors[other] = 1 - colors[line]
                    queue.append(other)
                elif colors[other] == colors[line]:
                    return None
        remaining.difference_update(colors)
        sides = [[lines[i] for i, color in colors.items() if color == side] for side in (0, 1)]
        if any(not 3 <= len(side) <= 16 for side in sides):
            return None
        rows, columns = sorted(sides, key=median_bandwidth, reverse=True)
        rows.sort(key=min)
        columns.sort(key=min)
        cells = [[row & column for column in columns] for row in rows]
        if any(len(cell) != 1 for row in cells for cell in row):
            return None
        blocks.append([[ranks[next(iter(cell))] for cell in row] for row in cells])
    sizes = {(len(block), len(block[0])) for block in blocks}
    return blocks if 2 <= len(blocks) <= 16 and len(sizes) == 1 else None
