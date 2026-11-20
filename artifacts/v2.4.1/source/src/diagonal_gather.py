"""Price a two-hop broadcast per source coordinate on an observed direct grid."""
from collections import defaultdict

from .capacity_backend import native
from .balanced_stages import BalancedStageSpace
from .shard_pipeline import profiles_for
from .grid_axes import axes as physical_axes


def axes(order, edges):
    n = len(order)
    if not 32 <= n <= 256 or n & (n - 1) or (order[0], order[1]) not in edges:
        return None

    def linked(mask):
        return sum((r, order[i ^ mask]) in edges for i, r in enumerate(order)) * 8 >= 7*n

    def axis_length(stride):
        width = 1
        while width < 16 and 2*width*stride <= n and all(
                linked(stride*m) for m in range(width, 2*width)):
            width *= 2
        return width

    inner = axis_length(1)
    if inner < 2 or inner >= n or linked(inner+1):
        return None
    outer = axis_length(inner)
    if outer < 2 or not 2 <= n//(inner*outer) <= 8:
        return None
    return inner, outer


def price(task, graph, parent, catalog, allowed):
    if parent['split'] != 1 or 1 not in allowed or task['max_nonempty_rounds'] < 3:
        return None
    order = graph.ranks
    dims = axes(order, graph.edges)
    if dims is None:
        blocks = physical_axes(graph)
        if not blocks or len(blocks) > 8:
            return None
        dims = len(blocks[0][0]), len(blocks[0])
        if any(n & (n-1) for n in (*dims, len(blocks))):
            return None
        order = [rank for block in blocks for row in block for rank in row]
    inner, outer = dims
    tile, n = inner*outer, len(order)
    space = BalancedStageSpace(task, graph, 1, order=order, catalog=catalog)
    prefix = (n//tile-1).bit_length()
    moves = space._movements(0, prefix, atoms=True)[0]
    routes, _ = space._route_counts({k: len(v) for k, v in moves.items()})
    waves = [defaultdict(list) for _ in range(3)]
    for item in space._materialize(routes, moves):
        waves[0][tuple(item['path'])].extend(item['chunk_ranges'])
    atoms = [f'{rank}_c0' for rank in order]
    costs = catalog.capacity_costs
    pairs = {}
    def paths(src, dst):
        if (src, dst) not in pairs:
            options = (((src, dst),) if (src, dst) in graph.edges else catalog.options(src, dst))
            pairs[src, dst] = tuple(costs.compile(p) for p in options)
        return pairs[src, dst]

    packets, origins = [], []
    for i, owner in enumerate(order):
        for delta in range(1, tile):
            dst = order[i ^ delta]
            column, row = delta % inner, delta // inner
            if row and column:
                options = paths(order[i ^ column], dst) + paths(order[i ^ (row*inner)], dst)
                depth = 2
            else:
                options = paths(owner, dst)
                depth = 1
            if not options or len(options) > 128:
                return None
            packets.append((options, space.size*(n//tile), space.size))
            origins.append((i, depth))
    # Price the two logical layers separately. Choose a relay for each diagonal
    # destination. The origin's own domain can start a full round earlier.
    for depth in (1, 2):
        ids = [j for j, (_, d) in enumerate(origins) if d == depth]
        subset = [packets[j] for j in ids]
        selected, _, _, _ = native.optimize_round(len(costs.indices), subset, 2, costs.round_cache)
        for j, choice in zip(ids, selected):
            i, _ = origins[j]
            path = packets[j][0][choice][3]
            for source in range(i % tile, n, tile):
                waves[depth - int(source == i)][path].append(atoms[source])
    profiles = profiles_for(waves, costs)
    factors = [s for s in allowed if s <= 4 and s+2 <= task['max_nonempty_rounds']
               and s*n*(n-1) <= 530000]
    if not factors:
        return None
    values = native.pipeline_costs(len(costs.indices), space.size, profiles, factors)
    factor, times = min(zip(factors, values), key=lambda item: sum(item[1]))
    return waves, factor, times, None
