"""Compact, input-derived planning facts; full topology stays in the constructor."""

from collections import Counter, defaultdict
import statistics


def compact_summary(graph) -> dict:
    ranks = set(graph.ranks)
    kinds = defaultdict(list)
    resource_edges = defaultdict(list)
    neighbors = defaultdict(list)
    for edge, capacity in graph.edges.items():
        src, dst = edge
        kind = ("r" if src in ranks else "s") + ("r" if dst in ranks else "s")
        kinds[kind].append(capacity)
        if src in ranks and dst in graph.switches:
            neighbors[src].append(dst)
        for resource in graph.edge_resources.get(edge, ()):
            resource_edges[resource].append(edge)
    domains = Counter(tuple(sorted(neighbors[rank])) for rank in graph.ranks)

    def rate(value):
        return float(f"{value / 1e9:.3g}")

    pool_classes = {}
    for name, capacity in graph.resources.items():
        members = resource_edges[name]
        total = sum(graph.edges[edge] for edge in members)
        incident = set(node for edge in members for node in edge if node in ranks)
        edge_kinds = Counter(("r" if src in ranks else "s") + ("r" if dst in ranks else "s")
                             for src, dst in members)
        signature = (capacity, len(members), len(incident), round(total / capacity, 6), tuple(sorted(edge_kinds.items())))
        if signature in pool_classes:
            pool_classes[signature][1]["count"] += 1
            continue
        pool_classes[signature] = (total / capacity, {
            "GBs": rate(capacity), "edges": len(members), "ranks": len(incident),
            "edge_sum_over_capacity": round(total / capacity, 1),
            "member_samples": [list(edge) for edge in members[:2]], "count": 1,
        })
    pools = list(pool_classes.values())
    pools.sort(key=lambda pair: pair[0], reverse=True)
    return {
        "ranks": len(graph.ranks), "switches": len(graph.switches),
        "edges_by_kind": {kind: [len(values), rate(min(values)), rate(statistics.median(values)), rate(max(values))]
                          for kind, values in sorted(kinds.items())},
        "one_way_edges": sum((dst, src) not in graph.edges for src, dst in graph.edges),
        "unequal_reverse_edges": sum((dst, src) in graph.edges and graph.edges[dst, src] != bw
                                     for (src, dst), bw in graph.edges.items()),
        "attachment_groups": sorted(Counter(domains.values()).items()),
        "edge_examples": [[src, dst, rate(bw), len(graph.edge_resources.get((src, dst), ()))]
                          for (src, dst), bw in sorted(graph.edges.items(), key=lambda item: item[1])[:4]],
        "neighborhood_examples": [[rank, neighbors[rank][:4]] for rank in graph.ranks[:2]],
        "shared_pool_count": len(graph.resources), "tight_pools": [pool for _, pool in pools[:3]],
        "overlapping_resource_edges": sum(len(ids) > 1 for ids in graph.edge_resources.values()),
        "overhead_us": [graph.startup_us, graph.switch_overhead_us],
        "topology_fingerprint": graph.input_sha256[:12],
    }
