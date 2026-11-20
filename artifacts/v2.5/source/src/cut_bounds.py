"""Necessary AllGather communication across observed rank and access-domain cuts."""
from collections import defaultdict

from .group_sketch import access_groups


def covered_capacity(graph, links):
    """Upper bound total cut throughput by covering its edges with pools or links."""
    uncovered = set(links)
    members = defaultdict(list)
    selected = []
    for e in uncovered:
        for pool in graph.edge_resources[e]:
            members[pool].append(e)
    for pool, edges in sorted(members.items(), key=lambda row:
                              graph.resources[row[0]] / sum(graph.edges[e] for e in row[1])):
        included = [e for e in edges if e in uncovered]
        saved = sum(graph.edges[e] for e in included) - graph.resources[pool]
        if saved > 0:
            selected.append(graph.resources[pool])
            uncovered.difference_update(included)
    return sum(selected) + sum(graph.edges[e] for e in uncovered)


def gather_bound(task, graph):
    ranks, edges = graph.ranks, graph.edges
    total = task['data_size']['per_rank_bytes']
    incoming, ports = defaultdict(list), defaultdict(set)
    groups = access_groups(graph)
    domain = {r: i for i, group in enumerate(groups) for r in group}
    for u,v in edges:
        if v in domain:
            incoming[v].append((u,v))
            if u not in domain:
                ports[u].add(domain[v])
        if u in domain and v not in domain:
            ports[v].add(domain[u])
    local = dict(domain)
    local.update((switch,next(iter(owners))) for switch,owners in ports.items() if len(owners)==1)
    best = 0.
    for r in ranks:
        if incoming[r]:
            best = max(best,(len(ranks)-1) * total / covered_capacity(graph,incoming[r]))
    if len(groups)>1:
        for reverse in (False,True):
            cuts=defaultdict(list)
            for u,v in edges:
                a,b = (v,u) if reverse else (u,v)
                if b in local and local.get(a)!=local[b]:
                    cuts[local[b]].append((u,v))
            if len(cuts)!=len(groups):
                continue
            for d,links in cuts.items():
                chunks=len(groups[d]) if reverse else len(ranks)-len(groups[d])
                best=max(best, chunks*total/covered_capacity(graph,links))
            all_links=[edge for es in cuts.values() for edge in es]
            # Every foreign group must receive each original at least once.
            # An unlabelled backbone can fan out only via another rank; each
            # transfer has one sender, one receiver and explicit point-to-point edges.
            best=max(best,(len(groups)-1)*len(ranks)*total/covered_capacity(graph,all_links))
    return 1e6*best
