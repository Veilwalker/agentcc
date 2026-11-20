"""Stencil for independent multicast trees on measured local access domains."""
from collections import defaultdict
from .capacity_backend import native
from .group_sketch import access_groups
from .shard_pipeline import profiles_for


def direct_cliques(groups, graph):
    # Local links are inferred from both declared directions. If a domain has
    # overlapping axes instead of separated neighborhoods this family is skipped.
    layout = {}
    for group in groups:
        neighbors = {u: [v for v in group if (u,v) in graph.edges and (v,u) in graph.edges]
                     for u in group}
        if not any(neighbors.values()):
            for u in group:
                layout[u] = (group, group.index(u))
            continue
        seen = set()
        for u in group:
            if u in seen:
                continue
            component, queue = set([u]), [u]
            while queue:
                for v in neighbors[queue.pop()]:
                    if v not in component:
                        component.add(v)
                        queue.append(v)
            seen.update(component)
            members = [r for r in group if r in component]
            if not 2 <= len(members) <= 24 or sum(len(neighbors[r]) for r in members) < .75*len(members)*(len(members)-1):
                return None
            for pos,r in enumerate(members):
                layout[r] = (members,pos)
    return layout


def price(task, graph, catalog, allowed):
    # The guard and final flow search share one request-local numeric recipe.
    # Its paths and costs depend only on this graph and the allowed shard sizes.
    key = tuple(allowed)
    cached = getattr(catalog, 'fanout_recipe', None)
    if cached is None or cached[0] != key:
        cached = key, _price(task, graph, catalog, allowed)
        catalog.fanout_recipe = cached
    return cached[1]


def _price(task, graph, catalog, allowed):
    nodes = graph.ranks
    if min((len(catalog.outgoing[r]) for r in nodes), default=0) < 3 or 1 not in allowed:
        return None
    groups = access_groups(graph)
    if not 2 <= len(groups) <= 8 or min(map(len,groups)) < 4 or max(map(len,groups)) > 96:
        return None
    layout = direct_cliques(groups,graph)
    if layout is None:
        return None
    n, idx = len(nodes), catalog.rank_index
    size = task['data_size']['per_rank_bytes']
    sources = {r: tuple(port for port,_ in catalog.outgoing[r]) for r in nodes}
    sinks = {r: set(port for port,_ in catalog.incoming[r]) for r in nodes}
    atoms=[f'{r}_c0' for r in nodes]
    rows=[defaultdict(list) for _ in range(3)]
    offsets={r:(d,i) for d,grp in enumerate(groups) for i,r in enumerate(grp)}
    for s,owner in enumerate(nodes):
        home,i=offsets[owner]
        for d,grp in enumerate(groups):
            pos=(i+(d-home))%len(grp);root=grp[pos];remote=int(d!=home)
            if remote:rows[0][owner,root,True].append(atoms[s])
            for j,peer in enumerate(grp):
                if peer==root:continue
                # Keep a balanced fan-out. All arcs must resolve using actual
                # directed links. Numerical positions specify no connectivity.
                members, _ = layout[peer]
                root_members, root_position = layout[root]
                hub = root if members is root_members else members[root_position % len(members)]
                if hub == peer:
                    hub = root
                rows[remote+int(hub!=root)][hub,peer,False].append(atoms[s])
    cache={};costs=catalog.capacity_costs;waves=[]
    for row in rows:
        if not row:continue
        jobs=[];payload=[]
        for (u,v,remote),ts in row.items():
            key=u,v,remote
            if key not in cache:
                local=[(u,v)] if (u,v) in graph.edges else []
                ports=[port for port in sources[u] if port in sinks[v]]
                if ports:
                    start=(idx[u]*31+idx[v]*17)%len(ports)
                    # Sample across the actual ports; different directed pairs
                    # select different subsets and can use distinct shared pools.
                    local.extend((u,ports[(start+i*max(1,len(ports)//4))%len(ports)],v)
                                 for i in range(min(4,len(ports))))
                if remote or not local:
                    try:all_paths=catalog.options(u,v)
                    except ValueError:all_paths=()
                    if len(all_paths)>12:
                        start=(idx[u]+17*idx[v])%len(all_paths)
                        all_paths=[all_paths[(start+3*i)%len(all_paths)] for i in range(12)]
                    local.extend(all_paths)
                if not local:local.append(tuple(graph.path(u,v,size)))
                cache[key]=tuple(costs.compile(p) for p in dict.fromkeys(local))
            opts=cache[key]
            jobs.append((opts,len(ts)))
            payload.append(ts)
        choices,_,_,_=native.optimize_groups(len(costs.indices),jobs,1,costs.round_cache,3,size,True)
        choices=iter(choices)
        selected=defaultdict(list)
        for (opts,count),ts in zip(jobs,payload):
            lanes=min(3,count,len(opts))
            for j in range(lanes):
                path=opts[next(choices)][3]
                selected[path].extend(ts[count*j//lanes:count*(j+1)//lanes])
        waves.append(selected)
    factors=[f for f in allowed if f<=4 and f*n*(n-1)<=280000 and f+len(waves)-1<=task['max_nonempty_rounds']]
    if not factors:return None
    times=native.pipeline_costs(len(costs.indices),size,profiles_for(waves,costs),factors)
    f,d=min(zip(factors,times),key=lambda v:sum(v[1]))
    return sum(d),(waves,f,d,None)
