"""Reduce on physical row and column cliques, then cross the access-domain cut."""
from collections import defaultdict
from math import floor, gcd
from time import monotonic

from . import grid_matching, mesh_elbows
from .capacity_backend import native
from .collectives import _initial
from .grid_axes import axes
from .reduction_parents import regenerate
from .replay import certify


class _Grid:
    def __init__(self, task, graph, catalog, blocks, split):
        self.task,self.graph,self.catalog=task,graph,catalog
        self.blocks,self.split=blocks,split
        self.size=task['data_size']['per_rank_bytes']//split
        self.costs=catalog.capacity_costs
        self.count=len(blocks)
        self.width=len(blocks[0][0])
        self.common={rank:{p for p,_,_ in graph.adjacency[rank] if p in graph.switches
                           and (p,rank) in graph.edges} for rank in graph.ranks}
        self.routes={}
        self.roots=self.choose_roots()

    def choose_roots(self):
        # One column collects each local row's stripe in a single direct hop.
        # Assign different column roots to successive stripes to distribute
        # endpoint work and favor roots with strong inbound AND outbound access.
        placement=[]
        for i,block in enumerate(self.blocks):
            by_column=[]
            for c in range(self.width):
                column=[row[c] for row in block]
                def capacity(rank):
                    links=[min(self.graph.edges.get((r,rank),0.),self.graph.edges.get((rank,r),0.))
                           for r in column if r!=rank]
                    ports=sorted((min(self.graph.edges[rank,p],self.graph.edges[p,rank])
                                  for p in self.common[rank]),reverse=True)
                    uplink=ports[min(2,len(ports)-1)] if ports else 0.
                    return min(2.7*min(links),uplink), min(links),sum(links)
                by_column.append((column,sorted(column,key=capacity,reverse=True)))
            group=[]
            for s in range(self.split):
                column,ranked=by_column[(s+i)%self.width]
                group.append((column,ranked[s//self.width%len(ranked)]))
            placement.append(group)
        return list(zip(*placement))

    def paths(self,u,v,direct):
        key=u,v,direct
        if key not in self.routes:
            options=[(u,v)] if (u,v) in self.graph.edges else []
            if not direct or not options:
                options.extend((u,p,v) for p in sorted(self.common[u]&self.common[v]))
            if not options: options=self.catalog.options(u,v)
            self.routes[key]=tuple(self.costs.compile(p) for p in options)
        return self.routes[key]

    def bypass_budget(self, recipe):
        # Before crossing the cut most stripes still travel on direct links.
        # Estimate how many whole satellite rows could fit simultaneously on
        # the two-hop planes. Count each plane's tighter ingress or egress cap.
        graph=self.graph
        capacity=0.
        for port in graph.switches:
            sides=([(r,port) for r in graph.ranks if (r,port) in graph.edges],
                   [(port,r) for r in graph.ranks if (port,r) in graph.edges])
            if not all(sides):
                continue
            limits=[]
            for edges in sides:
                shared=set(graph.edge_resources[edges[0]])
                for edge in edges[1:]:
                    shared.intersection_update(graph.edge_resources[edge])
                limits.append(min([sum(graph.edges[e] for e in edges),
                                   *(graph.resources[p] for p in shared)]))
            capacity+=min(limits)
        duration=max(0.,min(recipe[1][0],recipe[4][0])-graph.startup_us-graph.switch_overhead_us)
        rows=len(self.blocks[0])
        # Leave half the window for local overflow and endpoint contention.
        return min(self.split//4, int(capacity*duration/(2e6*self.size*rows)))

    def stages(self, bypass_count, flexible_local):
        # A few satellites bypass their regional root. Their ready row sums
        # use otherwise idle switch capacity during the weak-axis reduction.
        # Other stripes retain ordinary regional aggregation.
        selected={}
        stride=max(1,self.split//3)
        while gcd(stride,self.split)!=1:stride-=1
        for j in range(bypass_count):
            s=(j*stride)%self.split
            root=(s+s//self.width)%self.count
            peers=[i for i in range(self.count) if i!=root]
            selected.setdefault(s,set()).add(peers[(s+j//self.split)%len(peers)])
        rows=[defaultdict(list) for _ in range(6)]
        for s,groups in enumerate(self.roots):
            root=groups[(s+s//self.width)%self.count][1]
            removed=selected.get(s,set())
            live=[hub for i,(_,hub) in enumerate(groups) if i not in removed]
            for i,(block,(middle,hub)) in enumerate(zip(self.blocks,groups)):
                for j,(row,rep) in enumerate(zip(block,middle)):
                    for rank in row:
                        if rank!=rep:
                            rows[0][rank,rep,False,not flexible_local].append(s)
                            rows[5][rep,rank,True,not flexible_local].append(s)
                    next_hub=live[(s+i+j)%len(live)] if i in removed else hub
                    if next_hub!=rep:
                        direct=bool(bypass_count and i not in removed)
                        rows[1][rep,next_hub,False,direct].append(s)
                        rows[4][next_hub,rep,True,direct].append(s)
                if i not in removed and hub!=root:
                    rows[2][hub,root,False,False].append(s)
                    rows[3][root,hub,True,False].append(s)
        return rows

    def price(self,stages,parent=None):
        recipe=[]
        for i,stage in enumerate(stages):
            if parent is not None and i in (0,5):
                recipe.append(parent[i])
                continue
            jobs,entries=[],[]
            for (u,v,copy,local),slots in stage.items():
                options=self.paths(u,v,local)
                if not options or len(options)>128:return None
                jobs.append((options,len(slots)))
                entries.append((options,copy,slots))
            if not entries:return None
            result=native.optimize_groups(len(self.costs.indices),jobs,2,self.costs.round_cache,3,self.size,True)
            picks,duration=iter(result[0]),result[2]
            batch=defaultdict(list)
            for options,copy,slots in entries:
                n=min(3,len(options),len(slots))
                for j in range(n):
                    choice=options[next(picks)][3]
                    batch[choice,copy].extend(slots[len(slots)*j//n:len(slots)*(j+1)//n])
            recipe.append((duration,batch))
        return recipe

    def materialize(self,recipe):
        rows=[]
        for duration,batch in recipe:
            moves=[]
            for (p,copy),slots in batch.items():
                _,latency,serial,_=self.costs.compile(p)
                width=max(1,floor((duration-latency)/(self.size*serial)+1e-10))
                labels=[f'c{s}_sum' if copy else f'{p[0]}_c{s}' for s in slots]
                for i in range(0,len(labels),width):
                    moves.append(dict(src=p[0],dst=p[-1],path=list(p),
                                      action='copy' if copy else 'reduce',chunk_ranges=labels[i:i+width]))
            rows.append(dict(round=len(rows)+1,transfers=moves))
        out=_initial(self.task,self.graph.ranks,self.split)
        out.update(rounds=regenerate(rows,self.graph.ranks,self.split),round_count=len(rows))
        return out


def improve(task,graph,plan,report,catalog,allowed):
    if (native is None or task['operator']!='all-reduce' or not 64<=len(graph.ranks)<=256
            or task['max_nonempty_rounds']<6 or len(graph.switches)>32):
        return plan,report,None
    blocks=axes(graph)
    if not blocks:return plan,report,None
    width=len(blocks[0][0])
    choices=[s for s in allowed if width<=s<=64 and s%width==0]
    if not choices:return plan,report,None
    started=monotonic()
    grid=_Grid(task,graph,catalog,blocks,max(choices))
    rank_set=set(graph.ranks)
    flexible=any(pools and u in rank_set and v in rank_set for (u,v),pools in graph.edge_resources.items())
    stages=grid.stages(0,flexible)
    recipe=grid.price(stages)
    if recipe is None:return plan,report,None
    best=sum(t for t,_ in recipe)
    count=0 if flexible else grid.bypass_budget(recipe)
    if count:
        alternate=grid.stages(count,False)
        other=grid.price(alternate,recipe)
        if other is not None:
            other=grid_matching.polish(grid,alternate,other)
            price=sum(t for t,_ in other)
            if price<best:best,recipe=price,other
    elbows = mesh_elbows.improve(grid, recipe)
    if elbows is not None:
        elbow_us = sum(t for t, _ in elbows)
        if elbow_us < best * .98:
            best, recipe = elbow_us, elbows
    before=report['total_time_us']
    detail=dict(domains=grid.count,width=width,predicted_us=best,elapsed_s=round(monotonic()-started,6))
    if best>=before*.995:
        return plan,report,detail
    candidate=grid.materialize(recipe)
    checked=certify(task,graph,candidate)
    detail.update(valid=checked['valid'],elapsed_s=round(monotonic()-started,6))
    if checked['valid'] and abs(checked['total_time_us']-best)<max(.0001,best*1e-9):
        return candidate,checked,detail
    return plan,report,detail
