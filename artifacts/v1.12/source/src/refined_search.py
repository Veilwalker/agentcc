"""Capacity search with interchangeable, state-equivalent stage kernels.

Cost comparisons use the same round-start ownership state. A holder change is
allowed only for AllGather, where every member of the processed dimension group
already owns the identical atom set. No topology name participates in a choice.
"""
from .balanced_stages import BalancedStageSpace
from .holder_stages import HolderStageSpace
from .collectives import locality_order
from .fabric_routes import RouteCatalog
from .schedule_edit import _input_digest
from .stage_program import StageSpace
from .stage_search import SearchSpace, Profile


class RefinedProfile(Profile):
    def __init__(self, plain, balanced, holders=None):
        super().__init__(plain, balanced, {})
        self.holders = holders
        balanced.movement_counts = plain.movement_counts
        if holders is not None:
            holders.movement_counts = plain.movement_counts

    def transition(self, i, w, phase=None):
        key = i, w, phase
        if key not in self.selected:
            variants = [self.plain]
            old = self.plain.transition(i, w)
            if any(row[4] > 1 for rows in old[:2] for row in rows):
                variants.append(self.striped)
            if self.holders is not None and i >= 3 and w == 1:
                variants.append(self.holders)
            def cost(space):
                value = space.transition(i,w)[2]
                return value if phase is None else space.phase_times[i,w][phase]
            self.selected[key] = min(variants, key=cost)
        return self.selected[key].transition(i,w)


class RefinedSearchSpace(SearchSpace):
    def __init__(self, task, graph, split):
        n = len(graph.ranks)
        if not 2 <= n <= 512 or n & (n-1):
            raise ValueError('stage search requires 2..512 power-of-two ranks')
        self.task, self.graph, self.input_hash = task, graph, _input_digest(task,graph)
        self.physical_dimensions = d = (n-1).bit_length()
        self.independent_ar = task['operator'] == 'all-reduce'
        self.dimensions = d * (2 if self.independent_ar else 1)
        self.limit = min(self.dimensions, task['max_nonempty_rounds'])
        catalog = RouteCatalog(graph, task['data_size']['per_rank_bytes']//split, branch_ports=True)
        orders = [list(graph.ranks)]
        local = locality_order(graph)
        if local != orders[0]:
            orders.append(local)
        width = 5 if self.independent_ar else 3
        plain = [StageSpace(task,graph,split,order=o,catalog=catalog,max_width=width) for o in orders]
        binary = [sum(s.transition(i,1)[2] for i in range(d)) for s in plain]
        self.profile_screen = {'binary_us': binary, 'ratio_limit': 1.15}
        narrow_access = max((len(catalog.outgoing[r]) for r in graph.ranks), default=0) <= 2
        self.profiles = []
        for s,c in zip(plain,binary):
            if c > min(binary)*1.15+1e-9:
                continue
            balanced = BalancedStageSpace(task,graph,split,order=s.order,catalog=catalog,max_width=width)
            holders = (HolderStageSpace(task,graph,split,order=s.order,catalog=catalog)
                       if narrow_access and not self.independent_ar else None)
            self.profiles.append(RefinedProfile(s, balanced, holders))
        self.costs = []
        for p in self.profiles:
            if self.independent_ar:
                forward = [[p.phase_cost(i,w,0) for w in range(1,min(3 if i < 2 else width,d-i)+1)]
                           for i in range(d)]
                backward = []
                for i in range(d):
                    costs = []
                    for w in range(1,min(width,d-i)+1):
                        first = d-i-w
                        if w > 3 and first < 2:
                            break
                        costs.append(p.phase_cost(first,w,1))
                    backward.append(costs)
                self.costs.append(forward + backward)
            else:
                self.costs.append([[p.transition(i,w)[2] for w in range(1,min(width,d-i)+1)]
                                   for i in range(d)])

    def evidence(self):
        result = super().evidence()
        for metadata, profile in zip(result['profiles'],self.profiles):
            metadata['holder_transitions'] = [list(k) for k,s in profile.selected.items()
                                               if s is profile.holders]
            metadata['routing_audit'] = profile.striped.routing_audit
        return result
