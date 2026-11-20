"""Capacity search with interchangeable, state-equivalent stage kernels.

Cost comparisons use the same round-start ownership state. A holder change is
allowed only for AllGather, where every member of the processed dimension group
already owns the identical atom set. No topology name participates in a choice.
"""
from .balanced_stages import BalancedStageSpace
from .holder_stages import HolderStageSpace
from .meeting_stage import MeetingStage
from .group_sketch import add_gather_sketches
from .collectives import locality_order
from .cohort_pipeline import pipeline
from .fabric_routes import RouteCatalog
from .schedule_edit import _input_digest
from .stage_program import StageSpace
from .stage_search import SearchSpace, Profile


class RefinedProfile(Profile):
    def __init__(self, plain, balanced, holders=None):
        super().__init__(plain, balanced, {})
        self.holders = holders
        self.meetings = {}
        self.gather_arcs = {}
        self.narrow_access = max(len(plain.catalog.outgoing[r]) for r in plain.order) <= 2
        balanced.movement_counts = plain.movement_counts
        if holders is not None:
            holders.movement_counts = plain.movement_counts

    def transition(self, i, w, phase=None):
        key = i, w, phase
        if key not in self.selected:
            variants = [self.plain]
            old = self.plain.transition(i, w)
            if (any(row[4] > 1 for rows in old[:2] for row in rows)
                    or (self.narrow_access and self.plain.task['operator'] == 'all-reduce' and i >= 2
                        and any(len(self.plain.options[row[1], row[2]]) > 1
                                for row in old[0]))):
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
                # A bridge goes directly from several disjoint partials to
                # multiple complete roots. Width still counts consumed logical
                # dimensions; every arc compiles into one real communication round.
                for remaining in range(1, min(3, d) + 1):
                    first = d - remaining
                    if len(forward[first]) != remaining:
                        continue
                    for copies in range(1, min(remaining, width - remaining) + 1):
                        try:
                            stage = MeetingStage(p.striped, first, copies)
                        except ValueError:
                            break  # ragged rows contain only available widths
                        span = remaining + copies
                        p.meetings[first, span] = stage
                        forward[first].append(stage.cost)
                self.costs.append(forward + backward)
            else:
                self.costs.append([[p.transition(i,w)[2] for w in range(1,min(width,d-i)+1)]
                                   for i in range(d)])
                add_gather_sketches(p, self.costs[-1], self.limit)

    def compile(self, output):
        candidate, predicted = super().compile(output)
        if not self.independent_ar:
            plain = self.profiles[output[0]].plain
            candidate, total = pipeline(candidate, predicted, plain, self.task['max_nonempty_rounds'])
            self.timing_sketch = {'before_us': predicted, 'after_us': total,
                                  'applied': total < predicted, 'rounds': candidate['round_count']}
            return candidate, total
        return candidate, predicted

    def evidence(self):
        result = super().evidence()
        if hasattr(self, 'timing_sketch'):
            result['cohort_pipeline'] = self.timing_sketch
        for metadata, profile in zip(result['profiles'],self.profiles):
            metadata['holder_transitions'] = [list(k) for k,s in profile.selected.items()
                                               if s is profile.holders]
            metadata['gather_sketches'] = [[*key, type(value).__name__] for key,value in profile.gather_arcs.items()]
            metadata['multi_root_arcs'] = [list(key) for key in profile.meetings]
            metadata['routing_audit'] = profile.striped.routing_audit
        return result
