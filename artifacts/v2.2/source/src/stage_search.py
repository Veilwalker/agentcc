"""Exact transition costs and compilation for model-authored global stage search.

The capacity kernels define a finite search space. The runtime model supplies
the program that searches it. A separate lower-bound auditor returns only a
number for feedback; it never supplies an answer on the model's behalf.
"""
from copy import deepcopy
from dataclasses import dataclass
import math

from .collectives import locality_order, _initial
from .fabric_routes import RouteCatalog
from .stage_program import StageSpace, validate_widths
from .striped_stages import StripedStageSpace
from .solver import validate_task
from .schedule_edit import _input_digest


def instance_context(task, graph):
    capacities = sorted(graph.edges.values())
    edges = sorted(graph.edges, key=lambda e: (graph.edges[e], e))[:3]
    pools = sorted(graph.resources, key=lambda r: (graph.resources[r], r))[:2]
    members = {r: sorted(e for e, names in graph.edge_resources.items() if r in names) for r in pools}
    sampled_edges = edges + [e for r in pools for e in members[r][:2]]
    nodes = {n: i for i, n in enumerate(sorted({n for e in sampled_edges for n in e}))}
    steps = (len(graph.ranks)-1).bit_length() * (2 if task['operator']=='all-reduce' else 1)
    return {'op': task['operator'], 'ranks': len(graph.ranks),
        'step_bound_free': task['max_nonempty_rounds'] >= steps,
        'bytes': task['data_size']['per_rank_bytes'], 'splits': validate_task(task, graph),
        'rounds': task['max_nonempty_rounds'], 'edges': len(graph.edges),
        'directed_edge_GBs': [[nodes[a], nodes[b], round(graph.edges[a,b]/1e9, 4)] for a,b in edges],
        'shared_pools': [[round(graph.resources[r]/1e9, 4), len(members[r]),
            [[nodes[a],nodes[b]] for a,b in members[r][:2]]] for r in pools],
        'edge_GBs_range': [round(capacities[0]/1e9,4), round(capacities[-1]/1e9,4)] if capacities else [],
        'latency_us': [graph.startup_us, graph.switch_overhead_us]}


@dataclass
class Profile:
    plain: StageSpace
    striped: StripedStageSpace
    selected: dict

    def transition(self, i, w, phase=None):
        key = i, w, phase
        if key not in self.selected:
            old = self.plain.transition(i, w)
            # A one-atom movement cannot be striped. Its route assignment is
            # identical, so avoid rebuilding the same capacity preview.
            if all(row[4] == 1 for rows in old[:2] for row in rows):
                self.selected[key] = self.plain
                return old
            new = self.striped.transition(i, w)
            before = old[2] if phase is None else self.plain.phase_times[i,w][phase]
            after = new[2] if phase is None else self.striped.phase_times[i,w][phase]
            self.selected[key] = self.striped if after < before - 1e-9 else self.plain
        return self.selected[key].transition(i, w)

    def phase_cost(self, i, w, phase):
        self.transition(i,w,phase)
        return self.selected[i,w,phase].phase_times[i,w][phase]

    def compile(self, widths):
        space = self.plain
        sketches = getattr(self, "gather_arcs", {})
        max_width = max((w for _, w in sketches), default=0)
        validate_widths(widths, space.dimensions, max(space.max_width, max_width))
        answer = _initial(space.task, list(space.graph.ranks), space.split)
        forward, backward, first = [], [], 0
        for w in widths:
            sketch = sketches.get((first, w))
            if sketch is not None:
                forward.extend(sketch.compile())
                first += w
                continue
            self.transition(first, w)
            chosen = self.selected[first, w, None]
            plans, copies, _cost = chosen.transition(first, w)
            f, b = chosen._movements(first, w, atoms=True)
            forward.append(chosen._materialize(plans, f))
            if b:
                backward.append(chosen._materialize(copies, b))
            first += w
        answer['rounds'] = [{'round': i+1, 'transfers': transfers}
            for i, transfers in enumerate(forward + backward[::-1])]
        answer['round_count'] = len(answer['rounds'])
        if answer['round_count'] > space.task['max_nonempty_rounds']:
            raise ValueError('compiled schedule exceeds round bound')
        return answer

    def compile_independent_ar(self, widths):
        space, rounds, position = self.plain, [], 0
        d = space.dimensions
        for w in widths:
            meeting = getattr(self, 'meetings', {}).get((position, w))
            if meeting is not None:
                rounds.append(meeting.compile())
                position += w
                continue
            phase = 0 if position < d else 1
            i = position if phase == 0 else 2*d-position-w
            self.transition(i,w,phase)
            chosen = self.selected[i,w,phase]
            plans = chosen.transition(i,w)[phase]
            movements = chosen._movements(i,w,atoms=True,phase=phase)[phase]
            rounds.append(chosen._materialize(plans,movements))
            position += w
        answer = _initial(space.task,list(space.graph.ranks),space.split)
        answer['rounds'] = [{'round':i+1,'transfers':row} for i,row in enumerate(rounds)]
        answer['round_count'] = len(rounds)
        return answer


class SearchSpace:
    def __init__(self, task, graph, split, max_width=3, independent_ar=True):
        self.task, self.graph, self.input_hash = task, graph, _input_digest(task, graph)
        n = len(graph.ranks)
        if not 2 <= n <= 512 or n & (n-1):
            raise ValueError('stage search requires 2..512 power-of-two ranks')
        self.physical_dimensions = d = (n-1).bit_length()
        self.independent_ar = independent_ar and task['operator']=='all-reduce'
        self.dimensions = d*2 if self.independent_ar else d
        self.limit = min(self.dimensions, task['max_nonempty_rounds'] // (
            2 if task['operator']=='all-reduce' and not self.independent_ar else 1))
        catalog = RouteCatalog(graph, task['data_size']['per_rank_bytes']//split, branch_ports=True)
        orders = [list(graph.ranks)]
        local = locality_order(graph)
        if local != orders[0]:
            orders.append(local)
        plain = [StageSpace(task, graph, split, order=o, catalog=catalog, max_width=max_width) for o in orders]
        # Binary costs are only used to cap extra profile work, never treated as
        # proof that an excluded order cannot improve. Keep the 1.6 best order.
        binary = [sum(s.transition(i,1)[2] for i in range(d)) for s in plain]
        self.profile_screen = {'binary_us': binary, 'ratio_limit': 1.15}
        self.profiles = [Profile(s, StripedStageSpace(task, graph, split, order=s.order, catalog=catalog, max_width=max_width), {})
            for s, cost in zip(plain, binary) if cost <= min(binary)*1.15 + 1e-9]
        if self.independent_ar:
            self.costs = [
                [[p.phase_cost(i,w,0) for w in range(1,min(max_width,d-i)+1)] for i in range(d)] +
                [[p.phase_cost(d-i-w,w,1) for w in range(1,min(max_width,d-i)+1)] for i in range(d)]
                for p in self.profiles]
        else:
            self.costs = [[[p.transition(i,w)[2] for w in range(1,min(max_width,d-i)+1)]
                           for i in range(d)] for p in self.profiles]

    def evaluate(self, output):
        if (not isinstance(output, (list,tuple)) or len(output)!=2
                or type(output[0]) is not int or not 0 <= output[0] < len(self.profiles)):
            raise ValueError('return [profile_index, widths]')
        p, widths = output
        if isinstance(widths, tuple):
            widths = list(widths)
        validate_widths(widths, self.dimensions, max(map(len, self.costs[p])))
        if len(widths)>self.limit:
            raise ValueError('program exceeds stage limit')
        first, cost = 0, 0.0
        for w in widths:
            if w > len(self.costs[p][first]):
                raise ValueError('program crosses phase boundary or unavailable transition')
            cost += self.costs[p][first][w-1]
            first += w
        if not math.isfinite(cost):
            raise ValueError('nonfinite program cost')
        return p, widths, cost

    def audit_bound(self):
        """Exact optimum in this finite stage space; no answer is returned."""
        optimum = math.inf
        for table in self.costs:
            dp = {(0,0): 0.0}
            for i in range(self.dimensions):
                for k in range(self.limit):
                    if (i,k) not in dp:
                        continue
                    for w,c in enumerate(table[i],1):
                        key = i+w,k+1
                        dp[key] = min(dp.get(key,math.inf),dp[i,k]+c)
            optimum = min(optimum, *(dp.get((self.dimensions,k),math.inf) for k in range(1,self.limit+1)))
        return optimum

    def best_legal_path(self):
        """Recover a budget-feasible local path only when the model proposal fails."""
        choices = []
        for profile, table in enumerate(self.costs):
            frontier = {(0, 0): (0.0, ())}
            for node, row in enumerate(table):
                for steps in range(self.limit):
                    state = frontier.get((node, steps))
                    if state is None:
                        continue
                    cost, path = state
                    for width, duration in enumerate(row, 1):
                        key = node + width, steps + 1
                        value = cost + duration, path + (width,)
                        if key not in frontier or value < frontier[key]:
                            frontier[key] = value
            choices.extend((cost, profile, path) for (node, _), (cost, path)
                           in frontier.items() if node == self.dimensions and math.isfinite(cost))
        if not choices:
            return None
        _, profile, path = min(choices)
        return [profile, list(path)]

    def compile(self, output):
        if _input_digest(self.task,self.graph)!=self.input_hash:
            raise ValueError('search input changed')
        p,widths,cost = self.evaluate(output)
        return (self.profiles[p].compile_independent_ar(widths) if self.independent_ar
                else self.profiles[p].compile(widths)), cost

    def evidence(self):
        return {'input_sha256': self.input_hash, 'cost_tables_us': deepcopy(self.costs),
            'cost_table_convention': 'cost_tables_us[p][i][w-1] is transition i -> i+w',
            'program_input_conventions': {
                'outgoing_v1': 'outgoing table with an empty terminal row appended',
                'incoming_v1': 'c[0]=[]; c[i][w-1] is i-w -> i; unavailable transitions are infinity',
            },
            'stage_limit': self.limit, 'profile_screen': self.profile_screen,
            'independent_ar': self.independent_ar,
            'profiles': [{'rank_order': p.plain.order, 'split': p.plain.split,
                'striped_transitions': [list(key) for key,value in p.selected.items() if value is p.striped]}
                for p in self.profiles]}
