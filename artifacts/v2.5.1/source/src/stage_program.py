"""Compile mixed-radix collective programs from current topology capacities.

A stage consumes 1..3 consecutive binary rank dimensions. It sends each
disjoint input block directly to every required peer within that stage, so no
within-round rank forwarding is assumed. AR reduction and broadcast mirror
the same partition, preserving contributor ownership exactly.
"""
from collections import defaultdict
import math

from .collectives import _initial
from .capacity_backend import native
from .fabric_routes import RouteCatalog, CapacityAssignment


def validate_widths(widths, dimensions, max_width=3):
    if (not isinstance(widths, list) or not widths
            or any(type(w) is not int or not 1 <= w <= max_width for w in widths)
            or sum(widths) != dimensions):
        raise ValueError('stage widths must be integers 1..3 summing to dimension count')


class PacketPlan(list):
    """Delay packet coalescing until this transition is actually emitted."""
    def __init__(self, costs, size):
        super().__init__()
        self.costs, self.size = costs, size
        self.duration = 0.

    def packet_width(self, path, atoms, step):
        _, overhead, serial, _ = self.costs.paths[path]
        limit = (self.duration - overhead) / (serial * self.size)
        return max(step, min(atoms, math.floor(limit * (1 - 1e-12)))) if math.isfinite(limit) else step


class LazyPlan:
    """Preserve the sequence operations supported by a materialized route plan."""
    def __add__(self, other):
        return [*self, *other]

    def __radd__(self, other):
        return [*other, *self]

    def __getitem__(self, key):
        return list(self)[key]


class PlainPlan(LazyPlan):
    """Keep only chosen indices while dynamic programming explores stage widths."""
    packet_width = PacketPlan.packet_width

    def __init__(self, space, entries, jobs, selected, duration):
        self.costs, self.size = space.costs, space.size
        self.entries, self.jobs = entries, jobs
        self.selected, self.duration = selected, duration
        self.multiple_atoms = bool(entries and entries[0][1] > 1)

    def __len__(self):
        return len(self.selected)

    def __iter__(self):
        for ((action, src, dst), count), (options, _), choice in zip(self.entries, self.jobs, self.selected):
            yield action, src, dst, options[choice][3], count, (count+3)//4


class StageSpace:
    """Cache independently measurable stage transitions, not instance answers."""
    def __init__(self, task, graph, split, order=None, catalog=None, max_width=3, root_positions=None):
        self.task, self.graph, self.split = task, graph, split
        self.order = list(graph.ranks) if order is None else list(order)
        n = len(self.order)
        if n != len(graph.ranks) or n < 2 or n & (n - 1) or set(self.order) != set(graph.ranks):
            raise ValueError('stage space needs a power-of-two rank permutation')
        self.dimensions = (n - 1).bit_length()
        if type(max_width) is not int or not 1 <= max_width <= 8:
            raise ValueError('maximum dimension width must be in 1..8')
        self.max_width = max_width
        if root_positions is not None and (len(root_positions) != split or
                any(type(r) is not int or not 0 <= r < n for r in root_positions)):
            raise ValueError("root positions must provide one rank index per chunk")
        self.root_positions = tuple(root_positions) if root_positions is not None else None
        if split not in task.get('split_policy', {}).get('candidates', [task.get('split', 1)]):
            raise ValueError('split is not allowed')
        total = task['data_size']['per_rank_bytes']
        if total % split or (total // split) % task.get('split_policy', {}).get('alignment_bytes', 1):
            raise ValueError('split violates alignment')
        self.size = total // split
        self.catalog = catalog or RouteCatalog(graph, self.size)
        # Input/locality orders use the same graph and route capacities. Share
        # compiled incidence within this request, never across instances.
        self.costs = self.catalog.capacity_costs
        self.options = self.catalog.compiled_options
        self.stages = {}
        self.movement_counts = {}
        self.phase_times = {}
        self.materialized = {}

    def _route(self, movements):
        """Research helper for explicit movements; production previews counts."""
        plans, cost = self._route_counts({key: len(atoms) for key, atoms in movements.items()})
        return self._materialize(plans, movements), cost

    def _route_counts(self, movements):
        """Capacity preview without constructing per-atom expressions or JSON."""
        packets = sorted(movements.items(), key=lambda row: -row[1])
        if native is not None and hasattr(native, 'optimize_groups'):
            jobs = self.route_jobs(packets)
            try:
                selected, _, duration, _ = native.optimize_groups(len(self.costs.indices), jobs, 0,
                    self.costs.round_cache, 1, self.size, False)
            except ValueError:
                pass
            else:
                return PlainPlan(self, packets, jobs, selected, duration), duration
        load, rows = CapacityAssignment(self.costs), PacketPlan(self.costs, self.size)
        for (action, source, destination), atoms in packets:
            step = max(1, (atoms + 3) // 4)
            pair = source, destination
            if pair not in self.options:
                self.options[pair] = tuple(self.costs.compile(path)
                    for path in self.catalog.options(source, destination))
            count = atoms * self.size
            path = load.choose(self.options[pair], count, step * self.size)
            rows.append((action, source, destination, path, atoms, step))
        # Coalesce whole atoms only while the real packet-path bound remains
        # within the already measured round duration. Resource bytes do not
        # change. This cuts response size without buying speed by losing cost.
        rows.duration = load.time_us
        return rows, rows.duration

    def route_jobs(self, entries):
        if (entries and isinstance(entries, list) and hasattr(self.costs, "native_graph")
                and native is not None and hasattr(native, "stage_route_jobs")):
            return native.stage_route_jobs(self.costs.native_graph(), entries, self.options,
                                           self.costs.paths, self.catalog.options)
        jobs = []
        for (_, source, destination), atoms in entries:
            pair = source, destination
            if pair not in self.options:
                self.options[pair] = tuple(self.costs.compile(path)
                    for path in self.catalog.options(source, destination))
            jobs.append((self.options[pair], atoms))
        return jobs

    @staticmethod
    def _materialize(plans, movements):
        rows = []
        for action, source, destination, path, count, step in plans:
            if hasattr(plans, "packet_width"):
                step = plans.packet_width(path, count, step)
            atoms = movements[action, source, destination]
            if len(atoms) != count:
                raise ValueError('materialized atom count differs from capacity preview')
            rows.extend({'action': action, 'src': source, 'dst': destination,
                         'chunk_ranges': atoms[i:i + step], 'path': list(path)}
                        for i in range(0, count, step))
        return rows

    def _movements(self, first, width, *, atoms=False, phase=None):
        if phase not in (None, 0, 1):
            raise ValueError('phase must be reduction, broadcast or both')
        task, order, n = self.task, self.order, len(self.order)
        if task['operator'] == 'all-gather':
            movements = {}
            shift = self.dimensions - first - width
            for i, source in enumerate(order):
                value = (1 << first) * self.split
                if atoms:
                    owners = [order[i ^ (j << (self.dimensions - first))] for j in range(1 << first)]
                    value = [f'{owner}_c{slot}' for owner in owners for slot in range(self.split)]
                for j in range(1, 1 << width):
                    movements['copy', source, order[i ^ (j << shift)]] = value
            return movements, {}
        reduce, broadcast = defaultdict(list if atoms else int), defaultdict(list if atoms else int)
        stride = n // self.split + 1
        stride += stride % 2 == 0
        distance, factor = 1 << first, 1 << width
        for slot in range(self.split):
            root = self.root_positions[slot] if self.root_positions is not None else slot * stride % n
            perm = [order[i ^ root] for i in range(n)] if atoms and phase != 1 else None
            for start in range(0, n, distance * factor):
                destination = order[start ^ root]
                for j in range(1, factor):
                    branch = start + j * distance
                    source = order[branch ^ root]
                    if atoms:
                        if phase != 1:
                            contributors = '+'.join(f'{rank}_c{slot}' for rank in perm[branch:branch + distance])
                            reduce['reduce', source, destination].append(contributors)
                        if phase != 0:
                            broadcast['copy', destination, source].append(f'c{slot}_sum')
                    else:
                        if phase != 1:
                            reduce['reduce', source, destination] += 1
                        if phase != 0:
                            broadcast['copy', destination, source] += 1
        return reduce, broadcast

    def transition(self, first, width):
        if type(first) is not int or type(width) is not int or first < 0 or width < 1 or width > self.max_width or first + width > self.dimensions:
            raise ValueError('invalid stage transition')
        key = first, width
        if key not in self.stages:
            if key not in self.movement_counts:
                self.movement_counts[key] = self._movements(first, width)
            forward, backward = self.movement_counts[key]
            rows, cost_r = self._route_counts(forward)
            copies, cost_b = self._route_counts(backward) if backward else ([], 0.0)
            self.stages[key] = rows, copies, cost_r + cost_b
            self.phase_times[key] = (cost_r, cost_b)
        return self.stages[key]

    def table(self):
        return [[round(self.transition(first, width)[2], 3)
                 for width in range(1, min(self.max_width, self.dimensions - first) + 1)]
                for first in range(self.dimensions)]

    def compile(self, widths):
        validate_widths(widths, self.dimensions, self.max_width)
        solution = _initial(self.task, list(self.graph.ranks), self.split)
        first, forwards, backwards, cost = 0, [], [], 0.0
        for width in widths:
            plans, reverse_plans, duration = self.transition(first, width)
            key = first, width
            if key not in self.materialized:
                forward, backward = self._movements(first, width, atoms=True)
                self.materialized[key] = self._materialize(plans, forward), self._materialize(reverse_plans, backward)
            rows, copies = self.materialized[key]
            forwards.append(rows)
            if copies:
                backwards.append(copies)
            cost += duration
            first += width
        batches = forwards + backwards[::-1]
        solution['rounds'] = [{'round': i + 1, 'transfers': rows} for i, rows in enumerate(batches) if rows]
        solution['round_count'] = len(solution['rounds'])
        if solution['round_count'] > self.task['max_nonempty_rounds']:
            raise ValueError('stage program exceeds task round limit')
        return solution, cost

    def oracle(self):
        """Offline additive-cost optimum; not a model-authored decision."""
        best = {self.dimensions: (0.0, [])}
        for first in reversed(range(self.dimensions)):
            best[first] = min((self.transition(first, width)[2] + best[first + width][0],
                               [width] + best[first + width][1])
                              for width in range(1, min(self.max_width, self.dimensions - first) + 1))
        return best[0]
