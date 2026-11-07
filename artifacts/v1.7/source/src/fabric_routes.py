"""Directed multipath catalog and whole-round capacity assignment.

Each source access switch contributes a route, rather than varying a scalar
shortest-path weight that can keep choosing the same shared bandwidth pool.
Switch trees are reused across ranks. Paths never relay through another rank.
"""
from collections import defaultdict
import heapq
import math
import time

from .route_balance import _PathCosts, _RoundLoad
from .replay import replay


class RouteCatalog:
    def __init__(self, graph, size_bytes, max_ports=16, branch_ports=False):
        if type(max_ports) is not int or not 1 <= max_ports <= 16:
            raise ValueError('route catalog permits 1..16 access ports')
        self.graph, self.size = graph, size_bytes
        self.max_ports = max_ports
        self.branch_ports = branch_ports
        self.trees, self.paths = {}, {}
        self.fabric_paths = {}
        self.capacity_costs = CapacityCosts(graph)
        self.compiled_options = {}
        self.relaxations = 0
        self.outgoing, self.incoming = defaultdict(list), defaultdict(list)
        self.adjacency = {s: [] for s in graph.switches}
        for (src, dst), capacity in graph.edges.items():
            cost = size_bytes * 1e6 * (1 / capacity + sum(
                1 / graph.resources[r] for r in graph.edge_resources[src, dst]))
            if not math.isfinite(cost):
                raise ValueError('nonfinite fabric route cost')
            if src in graph.switches and dst in graph.switches:
                self.adjacency[src].append((dst, cost + graph.switch_overhead_us))
            elif src not in graph.switches and dst in graph.switches:
                self.outgoing[src].append((dst, cost))
            elif src in graph.switches and dst not in graph.switches:
                self.incoming[dst].append((src, cost))
        for rows in self.adjacency.values():
            rows.sort()
        for rows in self.outgoing.values():
            rows.sort(key=lambda row: (row[1], row[0]))
        for rows in self.incoming.values():
            rows.sort()

    def _tree(self, start):
        if start not in self.trees:
            distances, previous = {start: 0.0}, {}
            queue = [(0.0, start)]
            while queue:
                distance, source = heapq.heappop(queue)
                if distance != distances[source]:
                    continue
                for destination, cost in self.adjacency[source]:
                    self.relaxations += 1
                    if self.relaxations > 2_000_000:
                        raise ValueError('fabric route relaxation budget exhausted')
                    new = distance + cost
                    if new < distances.get(destination, math.inf):
                        distances[destination], previous[destination] = new, source
                        heapq.heappush(queue, (new, destination))
            self.trees[start] = distances, previous
        return self.trees[start]

    def options(self, source, destination, original=()):
        pair = source, destination
        if pair not in self.paths:
            if len(self.paths) >= 32_768:
                raise ValueError('fabric route pair budget exhausted')
            # Ranks with identical access switches share the switch-only
            # portion. Endpoint edges remain explicit in every final path.
            starts = tuple(start for start, _ in self.outgoing[source][:self.max_ports])
            exits = tuple(self.incoming[destination])
            branch = self.branch_ports and len(self.outgoing[source]) <= 2
            key = starts, exits, branch
            if key not in self.fabric_paths:
                self.fabric_paths[key] = self._fabric_options(starts, exits, branch)
            choices = {(source, *core, destination) for core in self.fabric_paths[key]}
            if pair in self.graph.edges:
                choices.add(pair)
            self.paths[pair] = tuple(sorted(choices))
        if not original:
            return self.paths[pair]
        return tuple(sorted(set(self.paths[pair]) | {tuple(original)}))

    def _fabric_options(self, starts, exits, branch):
        choices = set()
        for start in starts:
            distances, previous = self._tree(start)
            reachable = [(distances[end] + cost, end) for end, cost in exits
                         if end in distances]
            if not reachable:
                continue
            _, end = min(reachable)
            path = [end]
            while end != start:
                end = previous[end]
                path.append(end)
            choices.add(tuple(reversed(path)))
            if branch:
                # Explore a bounded set of actual second switch hops when
                # a rank has too few access switches to expose fabric paths.
                for middle, _cost in self.adjacency[start][:8]:
                    distances2, previous2 = self._tree(middle)
                    reachable2 = [(distances2[end] + cost, end) for end, cost in exits
                                  if end in distances2]
                    if not reachable2:
                        continue
                    _, end2 = min(reachable2)
                    tail = [end2]
                    while end2 != middle:
                        end2 = previous2[end2]
                        tail.append(end2)
                    if start not in tail:
                        choices.add((start, *reversed(tail)))
        return tuple(sorted(choices))


class CapacityCosts:
    """Compile shared-resource incidence once for repeated stage evaluations."""
    def __init__(self, graph):
        self.graph = graph
        self.indices = {('edge', edge): i for i, edge in enumerate(graph.edges)}
        self.indices.update({('pool', key): len(graph.edges) + i for i, key in enumerate(graph.resources)})
        self.paths = {}

    def compile(self, path):
        path = tuple(path)
        if path not in self.paths:
            graph, pools, terms, serial = self.graph, defaultdict(int), [], 0.0
            for edge in zip(path, path[1:]):
                inverse = 1e6 / graph.edges[edge]
                serial += inverse
                terms.append((self.indices['edge', edge], inverse))
                for key in graph.edge_resources[edge]:
                    pools[key] += 1
            terms.extend((self.indices['pool', key], multiple * 1e6 / graph.resources[key])
                         for key, multiple in pools.items())
            self.paths[path] = (tuple(terms), graph.startup_us + (len(path) - 2) * graph.switch_overhead_us,
                                serial, path)
        return self.paths[path]


class CapacityAssignment:
    def __init__(self, costs):
        self.loads = [0.0] * len(costs.indices)
        self.busy = self.overhead = self.serial = 0.0

    @property
    def time_us(self):
        return max(self.serial, self.overhead + self.busy)

    def choose(self, options, count, packet_bytes=None):
        """Exact greedy insertion score; sorted options make ties reproducible."""
        loads, peak, base_overhead, base_serial = self.loads, self.busy, self.overhead, self.serial
        packet_bytes = count if packet_bytes is None else packet_bytes
        best, selected = None, None
        for option in options:
            terms, overhead, serial, _ = option
            busy, energy = 0.0, 0.0
            for index, factor in terms:
                before = loads[index]
                after = before + count * factor
                if after > busy:
                    busy = after
                energy += after * after - before * before
            path_time = overhead + packet_bytes * serial
            score = (max(base_serial, path_time, max(base_overhead, overhead) + max(peak, busy)),
                     busy, energy, path_time)
            if best is None or score < best:
                best, selected = score, option
        if selected is None:
            raise ValueError('no legal route for this transfer')
        terms, overhead, serial, path = selected
        for index, factor in terms:
            loads[index] += count * factor
            if loads[index] > self.busy:
                self.busy = loads[index]
        self.overhead = max(self.overhead, overhead)
        self.serial = max(self.serial, overhead + packet_bytes * serial)
        return path


def assign(task, graph, solution, *, catalog=None, validated_report=None):
    """Assign complete packets across legal fabric routes and replay any gain."""
    started = time.monotonic()
    size = task['data_size']['per_rank_bytes'] // solution['split']
    catalog = catalog or RouteCatalog(graph, size)
    costs = _PathCosts(graph)
    result, before, after, changes = [], 0.0, 0.0, 0
    for row in solution['rounds']:
        old, load = _RoundLoad(costs), _RoundLoad(costs)
        transfers = row['transfers']
        for item in transfers:
            old.add(item['path'], len(item['chunk_ranges']) * size)
        changed = list(transfers)
        for index in sorted(range(len(transfers)), key=lambda i: -len(transfers[i]['chunk_ranges'])):
            item = transfers[index]
            count = len(item['chunk_ranges']) * size
            path = min(catalog.options(item['src'], item['dst'], item['path']),
                       key=lambda p: load.score(p, count))
            load.add(path, count)
            if tuple(item['path']) != path:
                changed[index] = {**item, 'path': list(path)}
        before += old.time_us
        if load.time_us < old.time_us - 1e-6:
            result.append({**row, 'transfers': changed})
            after += load.time_us
            changes += sum(a is not b for a, b in zip(transfers, changed))
        else:
            result.append(row)
            after += old.time_us
    candidate = {**solution, 'rounds': result} if changes else solution
    checked = replay(task, graph, candidate) if changes or validated_report is None else validated_report
    accepted = bool(changes and checked['valid'] and checked['total_time_us'] < before - 1e-6)
    if not accepted:
        candidate = solution
        checked = validated_report or replay(task, graph, solution)
    else:
        candidate['metrics'] = {'total_time': checked['total_time_us'] / 1e6,
            'round_times': [value / 1e6 for value in checked['round_times_us']]}
    return candidate, {'accepted': accepted, 'before_us': before,
        'after_us': checked['total_time_us'], 'changed_transfers': changes if accepted else 0,
        'switch_trees': len(catalog.trees), 'pairs': len(catalog.paths),
        'elapsed_s': time.monotonic() - started, 'final_replay': checked}
