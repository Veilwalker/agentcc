"""Assign remaining single-switch flows within round and port budgets."""
from collections import defaultdict, deque
from math import floor


class _Network:
    """Integer flow network for atom-to-switch assignments."""

    def __init__(self):
        self.adjacency = []

    def node(self):
        self.adjacency.append([])
        return len(self.adjacency) - 1

    def add(self, source, destination, capacity):
        edge = [destination, len(self.adjacency[destination]), capacity]
        self.adjacency[source].append(edge)
        self.adjacency[destination].append([source, len(self.adjacency[source]) - 1, 0])
        return edge

    def maximize(self, source, destination, target):
        flow = 0
        while flow < target:
            depth = [-1] * len(self.adjacency)
            depth[source] = 0
            queue = deque([source])
            while queue:
                node = queue.popleft()
                for peer, _, capacity in self.adjacency[node]:
                    if capacity and depth[peer] < 0:
                        depth[peer] = depth[node] + 1
                        queue.append(peer)
            if depth[destination] < 0:
                break
            cursor = [0] * len(self.adjacency)

            def push(node, limit):
                if node == destination:
                    return limit
                edges = self.adjacency[node]
                while cursor[node] < len(edges):
                    edge = edges[cursor[node]]
                    peer, back, capacity = edge
                    if capacity and depth[peer] == depth[node] + 1:
                        sent = push(peer, min(limit, capacity))
                        if sent:
                            edge[2] -= sent
                            self.adjacency[peer][back][2] += sent
                            return sent
                    cursor[node] += 1
                return 0

            while flow < target:
                sent = push(source, target - flow)
                if not sent:
                    break
                flow += sent
        return flow


def _assign_switches(planner, pending, plane_paths, spent, busy, chosen):
    graph, costs, atom = planner.graph, planner.costs, planner.size
    net = _Network()
    source, sink = net.node(), net.node()
    gates, ports = {}, {}
    for switch, routes in plane_paths.items():
        # Collect resources touched by every eligible path through this plane.
        shared = dict(routes[0])
        for terms in routes[1:]:
            incidence = dict(terms)
            for resource in list(shared):
                if resource in incidence:
                    shared[resource] = min(shared[resource], incidence[resource])
                else:
                    del shared[resource]
        capacities = (floor((busy - spent[i]) / (atom * rate) + 1e-8)
                      for i, rate in shared.items())
        gates[switch] = net.node()
        net.add(gates[switch], sink, max(0, min(capacities, default=len(pending))))

    decisions = []
    for sender, receiver, copy, slot, options in pending:
        job = net.node()
        net.add(source, job, 1)
        choices = []
        for switch, path in options:
            # The fan-in edge binds at the receiver during reduction and the
            # fan-out edge binds at the sender during broadcast.
            edge = (sender, switch) if copy else (switch, receiver)
            if edge not in ports:
                resource = costs.indices['edge', edge]
                count = floor((busy - spent[resource]) * graph.edges[edge] / (atom * 1e6) + 1e-8)
                ports[edge] = net.node()
                net.add(ports[edge], gates[switch], max(0, count))
            choices.append((net.add(job, ports[edge], 1), path))
        decisions.append((choices, copy, slot))

    if net.maximize(source, sink, len(pending)) < len(pending):
        return False
    for choices, copy, slot in decisions:
        path = next(path for edge, path in choices if edge[2] == 0)
        chosen[path, copy].append(slot)
    return True


def _test_row(planner, stage, incumbent, ratio):
    graph, costs, atom = planner.graph, planner.costs, planner.size
    target = incumbent * ratio
    busy = target - graph.startup_us - graph.switch_overhead_us
    if busy <= 0:
        return None
    chosen, plane_paths = defaultdict(list), defaultdict(list)
    pending, spent = [], defaultdict(float)
    for (sender, receiver, copy, _), slots in stage.items():
        pair = sender, receiver
        count = 0
        if pair in graph.edges:
            terms, latency, serial, path = costs.compile(pair)
            if latency + serial * atom <= target + 1e-8:
                count = min(len(slots), *(floor((busy - spent[i]) / (rate * atom) + 1e-8)
                                          for i, rate in terms))
                count = max(0, count)
            if count:
                chosen[path, copy].extend(slots[:count])
                for i, rate in terms:
                    spent[i] += count * atom * rate
        if count == len(slots):
            continue
        options = []
        for switch in sorted(planner.common[sender] & planner.common[receiver]):
            terms, latency, serial, path = costs.compile((sender, switch, receiver))
            if (latency + atom * serial <= target + 1e-9
                    and all(rate * atom + spent[i] <= busy + 1e-8 for i, rate in terms)):
                options.append((switch, path))
                plane_paths[switch].append(terms)
        if not options:
            return None
        pending.extend((sender, receiver, copy, slot, options) for slot in slots[count:])
        if len(pending) > 256:
            return None
    if pending and not _assign_switches(planner, pending, plane_paths, spent, busy, chosen):
        return None

    # The assignment treats selected port and plane capacities exactly. Reprice
    # all other directed edges, pools and serial path costs before selecting it.
    load = defaultdict(float)
    overhead = serialized = 0.
    for (path, _), slots in chosen.items():
        terms, latency, serial, _ = costs.compile(path)
        overhead = max(overhead, latency)
        serialized = max(serialized, latency + atom * serial)
        for i, rate in terms:
            load[i] += atom * rate * len(slots)
    duration = max(serialized, overhead + max(load.values()))
    return (duration, chosen) if duration < incumbent - 1e-6 else None


def polish(planner, stages, recipe):
    # Only the two weak-axis and two inter-domain stages use the pooled fabric.
    improved = list(recipe)
    for i in range(1, len(recipe) - 1):
        for ratio in (.89, .94, .975, .995):
            trial = _test_row(planner, stages[i], recipe[i][0], ratio)
            if trial and trial[0] < improved[i][0]:
                improved[i] = trial
    return improved
