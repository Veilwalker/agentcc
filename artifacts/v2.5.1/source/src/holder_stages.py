"""AllGather stages may forward a block from any of its current holders.

After a stage, each logical group still owns the same union of original atoms.
Changing the forwarding holder therefore preserves the state used by later
stages, while allowing the capacity optimizer to use a different physical link.
"""
import math

from .balanced_stages import BalancedStageSpace
from .capacity_backend import native


class HolderStageSpace(BalancedStageSpace):
    def __init__(self, *args, holder_limit=4, **kwargs):
        super().__init__(*args, **kwargs)
        if self.task['operator'] != 'all-gather':
            raise ValueError('holder substitution applies to AllGather')
        self.holder_limit = holder_limit
        self.positions = {rank: i for i, rank in enumerate(self.order)}
        self.holder_choices = {}
        self.preferences = {}
        self.links_to_rank = [[] for _ in self.order]
        self.ports_to_rank = [[] for _ in self.order]
        self.senders_by_port = {}
        for (source, dest), bandwidth in self.graph.edges.items():
            src = self.positions.get(source)
            dst = self.positions.get(dest)
            if dst is not None:
                if src is None:
                    self.ports_to_rank[dst].append((source, bandwidth))
                else:
                    self.links_to_rank[dst].append((src, bandwidth))
            elif src is not None:
                self.senders_by_port.setdefault(dest, []).append((src, bandwidth))
        # Preserve the adjacency's lexical switch order when summing rates.
        for rows in self.ports_to_rank:
            rows.sort()

    def _preference(self, destination):
        if destination not in self.preferences:
            dst = self.positions[destination]
            direct = [0.0] * len(self.order)
            shared = [0.0] * len(self.order)
            for src, bandwidth in self.links_to_rank[dst]:
                direct[src] = bandwidth
            for port, recv in self.ports_to_rank[dst]:
                for src, send in self.senders_by_port.get(port, ()):
                    shared[src] += min(send, recv)
            ranked = sorted(range(len(self.order)), key=lambda i: (-direct[i], -shared[i], i))
            self.preferences[destination] = direct, shared, ranked
        return self.preferences[destination]

    def affinity(self, source, destination):
        direct, shared, _ = self._preference(destination)
        index = self.positions[source]
        return -direct[index], -shared[index]

    def _holders(self, source, destination, stride, count):
        pos = self.positions[source] % stride
        if self.holder_limit in (0, 1) or count == 1:
            return [source]
        if count == 2 and self.holder_limit > 0:
            first, second = self.order[pos], self.order[pos + stride]
            return [source, second if first == source else first]
        key = stride, destination
        if key not in self.holder_choices:
            # Restrict the destination's total ordering to each ownership group.
            groups = [[] for _ in range(stride)]
            limit = self.holder_limit if self.holder_limit > 0 else max(0, count + self.holder_limit)
            for i in self._preference(destination)[2]:
                bucket = groups[i % stride]
                if len(bucket) < limit:
                    bucket.append(self.order[i])
            self.holder_choices[key] = groups
        pool = self.holder_choices[key][pos]
        return [source] + [rank for rank in pool if rank != source][:self.holder_limit - 1]

    def transition(self, first, width):
        self.first = first
        return super().transition(first, width)

    def _route_counts(self, movements):
        packets, bindings = [], []
        stride = 1 << (self.dimensions - self.first)
        count = 1 << self.first
        for (action, source, destination), atoms in sorted(movements.items(), key=lambda r: -r[1]):
            holders = self._holders(source, destination, stride, count)
            options = []
            for holder in holders:
                pair = holder, destination
                if pair not in self.options:
                    self.options[pair] = tuple(self.costs.compile(path)
                        for path in self.catalog.options(holder, destination))
                options.extend(self.options[pair])
            options = tuple(options[:128])
            lanes = min(self.stripes, atoms, len(options))
            for lane in range(lanes):
                start, end = atoms * lane // lanes, atoms * (lane + 1) // lanes
                packets.append((options, (end-start)*self.size, self.size))
                bindings.append((action, source, destination, start, end))
        selected, before, duration, moved = native.optimize_round(len(self.costs.indices), packets, self.passes)
        self.routing_audit.append({'packets': len(packets), 'before_us': before,
                                   'after_us': duration, 'route_changes': moved})
        result = []
        for binding, packet, choice in zip(bindings, packets, selected):
            _, overhead, serial, path = packet[0][choice]
            limit = (duration-overhead)/(serial*self.size)
            action, source, destination, start, end = binding
            step = max(1, min(end-start, math.floor(limit*(1-1e-12)))) if math.isfinite(limit) else 1
            result.append((action, source, destination, path, start, end, step))
        return result, duration

    @staticmethod
    def _materialize(plans, movements):
        rows = BalancedStageSpace._materialize(plans, movements)
        for row in rows:
            row['src'] = row['path'][0]
        return rows
