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
        self.affinities = {}
        self.positions = {rank: i for i, rank in enumerate(self.order)}
        self.holder_choices = {}
        self.ports = {r: [node for node, _, _ in self.graph.adjacency[r]
                          if node in self.graph.switches] for r in self.order}

    def affinity(self, source, destination):
        key = source, destination
        if key not in self.affinities:
            self.affinities[key] = (
                -self.graph.edges.get(key, 0),
                -sum(min(self.graph.edges.get((source, sw), 0), self.graph.edges.get((sw, destination), 0))
                     for sw in self.ports[source]))
        return self.affinities[key]

    def transition(self, first, width):
        self.first = first
        return super().transition(first, width)

    def _route_counts(self, movements):
        packets, bindings = [], []
        positions = self.positions
        stride = 1 << (self.dimensions - self.first)
        count = 1 << self.first
        for (action, source, destination), atoms in sorted(movements.items(), key=lambda r: -r[1]):
            i = positions[source]
            # Always retain the original sender; other holders are chosen by
            # physical adjacency first, then a deterministic spread of holders.
            key = self.first, i % stride, destination
            if key not in self.holder_choices:
                holders = [self.order[(i % stride) + j * stride] for j in range(count)]
                holders.sort(key=lambda rank: (self.affinity(rank, destination), positions[rank]))
                self.holder_choices[key] = holders[:self.holder_limit]
            holders = self.holder_choices[key]
            holders = [source] + [h for h in holders if h != source][:self.holder_limit - 1]
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
