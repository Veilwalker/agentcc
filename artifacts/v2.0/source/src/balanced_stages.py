"""Capacity-aware whole-round routing with whole-atom striping and rebalancing.

The stage compiler fixes contributor dependencies. This layer changes only
legal routes for those same movements. Preview and materialization share exact
atom intervals, so the round objective can be independently checked by Replay.
"""
import math

from .capacity_backend import native
from .striped_stages import StripedStageSpace


class BalancedStageSpace(StripedStageSpace):
    def __init__(self, *args, passes=2, minimum_packet=True, **kwargs):
        super().__init__(*args, **kwargs)
        if native is None or not hasattr(native, 'optimize_round'):
            raise ValueError('whole-round balancing requires the native optimizer')
        self.passes = passes
        self.minimum_packet = minimum_packet
        self.routing_audit = []

    def _route_counts(self, movements):
        packets, bindings = [], []
        for (action, source, destination), atoms in sorted(movements.items(), key=lambda r: -r[1]):
            pair = source, destination
            if pair not in self.options:
                self.options[pair] = tuple(self.costs.compile(path)
                    for path in self.catalog.options(source, destination))
            options = self.options[pair]
            lanes = min(self.stripes, atoms, max(1, len(options)))
            for lane in range(lanes):
                start, end = atoms * lane // lanes, atoms * (lane + 1) // lanes
                step = 1 if self.minimum_packet else max(1, (end - start + 3) // 4)
                packets.append((options, (end - start) * self.size, step * self.size))
                bindings.append((action, source, destination, start, end, step))
        selected, before, duration, moved = native.optimize_round(len(self.costs.indices), packets, self.passes)
        self.routing_audit.append({'packets': len(packets), 'before_us': before,
                                   'after_us': duration, 'route_changes': moved})
        result = []
        for binding, packet, choice in zip(bindings, packets, selected):
            action, source, destination, start, end, step = binding
            _, overhead, serial, path = packet[0][choice]
            limit = (duration - overhead) / (serial * self.size)
            if math.isfinite(limit):
                step = max(step, min(end - start, math.floor(limit * (1 - 1e-12))))
            result.append((action, source, destination, path, start, end, step))
        return result, duration
