"""Whole-atom multipath stages; resource capacity is shared, not divided a priori."""
from collections import defaultdict
import math

from .fabric_routes import CapacityAssignment
from .stage_program import StageSpace, PacketPlan


class StripedStageSpace(StageSpace):
    """Use several legal routes for a rank pair without splitting an atom.

    All stripes read the same round-start state. The preview records the exact
    atom intervals later materialized, including indivisible AR expressions.
    """
    def __init__(self, *args, stripes=4, **kwargs):
        super().__init__(*args, **kwargs)
        if type(stripes) is not int or not 1 <= stripes <= 8:
            raise ValueError('stripes must be in 1..8')
        self.stripes = stripes

    def _route_counts(self, movements):
        load, rows = CapacityAssignment(self.costs), []
        packets = sorted(movements.items(), key=lambda row: -row[1])
        for (action, source, destination), atoms in packets:
            pair = source, destination
            if pair not in self.options:
                self.options[pair] = tuple(self.costs.compile(path)
                    for path in self.catalog.options(source, destination))
            count = min(self.stripes, atoms, max(1, len(self.options[pair])))
            for lane in range(count):
                start, end = atoms * lane // count, atoms * (lane + 1) // count
                step = max(1, (end - start + 3) // 4)
                path = load.choose(self.options[pair], (end - start) * self.size, step * self.size)
                rows.append((action, source, destination, path, start, end, step))
        compact = []
        for action, source, destination, path, start, end, step in rows:
            _terms, overhead, serial, _ = self.costs.compile(path)
            limit = (load.time_us - overhead) / (serial * self.size)
            if math.isfinite(limit):
                step = max(step, min(end - start, math.floor(limit * (1 - 1e-12))))
            compact.append((action, source, destination, path, start, end, step))
        return compact, load.time_us

    @staticmethod
    def _materialize(plans, movements):
        rows, consumed = [], defaultdict(int)
        for action, source, destination, path, start, end, step in plans:
            if isinstance(plans, PacketPlan):
                step = plans.packet_width(path, end - start, step)
            key = action, source, destination
            atoms = movements[key]
            if start != consumed[key] or not start < end <= len(atoms):
                raise ValueError('stripe intervals must partition each movement')
            consumed[key] = end
            rows.extend({'action': action, 'src': source, 'dst': destination,
                         'chunk_ranges': atoms[i:min(i + step, end)], 'path': list(path)}
                        for i in range(start, end, step))
        if any(consumed[key] != len(atoms) for key, atoms in movements.items()):
            raise ValueError('stripe lost atoms')
        return rows
