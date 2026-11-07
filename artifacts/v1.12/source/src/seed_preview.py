"""Price the protected binary constructor before allocating symbolic payloads.

The movement counts, fixed routes and two-packet policy exactly match the
legacy initial constructor. A preview is never a validated solution: compile
and independent Replay are required before it can be returned.
"""
from .fabric_routes import CapacityAssignment
from .stage_program import StageSpace


class SeedPreview(StageSpace):
    def _route_counts(self, movements):
        load, rows = CapacityAssignment(self.costs), []
        for (action, source, destination), atoms in movements.items():
            path = tuple(self.graph.path(source, destination, self.size, 4.0))
            step = (atoms + 1) // 2
            option = self.costs.compile(path)
            load.choose((option,), atoms * self.size, step * self.size)
            rows.append((action, source, destination, path, atoms, step))
        return rows, load.time_us

    def binary_cost(self):
        return sum(self.transition(i, 1)[2] for i in range(self.dimensions))
