"""Capacity-aware whole-round routing with whole-atom striping and rebalancing.

The stage compiler fixes contributor dependencies. This layer changes only
legal routes for those same movements. Preview and materialization share exact
atom intervals, so the round objective can be independently checked by Replay.
"""
from .stage_program import LazyPlan, PacketPlan

from .capacity_backend import native
from .striped_stages import StripedStageSpace


class StripePlan(LazyPlan):
    """Store only byte-sized route decisions until a winning stage is emitted."""
    packet_width = PacketPlan.packet_width

    def __init__(self, space, entries, jobs, selected, duration):
        self.costs, self.size, self.stripes = space.costs, space.size, space.stripes
        self.minimum = space.minimum_packet
        self.entries, self.jobs, self.selected = entries, jobs, selected
        self.duration = duration

    def __len__(self):
        return len(self.selected)

    def __iter__(self):
        choices = iter(self.selected)
        for ((action, source, destination), atoms), (options, _) in zip(self.entries, self.jobs):
            lanes = min(self.stripes, atoms, len(options))
            for lane in range(lanes):
                start, end = atoms*lane//lanes, atoms*(lane+1)//lanes
                step = 1 if self.minimum else (end-start+3)//4
                yield action, source, destination, options[next(choices)][3], start, end, step


class BalancedStageSpace(StripedStageSpace):
    def __init__(self, *args, passes=2, minimum_packet=True, **kwargs):
        super().__init__(*args, **kwargs)
        if native is None or not hasattr(native, 'optimize_round'):
            raise ValueError('whole-round balancing requires the native optimizer')
        self.passes = passes
        self.minimum_packet = minimum_packet
        self.routing_audit = []

    def _route_counts(self, movements):
        entries = sorted(movements.items(), key=lambda row: -row[1])
        jobs = self.route_jobs(entries)
        selected, before, duration, moved = native.optimize_groups(
            len(self.costs.indices), jobs, self.passes, self.costs.round_cache,
            self.stripes, self.size, self.minimum_packet)
        self.routing_audit.append({'packets': len(selected), 'before_us': before,
                                   'after_us': duration, 'route_changes': moved})
        return StripePlan(self, entries, jobs, selected, duration), duration
