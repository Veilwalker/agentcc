"""Topology-derived multi-round AllGather sketches with identical boundary ownership.

A sketch consumes contiguous rank dimensions of the existing stage graph. Only
whole atoms move, and every holder reads its state at the start of each round.
"""
class AxisExchange:
    """Send two disjoint cohorts around a verified two-axis local fabric.

    After the same prefix, each lane holds one bundle from each outer group.
    Fraction q of the bundle goes on the slow axis first, the rest on the fast
    axis first. In the second round both cohorts complete the other dimension.
    Both rounds can therefore use the two sets of links simultaneously.
    """

    def __init__(self, space, inner, outer, first_count):
        self.space, self.inner, self.outer = space, inner, outer
        self.first_count = first_count
        self.prefix = len(space.order) // (inner * outer)
        self.plans = []
        self.cost = 0.0
        for phase in range(2):
            counts = self.movements(phase, atoms=False)
            plans, cost = space._route_counts(counts)
            self.plans.append(plans)
            self.cost += cost

    def movements(self, phase, *, atoms):
        space, inner, outer = self.space, self.inner, self.outer
        tile = inner * outer
        data = {}
        total = self.prefix * space.split
        for i, sender in enumerate(space.order):
            for outer_first, start, stop in ((True, 0, self.first_count),
                                              (False, self.first_count, total)):
                if start == stop:
                    continue
                along_outer = outer_first != bool(phase)
                size, stride = (outer, inner) if along_outer else (inner, 1)
                previous = (inner if along_outer else outer) if phase else 1
                previous_stride = 1 if along_outer else inner
                payload = (stop - start) * previous
                if atoms:
                    payload = []
                    for lane in range(previous):
                        pos = (i % tile) ^ (lane * previous_stride)
                        for index in range(start, stop):
                            group, slot = divmod(index, space.split)
                            payload.append(f'{space.order[group*tile+pos]}_c{slot}')
                for offset in range(1, size):
                    target = space.order[i ^ (stride * offset)]
                    data['copy', sender, target] = payload
        return data

    def compile(self):
        return [self.space._materialize(plans, self.movements(i, atoms=True))
                for i, plans in enumerate(self.plans)]


def direct_axes(space):
    """Find consecutive XOR dimensions with complete direct connectivity.

    Verify every member and every directed link. Axes that also connect across
    their product (a clique) do not constitute two distinct local dimensions.
    """
    order, edges = space.order, space.graph.edges
    n = len(order)
    def connected(mask):
        return all((rank, order[i ^ mask]) in edges for i, rank in enumerate(order))
    inner = 1
    while inner < min(n, 16) and all(connected(mask) for mask in range(inner, inner*2)):
        inner *= 2
    if inner < 2 or inner >= n or connected(inner+1):
        return None
    outer = 1
    while outer < 16 and inner*outer < n and all(
            connected(inner * mask) for mask in range(outer, 2*outer)):
        outer *= 2
    if outer < 2 or n // (outer*inner) * space.split < 2:
        return None
    return inner, outer


class PlainSequence:
    def __init__(self, profile, first, widths):
        self.profile, self.steps = profile, []
        self.cost = 0.0
        for width in widths:
            self.cost += profile.transition(first, width)[2]
            self.steps.append((first, width))
            first += width

    def compile(self):
        result = []
        for first, width in self.steps:
            space = self.profile.selected[first, width, None]
            plans = space.transition(first, width)[0]
            movements = space._movements(first, width, atoms=True)[0]
            result.append(space._materialize(plans, movements))
        return result


def add_gather_sketches(profile, table, round_limit):
    space = profile.plain
    dimensions = space.dimensions
    if round_limit < dimensions:
        return
    def offer(first, width, sketch):
        row = table[first]
        # Bridge any gaps by compiling real, already-priced ordinary stages.
        # Each sequence consumes at most one physical round per dimension.
        while len(row) < width:
            span = len(row) + 1
            sequence = PlainSequence(profile, first, (3, span-3)) if span <= 6 else None
            if sequence is None:
                return
            row.append(sequence.cost)
            profile.gather_arcs[first, span] = sequence
        if sketch.cost < row[width-1]:
            row[width-1] = sketch.cost
            profile.gather_arcs[first, width] = sketch

    axes = direct_axes(space)
    if axes:
        inner, outer = axes
        width = (inner * outer - 1).bit_length()
        total = len(space.order) // (inner * outer) * space.split
        if 3 < width <= 6:
            # Fractions of whole input chunks, including the opposite ordering.
            for q in sorted({total//2, total-1, 1}):
                if 0 < q < total:
                    offer(dimensions-width, width, AxisExchange(space, inner, outer, q))
