"""Meet the reduction trees at several live roots in the same round.

At a reduction/copy phase boundary, each surviving sender holds a disjoint
component of the total. Each chosen root receives all components except its
own. All roots can therefore start the next, unmodified copy stage together.
"""
from collections import defaultdict


class MeetingStage:
    def __init__(self, space, first, copies):
        self.space, self.first, self.copies = space, first, copies
        movements = self.movements(atoms=False)
        self.routes, self.cost = space._route_counts(movements)

    def movements(self, *, atoms):
        stage = self.space
        order, n = stage.order, len(stage.order)
        block = 1 << self.first
        root_gap = n >> self.copies
        stride = n // stage.split + 1
        stride += stride % 2 == 0
        result = defaultdict(list if atoms else int)
        for slot in range(stage.split):
            root = stage.root_positions[slot] if stage.root_positions else slot * stride % n
            destinations = [order[root ^ k] for k in range(0, n, root_gap)]
            for start in range(0, n, block):
                source = order[root ^ start]
                if atoms:
                    token = '+'.join(f'{order[root ^ i]}_c{slot}' for i in range(start, start+block))
                for destination in destinations:
                    if source == destination:
                        continue
                    key = 'reduce', source, destination
                    if atoms:
                        result[key].append(token)
                    else:
                        result[key] += 1
        return result

    def compile(self):
        return self.space._materialize(self.routes, self.movements(atoms=True))
