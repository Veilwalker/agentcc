"""Build independent up/down two-hop trees from the two physical grid axes."""
from collections import defaultdict


def improve(grid, reference):
    if len(reference) != 6 or any(duration <= 0 for duration, _ in reference):
        return None
    graph, blocks = grid.graph, grid.blocks
    size = grid.size
    # Regular strong-axis links already support balanced direct fan-in.  The
    # second legal corner of each cell is useful when those physical arcs
    # disappear, are throttled or compete for a declared capacity pool.
    rates = []
    impaired = False
    for block in blocks:
        for line in block:
            for a in line:
                for b in line:
                    if a == b:
                        continue
                    rate = graph.edges.get((a, b))
                    if rate is None or graph.edge_resources[a, b]:
                        impaired = True
                        break
                    rates.append(rate)
                if impaired:
                    break
            if impaired:
                break
        if impaired:
            break
    if not impaired:
        rates.sort()
        if not rates or rates[0] >= .60 * rates[len(rates) // 2]:
            return None

    waves = [defaultdict(list) for _ in range(6)]
    incidences = {}

    def edge(a, b):
        pair = a, b
        if pair not in incidences:
            terms = []
            if pair in graph.edges:
                terms.append((pair, 1e6 * size / graph.edges[pair]))
                terms.extend((p, 1e6 * size / graph.resources[p]) for p in graph.edge_resources[pair])
            incidences[pair] = terms
        return incidences[pair]

    for scatter in (False, True):
        loads = [defaultdict(float), defaultdict(float)]
        first, second = (5, 4) if scatter else (0, 1)
        outer = .55 * reference[first][0]
        inner = reference[second][0]

        def score(terms, level):
            if not terms:
                return float('inf')
            return max(loads[level][k] + use for k, use in terms)

        def commit(terms, level):
            for k, v in terms:
                loads[level][k] += v

        def path(a, b):
            return edge(b, a) if scatter else edge(a, b)

        def put(stage, a, b, s):
            if scatter:
                a, b = b, a
            waves[stage][a, b, scatter, False].append(s)

        for slot, stripe in enumerate(grid.roots):
            root = stripe[(slot + slot // grid.width) % grid.count][1]
            for block, (column, hub) in zip(blocks, stripe):
                r0 = column.index(hub)
                c0 = block[r0].index(hub)
                active = set()
                for i, line in enumerate(block):
                    if i == r0:
                        continue
                    for j, source in enumerate(line):
                        if j == c0:
                            continue
                        # A source at (i,j) can turn at (i,c0) or (r0,j).
                        # In both cases the second hop is directly aligned
                        # with the hub. Charge this hop once per live subtree.
                        chosen = None
                        for relay in (line[c0], block[r0][j]):
                            up = path(source, relay)
                            mid = path(relay, hub)
                            a = score(up, 0) / outer
                            if relay in active:
                                b = max((loads[1][k] / inner for k, _ in mid), default=float('inf'))
                            else:
                                b = score(mid, 1) / inner
                            metric = (max(1., a, b), a + b * (.25 if relay in active else .47))
                            if chosen is None or metric < chosen[0]:
                                chosen = metric, relay, up, mid
                        _, relay, up, mid = chosen
                        commit(up, 0)
                        if relay not in active:
                            active.add(relay)
                            commit(mid, 1)
                        put(first, source, relay, slot)
                for relay in dict.fromkeys((*column, *block[r0])):
                    if relay == hub:
                        continue
                    term = path(relay, hub)
                    if relay not in active and score(term, 0) / outer < score(term, 1) / inner:
                        commit(term, 0)
                        put(first, relay, hub, slot)
                    else:
                        if relay not in active:
                            commit(term, 1)
                        put(second, relay, hub, slot)
                if hub != root:
                    put(3 if scatter else 2, hub, root, slot)
    return grid.price(waves)
