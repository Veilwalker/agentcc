"""Build two-level multicast and aggregation trees from overlapping access ports."""
from collections import defaultdict
from math import floor

from .capacity_backend import native
from .collectives import _initial
from .code_policy import relay_expression
from .group_sketch import access_groups
from .replay import certify
from . import shard_pipeline


def heterogeneous_groups(task, graph, allowed):
    if (native is None or not 64 <= len(graph.ranks) <= 256
            or task['max_nonempty_rounds'] < 8 or max(allowed) > 32):
        return None
    ports = defaultdict(set)
    for src, dst in graph.edges:
        if dst in graph.switches and src not in graph.switches:
            ports[src].add(dst)
    if not all(1 <= len(ports[r]) <= 4 for r in graph.ranks):
        return None
    groups = access_groups(graph)
    if (not 3 <= len(groups) <= 16 or len({len(g) for g in groups}) < 2
            or max(map(len, groups)) > 64
            or any(len({frozenset(ports[r]) for r in g}) < 2 for g in groups)):
        return None
    return groups


class DominatedLayout(Exception):
    """Even the certificate tolerance cannot bring this layout below its incumbent."""

    def __init__(self, lower_bound_us):
        self.lower_bound_us = lower_bound_us


def stop_if_dominated(predicted, incumbent):
    # A successful certificate must agree with the predicted sum to this
    # tolerance. In a prefix, each remaining round adds non-negative time.
    lower = predicted - max(1e-4, predicted * 1e-9)
    if lower > incumbent:
        raise DominatedLayout(lower)


class AccessForest:
    def __init__(self, graph, catalog, groups, size):
        self.graph = graph
        self.catalog = catalog
        self.groups = groups
        self.size = size
        self.costs = catalog.capacity_costs
        self.ports = {r: frozenset(p for p, _ in catalog.outgoing[r]) for r in graph.ranks}
        self.local = {}
        self.remote = {}
        self.neighbors = {r: {other for other in group if other != r and
                               any((port, other) in graph.edges for port in self.ports[r])}
                          for group in groups for r in group}

        self.near_order = {root: tuple(r for r in group if r in self.neighbors[root])
                           for group in groups for root in group}
        self.gather_layers = {}
        self.reduce_choices = {}

    def paths(self, src, dst, local):
        cache = self.local if local else self.remote
        key = src, dst
        if key not in cache:
            paths = ([(src, port, dst) for port in sorted(self.ports[src])
                      if (port, dst) in self.graph.edges] if local else self.catalog.options(src, dst))
            cache[key] = tuple(self.costs.compile(p) for p in paths)
        return cache[key]

    def assign(self, requests):
        """An alternative can change both the relay endpoint and the route."""
        specs = [(paths, self.size*len(payload), self.size) for paths, payload in requests]
        selection, _, ceiling, _ = native.optimize_round(
            len(self.costs.indices), specs, 2, self.costs.round_cache)
        batches = defaultdict(list)
        for (paths, payload), index in zip(requests, selection):
            batches[paths[index][3]].extend(payload)
        return ceiling, batches

    def emit(self, batches, ceiling, action):
        result = []
        for path, tokens in batches.items():
            _, overhead, inverse, _ = self.costs.paths[path]
            width = max(1, floor((ceiling-overhead)/(self.size*inverse) + 1e-10))
            for start in range(0, len(tokens), width):
                result.append(dict(src=path[0], dst=path[-1], path=list(path),
                                   action=action, chunk_ranges=tokens[start:start+width]))
        return result

    def reduce(self, task, split, spacing, placement, incumbent=float('inf')):
        groups, ranks = self.groups, self.graph.ranks
        layouts = []
        ordered_local = {}
        for slot in range(split):
            domains = []
            for i, group in enumerate(groups):
                pivot = placement((slot, i, len(group))) % len(group)
                ordered = group[pivot:] + group[:pivot]
                root = ordered[0]
                if root not in ordered_local:
                    near = [r for r in ordered[1:] if r in self.neighbors[root] and root in self.neighbors[r]]
                    relays = (root, *near[::spacing])
                    ordered_local[root] = (root, relays, frozenset(relays))
                domains.append(ordered_local[root])
            layouts.append(domains)
        # Each slot tracks the exact contributor expression at its current rank.
        # Apply all arrivals only after pricing and reading the complete round.
        state = [{r: f'{r}_c{s}' for r in ranks} for s in range(split)]
        recipe, times = [], []
        for phase in range(6):
            broadcast = phase >= 3
            depth = 5-phase if broadcast else phase
            requests = []
            local = depth < 2
            choices = self.reduce_choices.setdefault((broadcast, local), {})
            for slot, domains in enumerate(layouts):
                global_root = domains[slot % len(groups)][0]
                for group, (root, relays, relay_set) in zip(groups, domains):
                    if depth == 0:
                        pairs = ((r, relays) for r in group if r not in relay_set)
                    elif depth == 1:
                        pairs = ((r, (root,)) for r in relays if r != root)
                    else:
                        pairs = [(root, (global_root,))] if root != global_root else []
                    for leaf, parents in pairs:
                        key = leaf, parents
                        if key not in choices:
                            if local:
                                reachable = (p for p in parents if leaf in self.neighbors[p]) if broadcast else (
                                    p for p in parents if p in self.neighbors[leaf])
                            else:
                                reachable = parents
                            options = tuple(p for parent in reachable for p in (
                                self.paths(parent, leaf, local) if broadcast else self.paths(leaf, parent, local)))
                            if not options or len(options) > 128:
                                return None
                            choices[key] = options
                        requests.append((choices[key], (slot,)))
            if not requests:
                continue
            ceiling, batches = self.assign(requests)
            stop_if_dominated(sum(times) + ceiling, incumbent)
            arrivals = defaultdict(list)
            tokens = {}
            for path, slots in batches.items():
                if broadcast:
                    tokens[path] = [f'c{s}_sum' for s in slots]
                else:
                    tokens[path] = [state[s][path[0]] for s in slots]
                    for s, token in zip(slots, tokens[path]):
                        arrivals[s, path[-1]].append(token)
            for (s, rank), parts in arrivals.items():
                state[s][rank] = '+'.join([state[s][rank], *parts])
            recipe.append(self.emit(tokens, ceiling, 'copy' if broadcast else 'reduce'))
            times.append(ceiling)
        return self.solution(task, split, recipe), sum(times)

    def gather(self, task, spacing, placement, *, grouped=False, incumbent=float('inf')):
        groups, ranks = self.groups, self.graph.ranks
        index = self.catalog.rank_index
        home = {r: i for i, group in enumerate(groups) for r in group}
        roots, exchange = [], defaultdict(list)
        for i, owner in enumerate(ranks):
            token = f'{owner}_c0'
            row = []
            for j, group in enumerate(groups):
                root = owner if home[owner] == j else group[placement((i, j, len(group))) % len(group)]
                row.append(root)
                if root != owner:
                    exchange[owner, root].append(token)
            roots.append(row)

        first = self.route_layer(exchange, local=False)
        if first is None:
            return None
        _, batches = first
        if incumbent < float('inf') and not any(
                len(p) > 2 and (p[0], p[-1]) in self.graph.edges for p in batches):
            # Shifting or splitting a tree keeps these physical arc bytes.
            # The busiest link/pool alone lower-bounds the full replay.
            loads = defaultdict(float)
            for path, tokens in batches.items():
                amount = self.size * len(tokens)
                for resource, coefficient in self.costs.paths[path][0]:
                    loads[resource] += amount * coefficient
            stop_if_dominated(max(loads.values(), default=0.), incumbent)

        middle, last = defaultdict(list), defaultdict(list)
        for i, (owner, owner_roots) in enumerate(zip(ranks, roots)):
            token = f'{owner}_c0'
            for group, root in zip(groups, owner_roots):
                near = self.near_order[root]
                start = i % len(near) if near else 0
                key = (root, start, spacing)
                layer = self.gather_layers.get(key)
                if layer is None:
                    relays = (root, *(near[start:] + near[:start])[::spacing])
                    relay_set = frozenset(relays)
                    leaves = []
                    for dst in group:
                        if dst in relay_set:
                            continue
                        ready = tuple(r for r in relays if dst in self.neighbors[r])
                        if not ready:
                            return None
                        leaves.append((dst, ready, index[dst]))
                    layer = self.gather_layers[key] = (relays[1:], leaves)
                relays, leaves = layer
                for relay in relays:
                    middle[root, relay].append(token)
                for dst, ready, dst_index in leaves:
                    last[ready[(i+dst_index) % len(ready)], dst].append(token)

        stages = [first]
        for layer in (middle, last):
            if layer:
                routed = self.route_layer(layer, local=True)
                if routed is None:
                    return None
                stages.append(routed)
        costs = [ceiling for ceiling, _ in stages]
        if grouped:
            return [row for _, row in stages], costs
        return self.solution(task, 1, [self.emit(row, cost, 'copy') for cost, row in stages]), sum(costs)

    def route_layer(self, moves, *, local):
        requests = []
        for (src, dst), payload in moves.items():
            paths = self.paths(src, dst, local)
            if not paths or len(paths) > 128:
                return None
            lanes = min(4, len(paths), len(payload))
            for lane in range(lanes):
                requests.append((paths, payload[len(payload)*lane//lanes:len(payload)*(lane+1)//lanes]))
        return self.assign(requests) if requests else (0., {})

    def solution(self, task, split, rows):
        result = _initial(task, self.graph.ranks, split)
        result['rounds'] = [dict(round=i+1, transfers=row) for i, row in enumerate(rows)]
        result['round_count'] = len(rows)
        return result


def evaluate(task, forest, allowed, spacing, placement, incumbent=float('inf')):
    graph, catalog = forest.graph, forest.catalog
    gather = task['operator'] == 'all-gather'
    if gather and allowed[0] != 1:
        return None
    split = 1 if gather else allowed[-1]
    try:
        placement = relay_expression(placement)
        if gather:
            stages = forest.gather(task, spacing, placement, grouped=True, incumbent=incumbent)
            if stages is None:
                return None
            rows, costs = stages
            predicted = sum(costs)
            candidate = forest.solution(task, 1, [])
            plan = shard_pipeline.price(task, graph, candidate, predicted, catalog, allowed, grouped=rows)
            if plan is None:
                stop_if_dominated(predicted, incumbent)
                candidate = forest.solution(task, 1, [
                    forest.emit(row, ceiling, 'copy') for row, ceiling in zip(rows,costs)])
            else:
                predicted = sum(plan[2])
                stop_if_dominated(predicted, incumbent)
                candidate = shard_pipeline.materialize(task, graph, candidate, catalog, plan)
        else:
            result = forest.reduce(task, split, spacing, placement, incumbent)
            if result is None:
                return None
            candidate, predicted = result
        checked = certify(task, graph, candidate)
        if checked['valid'] and abs(checked['total_time_us']-predicted) < max(1e-4,predicted*1e-9):
            return candidate, checked
    except (ValueError, ZeroDivisionError):
        # Optional catalog / route-incidence limits also apply to this family.
        return None
    return None


def solve_domains(task, graph, groups, allowed, pending, started, *, flow=None, competitor=None):
    """Use the original service on unsupported layouts or failed certificates."""
    import time
    from . import __version__
    from .capacity_backend import BACKEND
    from .fabric_routes import RouteCatalog
    from .llm import LLMError

    split = allowed[0] if task['operator'] == 'all-gather' else allowed[-1]
    if task['operator'] == 'all-gather' and split != 1:
        return None
    catalog = RouteCatalog(graph, task['data_size']['per_rank_bytes']//split,
                           branch_ports=True, deep_exits=task['operator'] == 'all-reduce')
    forest = AccessForest(graph, catalog, groups, task['data_size']['per_rank_bytes']//split)
    # Cover both ports of a typical rank before adding extra representatives.
    degree = max(2, min(9, max(map(len, forest.ports.values()))))
    domain_ports = max(len(set().union(*(forest.ports[r] for r in g))) for g in groups)
    alternate = max(2, min(9, domain_ports-1))
    default = (degree, 'i + g')
    speculative = (alternate, f'i + g * {alternate}')
    candidates, equivalent, dominated, bounds = {}, {}, {}, {}
    def comparable_cost(pair):
        if flow is not None and pair[0] is flow.parent and flow.result is not None:
            return flow.result[1]['total_time_us']
        return pair[1]['total_time_us']

    def attempt(layout):
        if layout not in candidates:
            spacing, expression = layout
            try:
                index = relay_expression(expression)
                count = len(graph.ranks) if task['operator']=='all-gather' else split
                roots = tuple(index((i,g,len(group))) % len(group)
                              for i in range(count) for g,group in enumerate(groups))
                key = spacing, roots
                if key not in equivalent:
                    incumbent = min((comparable_cost(pair) for pair in candidates.values() if pair is not None),
                                    default=float('inf'))
                    try:
                        equivalent[key] = evaluate(task, forest, allowed, *layout, incumbent=incumbent)
                    except DominatedLayout as cutoff:
                        equivalent[key] = None
                        bounds[key] = cutoff.lower_bound_us
                candidates[layout] = equivalent[key]
                if key in bounds:
                    dominated[layout] = bounds[key]
            except (ValueError, ZeroDivisionError):
                candidates[layout] = None
    attempt(default)
    cover_key = ('port_cover', 'balanced_local_roots')
    if task['operator'] == 'all-reduce' and not competitor and candidates.get(default):
        from .bridge_forest import construct
        try:
            cover = construct(task, forest, split, candidates[default][1]['total_time_us'])
        except (ValueError, TypeError, IndexError, KeyError, OverflowError):
            cover = None
        if cover:
            candidates[cover_key] = cover
    if flow is not None and pending is not None and not pending.done():
        eligible = [pair for pair in (candidates.get(default), candidates.get(cover_key), competitor)
                    if pair is not None]
        if eligible:
            chosen, checked = min(eligible, key=lambda pair: pair[1]['total_time_us'])
            flow.finish(chosen, checked)
    if pending is not None and not pending.done():
        attempt(speculative)
    interactions, hint = [], None
    if pending is not None:
        try:
            _, metadata = pending.result()
            interactions.append(metadata)
            if metadata.get('relay_layout'):
                hint = tuple(metadata['relay_layout'])
        except LLMError as exc:
            interactions.append(dict(called=True, error=str(exc), usage_unavailable=True,
                                     duration_ms=getattr(exc, 'duration_ms', 0)))
    if hint:
        attempt(hint)

    valid = [(comparable_cost(result), layout != hint, layout, result)
             for layout, result in candidates.items() if result is not None]
    if not valid:
        return None
    _, _, selected, (answer, report) = min(valid, key=lambda row: (row[0], row[1]))
    model_selected = selected == hint
    evidence = [dict(layout=list(layout), valid=res is not None, selected=layout == selected,
                     model_proposed=layout == hint, speculative_only=layout not in (default, hint), total_time_us=res[1]['total_time_us'] if res else None)
                for layout, res in candidates.items()]
    for row in evidence:
        layout = tuple(row['layout'])
        if layout in dominated:
            row.update(valid=None, lower_bound_us=dominated[layout], outcome='pruned_by_certified_incumbent')
    answer['metrics'] = dict(total_time=report['total_time_us']/1e6,
                             round_times=[v/1e6 for v in report['round_times_us']])
    decision = dict(semantics='access_relay_layout_v1', selected=model_selected,
                    accepted=model_selected, final_time_us=report['total_time_us'],
                    executions=([dict(output=list(hint), valid_program=(None if hint in dominated else candidates[hint] is not None),
                                      selected=model_selected)] if hint else []),
                    stop_reason='verified_layout' if model_selected else 'verified_local_layout')
    return dict(answer=answer, diagnostics=dict(version=__version__, capacity_backend=BACKEND,
                elapsed_s=time.monotonic()-started, local_replay=report,
                phases_s={'domain_search_and_model_s': time.monotonic()-started},
                search=dict(model_edit=decision, candidates=evidence, access_forest=dict(
                    group_sizes=list(map(len, groups)), layout=list(selected), selected=True,
                    model_proposed=model_selected,
                    local_us=(candidates[default][1]['total_time_us'] if candidates[default] else None))),
                llm=dict(called=bool(interactions), mode='access_relay_layout', calls=len(interactions),
                         interactions=interactions, duration_ms=sum(v.get('duration_ms',0) for v in interactions),
                         usage={k:sum(v.get('usage',{}).get(k,0) for v in interactions)
                                for k in ('prompt_tokens','completion_tokens','total_tokens')})))
