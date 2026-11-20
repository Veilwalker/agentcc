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

    def reduce(self, task, split, spacing, placement):
        groups, ranks = self.groups, self.graph.ranks
        layouts = []
        for slot in range(split):
            domains = []
            for i, group in enumerate(groups):
                pivot = placement((slot, i, len(group))) % len(group)
                ordered = group[pivot:] + group[:pivot]
                root = ordered[0]
                near = [r for r in ordered[1:] if self.paths(r, root, True) and self.paths(root, r, True)]
                domains.append((root, [root, *near[::spacing]]))
            layouts.append(domains)
        # Each slot tracks the exact contributor expression at its current rank.
        # Apply all arrivals only after pricing and reading the complete round.
        state = [{r: f'{r}_c{s}' for r in ranks} for s in range(split)]
        recipe, times = [], []
        for phase in range(6):
            broadcast = phase >= 3
            depth = 5-phase if broadcast else phase
            requests = []
            choices = {}
            for slot, domains in enumerate(layouts):
                global_root = domains[slot % len(groups)][0]
                for group, (root, relays) in zip(groups, domains):
                    if depth == 0:
                        pairs = ((r, relays) for r in group if r not in relays)
                    elif depth == 1:
                        pairs = ((r, (root,)) for r in relays if r != root)
                    else:
                        pairs = [(root, (global_root,))] if root != global_root else []
                    for leaf, parents in pairs:
                        key = leaf, tuple(parents)
                        if key not in choices:
                            options = tuple(p for parent in parents for p in (
                                self.paths(parent, leaf, depth < 2) if broadcast else
                                self.paths(leaf, parent, depth < 2)))
                            if not options or len(options) > 128:
                                return None
                            choices[key] = options
                        requests.append((choices[key], (slot,)))
            if not requests:
                continue
            ceiling, batches = self.assign(requests)
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

    def gather(self, task, spacing, placement):
        groups, ranks = self.groups, self.graph.ranks
        index = self.catalog.rank_index
        home = {r: i for i, group in enumerate(groups) for r in group}
        rows = [defaultdict(list) for _ in range(3)]
        for i, owner in enumerate(ranks):
            token = f'{owner}_c0'
            for j, group in enumerate(groups):
                root = owner if home[owner] == j else group[placement((i, j, len(group))) % len(group)]
                if root != owner:
                    rows[0][owner, root].append(token)
                near = [r for r in group if r in self.neighbors[root]]
                start = i % len(near) if near else 0
                relays = [root, *(near[start:] + near[:start])[::spacing]]
                for relay in relays[1:]:
                    rows[1][root, relay].append(token)
                for dst in group:
                    if dst in relays:
                        continue
                    ready = [r for r in relays if dst in self.neighbors[r]]
                    if not ready:
                        return None
                    rows[2][ready[(i+index[dst]) % len(ready)], dst].append(token)
        return self.route_gather(task, rows)

    def route_gather(self, task, rows):
        recipe, times = [], []
        for depth, moves in enumerate(rows):
            requests = []
            for (src, dst), payload in moves.items():
                paths = self.paths(src, dst, depth > 0)
                if not paths or len(paths) > 128:
                    return None
                lanes = min(4, len(paths), len(payload))
                for lane in range(lanes):
                    requests.append((paths, payload[len(payload)*lane//lanes:len(payload)*(lane+1)//lanes]))
            if requests:
                ceiling, batches = self.assign(requests)
                recipe.append(self.emit(batches, ceiling, 'copy'))
                times.append(ceiling)
        return self.solution(task, 1, recipe), sum(times)

    def solution(self, task, split, rows):
        result = _initial(task, self.graph.ranks, split)
        result['rounds'] = [dict(round=i+1, transfers=row) for i, row in enumerate(rows)]
        result['round_count'] = len(rows)
        return result


def evaluate(task, forest, allowed, spacing, placement):
    graph, catalog = forest.graph, forest.catalog
    gather = task['operator'] == 'all-gather'
    if gather and allowed[0] != 1:
        return None
    split = 1 if gather else allowed[-1]
    try:
        placement = relay_expression(placement)
        result = forest.gather(task, spacing, placement) if gather else forest.reduce(task, split, spacing, placement)
        if result is None:
            return None
        candidate, predicted = result
        if gather:
            plan = shard_pipeline.price(task, graph, candidate, predicted, catalog, allowed)
            if plan is not None:
                candidate = shard_pipeline.materialize(task, graph, candidate, catalog, plan)
                predicted = sum(plan[2])
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
    candidates, equivalent = {}, {}
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
                    equivalent[key] = evaluate(task, forest, allowed, *layout)
                candidates[layout] = equivalent[key]
            except (ValueError, ZeroDivisionError):
                candidates[layout] = None
    attempt(default)
    if flow is not None and pending is not None and not pending.done():
        eligible = [pair for pair in (candidates.get(default), competitor) if pair is not None]
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
    valid = [(result[1]['total_time_us'], layout != hint, layout, result)
             for layout, result in candidates.items() if result is not None and layout in (default, hint)]
    if not valid:
        return None
    _, _, selected, (answer, report) = min(valid)
    model_selected = selected == hint
    evidence = [dict(layout=list(layout), valid=res is not None, selected=layout == selected,
                     model_proposed=layout == hint, speculative_only=layout not in (default, hint), total_time_us=res[1]['total_time_us'] if res else None)
                for layout, res in candidates.items()]
    answer['metrics'] = dict(total_time=report['total_time_us']/1e6,
                             round_times=[v/1e6 for v in report['round_times_us']])
    decision = dict(semantics='access_relay_layout_v1', selected=model_selected,
                    accepted=model_selected, final_time_us=report['total_time_us'],
                    executions=([dict(output=list(hint), valid_program=candidates[hint] is not None,
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
