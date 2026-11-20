"""Bounded refinement of routes using exact per-round edge/resource loads.

The data schedule is immutable: only paths change. Each changed round must have
strictly lower modeled cost, and the complete candidate passes local replay.
"""

from collections import Counter, defaultdict
import math
import time

from .replay import certify as replay


class _PathCosts:
    def __init__(self, graph):
        self.graph = graph
        self.cache = {}

    def describe(self, path):
        path = tuple(path)
        if path not in self.cache:
            edges = tuple(zip(path, path[1:]))
            resources = Counter(resource for edge in edges for resource in self.graph.edge_resources[edge])
            overhead = self.graph.startup_us + (len(path) - 2) * self.graph.switch_overhead_us
            per_byte = 1e6 * sum(1 / self.graph.edges[edge] for edge in edges)
            self.cache[path] = edges, resources, overhead, per_byte
        return self.cache[path]


class _RoundLoad:
    def __init__(self, costs):
        self.costs = costs
        self.edge_bytes = defaultdict(int)
        self.resource_bytes = defaultdict(int)
        self.path_max = self.overhead_max = self.busy_max = 0.0

    @property
    def time_us(self):
        return max(self.path_max, self.overhead_max + self.busy_max)

    def score(self, path, count):
        graph = self.costs.graph
        edges, resources, overhead, per_byte = self.costs.describe(path)
        local_busy = energy = 0.0
        for edge in edges:
            before = 1e6 * self.edge_bytes[edge] / graph.edges[edge]
            after = before + 1e6 * count / graph.edges[edge]
            local_busy = max(local_busy, after)
            energy += after * after - before * before
        for resource, multiplicity in resources.items():
            before = 1e6 * self.resource_bytes[resource] / graph.resources[resource]
            after = before + 1e6 * count * multiplicity / graph.resources[resource]
            local_busy = max(local_busy, after)
            energy += after * after - before * before
        path_time = overhead + count * per_byte
        duration = max(self.path_max, path_time,
                       max(self.overhead_max, overhead) + max(self.busy_max, local_busy))
        return duration, local_busy, energy, path_time, tuple(path)

    def add(self, path, count):
        graph = self.costs.graph
        edges, resources, overhead, per_byte = self.costs.describe(path)
        self.path_max = max(self.path_max, overhead + count * per_byte)
        self.overhead_max = max(self.overhead_max, overhead)
        for edge in edges:
            self.edge_bytes[edge] += count
            self.busy_max = max(self.busy_max, 1e6 * self.edge_bytes[edge] / graph.edges[edge])
        for resource, multiplicity in resources.items():
            self.resource_bytes[resource] += count * multiplicity
            self.busy_max = max(self.busy_max, 1e6 * self.resource_bytes[resource] / graph.resources[resource])

    def bottlenecks(self):
        graph = self.costs.graph
        threshold = self.busy_max * 0.98
        edges = {edge for edge, count in self.edge_bytes.items()
                 if 1e6 * count / graph.edges[edge] >= threshold}
        resources = {resource for resource, count in self.resource_bytes.items()
                     if 1e6 * count / graph.resources[resource] >= threshold}
        return edges, resources


def refine(task: dict, graph, solution: dict, *, deadline: float = math.inf,
           time_budget_s: float = 0.2, weights=(0.0, 1.0, 3.0, 4.0),
           routing_bytes: float | None = None, max_route_queries: int | None = None,
           max_path_scores: int = 20000) -> tuple[dict, dict]:
    """Refine a valid explicit-path solution, returning (solution, evidence).

    Representative sources probe alternate paths first. Refinement runs
    only if a probe can avoid a current bottleneck or shorten a dominating path.
    The routing budget is best-effort at individual path-search boundaries;
    final replay is always completed before a changed solution is returned.
    """
    if type(time_budget_s) not in (int, float) or not math.isfinite(time_budget_s) or time_budget_s < 0:
        raise ValueError("routing time budget must be finite and nonnegative")
    if max_route_queries is not None and (type(max_route_queries) is not int or max_route_queries < 0):
        raise ValueError("max_route_queries must be a nonnegative integer")
    if type(max_path_scores) is not int or max_path_scores < 0:
        raise ValueError("max_path_scores must be a nonnegative integer")
    deterministic = max_route_queries is not None
    started = time.monotonic()
    end = min(deadline, started + time_budget_s)
    # Reserve part of the nominal budget for independent final validation.
    routing_end = min(end, started + time_budget_s * 0.75)
    if deterministic and time_budget_s > 0:
        routing_end = min(deadline, started + 3.0)
    route_queries = path_scores = 0
    if not weights or any(type(weight) not in (float, int) or not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("routing weights must be finite and nonnegative")
    chunk_bytes = task["data_size"]["per_rank_bytes"] // solution["split"]
    representative_bytes = chunk_bytes if routing_bytes is None else routing_bytes
    if (isinstance(representative_bytes, bool) or not isinstance(representative_bytes, (int, float))
            or not math.isfinite(representative_bytes) or representative_bytes <= 0):
        raise ValueError("routing_bytes must be finite and positive")
    costs = _PathCosts(graph)
    contexts, targets = defaultdict(list), defaultdict(set)
    options, original_loads = {}, []
    for number, round_ in enumerate(solution["rounds"]):
        load = _RoundLoad(costs)
        for transfer in round_["transfers"]:
            pair = transfer["src"], transfer["dst"]
            path = tuple(transfer["path"])
            count = len(transfer["chunk_ranges"]) * chunk_bytes
            load.add(path, count)
            contexts[pair].append((number, path, count))
            targets[pair[0]].add(pair[1])
            options.setdefault(pair, set()).add(path)
        original_loads.append(load)
    before = sum(load.time_us for load in original_loads)
    peaks = [load.bottlenecks() for load in original_loads]
    queried = set()

    def alternate_paths(pair):
        nonlocal route_queries
        for weight in weights:
            key = pair, weight
            if (key in queried or time.monotonic() >= routing_end
                    or deterministic and route_queries >= max_route_queries):
                continue
            queried.add(key)
            route_queries += 1
            try:
                options[pair].add(tuple(graph.path(*pair, size_bytes=representative_bytes, resource_weight=weight)))
            except ValueError:
                # A legal original path remains available; an optional routing
                # heuristic failure never discards that schedule.
                continue
        return options[pair]

    def useful(pair):
        for number, original_path, count in contexts[pair]:
            old_edges, old_resources, old_overhead, old_per_byte = costs.describe(original_path)
            critical_edges, critical_resources = peaks[number]
            original = original_loads[number]
            for path in options[pair]:
                new_edges, new_resources, new_overhead, new_per_byte = costs.describe(path)
                if any(edge in critical_edges and edge not in new_edges for edge in old_edges):
                    return True
                if any(resource in critical_resources and multiplicity > new_resources[resource]
                       for resource, multiplicity in old_resources.items()):
                    return True
                if (old_overhead + count * old_per_byte >= original.time_us - 1e-9
                        and new_overhead + count * new_per_byte < original.time_us - 1e-9):
                    return True
        return False

    source_scores = defaultdict(int)
    for pair, rows in contexts.items():
        source_scores[pair[0]] += sum(count for _, _, count in rows)
    sources = sorted(targets, key=lambda src: (-source_scores[src], src))
    probe_pairs = []
    probe_count = min(12, len(sources))
    probe_sources = [sources[index * len(sources) // probe_count] for index in range(probe_count)]
    for src in probe_sources:
        pairs = [(src, dst) for dst in sorted(targets[src])]
        by_hops = sorted(pairs, key=lambda pair: -max(len(row[1]) for row in contexts[pair]))
        choices = by_hops[:1] + by_hops[len(by_hops) // 2:len(by_hops) // 2 + 1]
        probe_pairs.extend(dict.fromkeys(choices))
    promising = False
    probing_end = routing_end if deterministic else min(routing_end, started + time_budget_s * 0.3)
    for pair in probe_pairs:
        if time.monotonic() >= probing_end:
            break
        alternate_paths(pair)
        promising |= useful(pair)
        if promising:
            break

    changed_rounds = list(solution["rounds"])
    improved = 0
    if promising:
        # Warm source trees by source, then retain routes independently of LRU
        # caches. This prevents repeated Dijkstra work on later rounds.
        warming_end = routing_end if deterministic else min(routing_end, started + time_budget_s * 0.45)
        for src in sources:
            if time.monotonic() >= warming_end:
                break
            for weight in weights:
                # A deadline must stop the same deterministic destination
                # prefix, independent of Python's per-process hash seed.
                for dst in sorted(targets[src]):
                    pair, key = (src, dst), ((src, dst), weight)
                    if (key in queried or time.monotonic() >= warming_end
                            or deterministic and route_queries >= max_route_queries):
                        continue
                    queried.add(key)
                    route_queries += 1
                    try:
                        options[pair].add(tuple(graph.path(src, dst, size_bytes=representative_bytes, resource_weight=weight)))
                    except ValueError:
                        continue
        for number in sorted(range(len(original_loads)), key=lambda i: -original_loads[i].time_us):
            if time.monotonic() >= routing_end or deterministic and path_scores >= max_path_scores:
                break
            round_ = solution["rounds"][number]
            load, new_transfers = _RoundLoad(costs), []
            transfers = sorted(round_["transfers"], key=lambda item: -len(item["chunk_ranges"]))
            for transfer in transfers:
                count = len(transfer["chunk_ranges"]) * chunk_bytes
                original_path = tuple(transfer["path"])
                if time.monotonic() >= routing_end or deterministic and path_scores >= max_path_scores:
                    path = original_path
                else:
                    path_scores += len(options[transfer["src"], transfer["dst"]])
                    path = min(sorted(options[transfer["src"], transfer["dst"]]), key=lambda candidate: load.score(candidate, count))
                load.add(path, count)
                new_transfers.append({**transfer, "path": list(path)})
            if load.time_us < original_loads[number].time_us - 1e-9:
                changed_rounds[number] = {**round_, "transfers": new_transfers}
                improved += 1
    result = {**solution, "rounds": changed_rounds} if improved else solution
    report = replay(task, graph, result)
    if not report["valid"]:
        raise ValueError("route refinement did not pass final replay")
    if report["total_time_us"] > before + max(1e-6, before * 1e-10):
        raise ValueError("route refinement unexpectedly increased communication cost")
    if improved and "metrics" in solution:
        result["metrics"] = {**solution["metrics"], "total_time": report["total_time_us"] / 1e6,
                             "round_times": [value / 1e6 for value in report["round_times_us"]]}
    info = {"probe_pairs": len(probe_pairs), "promising": promising,
            "pairs_with_alternatives": sum(len(paths) > 1 for paths in options.values()),
            "improved_rounds": improved, "before_us": before, "after_us": report["total_time_us"],
            "deadline_reached": time.monotonic() >= routing_end,
            "elapsed_s": round(time.monotonic() - started, 6), "replay_valid": True,
            "work": {"deterministic": deterministic, "route_queries": route_queries,
                     "path_scores": path_scores, "query_limit": max_route_queries,
                     "score_limit": max_path_scores}}
    return result, info
