"""Directed topology parsing and switch-only shortest paths.

All capacities are bytes/second. A resource is a shared capacity constraint,
never an extra edge. Routing weights are heuristics; replay computes true cost.
"""

from __future__ import annotations

import heapq
import hashlib
import json
import math
import re
from collections import OrderedDict

from .capacity_backend import native


def bandwidth(value) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean bandwidth")
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        match = re.fullmatch(r"\s*([\d.+eE-]+)\s*([A-Za-z/]+)?\s*", value)
        if not match:
            raise ValueError("invalid bandwidth format")
        number, unit = match.groups()
        factors = {"B/s": 1, "KB/s": 1e3, "MB/s": 1e6, "GB/s": 1e9,
                   "TB/s": 1e12, "bps": 1/8, "Kbps": 1e3/8,
                   "Mbps": 1e6/8, "Gbps": 1e9/8, "Tbps": 1e12/8,
                   "bit/s": 1/8, "Kbit/s": 1e3/8, "Mbit/s": 1e6/8,
                   "Gbit/s": 1e9/8, "Tbit/s": 1e12/8}
        if unit and unit not in factors:
            raise ValueError("unsupported bandwidth unit")
        result = float(number) * factors.get(unit, 1)
    else:
        raise ValueError("invalid bandwidth type")
    if not math.isfinite(result) or result <= 0:
        raise ValueError("bandwidth must be finite and positive")
    return result


class Topology:
    @classmethod
    def from_data(cls, data: dict) -> "Topology":
        if not isinstance(data, dict):
            raise ValueError("topology must be an object")
        obj = cls()
        obj.ranks = list(data.get("ranks", data.get("npu", [])))
        switch_list = list(data.get("switches", []))
        if not obj.ranks or any(not isinstance(n, str) or not n for n in obj.ranks + switch_list):
            raise ValueError("nodes must have nonempty string IDs")
        obj.switches = set(switch_list)
        if len(set(obj.ranks)) != len(obj.ranks) or len(obj.switches) != len(switch_list):
            raise ValueError("duplicate node ID")
        if set(obj.ranks) & obj.switches:
            raise ValueError("rank and switch sets overlap")
        obj.nodes = set(obj.ranks) | obj.switches
        obj.startup_us = obj._overhead(data.get("communication_startup_time_us", 1))
        obj.switch_overhead_us = obj._overhead(data.get("switch_processing_overhead_us", 0.3))
        obj.edges = {}
        bindings = {}
        parsed_rates = {}

        def parse_rate(value):
            if not isinstance(value, str):
                return bandwidth(value)
            if value not in parsed_rates:
                parsed_rates[value] = bandwidth(value)
            return parsed_rates[value]

        for edge in data.get("edges", []):
            src, dst = edge["src"], edge["dst"]
            if src not in obj.nodes or dst not in obj.nodes or src == dst:
                raise ValueError("edge references unknown or identical nodes")
            if not isinstance(edge.get("bidirectional", False), bool):
                raise ValueError("bidirectional must be boolean")
            capacity = parse_rate(edge["bw"])
            pairs = [(src, dst)]
            if edge.get("bidirectional", False):
                pairs.append((dst, src))
            for pair in pairs:
                merged_capacity = obj.edges.get(pair, 0.0) + capacity
                if not math.isfinite(merged_capacity):
                    raise ValueError("merged edge bandwidth must be finite")
                obj.edges[pair] = merged_capacity
                bindings.setdefault(pair, set())
                if "resource_id" in edge:
                    bindings[pair].add(edge["resource_id"])
        obj.resources = {}
        for resource in data.get("resources", []):
            name = resource["id"]
            if not isinstance(name, str) or not name or name in obj.resources:
                raise ValueError("invalid or duplicate resource ID")
            obj.resources[name] = parse_rate(resource.get("capacity", resource.get("bw")))
            for ref in resource.get("edges", []):
                if not isinstance(ref.get("bidirectional", False), bool):
                    raise ValueError("resource bidirectional must be boolean")
                pairs = [(ref["src"], ref["dst"])]
                if ref.get("bidirectional", False):
                    pairs.append((ref["dst"], ref["src"]))
                for pair in pairs:
                    if pair not in obj.edges:
                        raise ValueError("resource references an unknown directed edge")
                    bindings[pair].add(name)
        if any(name not in obj.resources for names in bindings.values() for name in names):
            raise ValueError("edge references an undefined resource")
        obj.edge_resources = {pair: tuple(sorted(names)) for pair, names in bindings.items()}
        obj.adjacency = {node: [] for node in obj.nodes}
        for (src, dst), capacity in obj.edges.items():
            resource_inverse = sum(1 / obj.resources[name] for name in obj.edge_resources[src, dst])
            obj.adjacency[src].append((dst, 1 / capacity, resource_inverse))
        for neighbors in obj.adjacency.values():
            neighbors.sort()
        # Routing uses compact integer indices internally. Ranks are terminals;
        # only switches enter Dijkstra's heap. Keep the public adjacency above
        # for topology analysis and candidate ordering.
        obj._node_names = obj.ranks + sorted(obj.switches)
        obj._node_indices = {name: index for index, name in enumerate(obj._node_names)}
        obj._rank_count = len(obj.ranks)
        obj._indexed_adjacency = [
            [(obj._node_indices[neighbor], edge_inverse, resource_inverse)
             for neighbor, edge_inverse, resource_inverse in obj.adjacency[node]]
            for node in obj._node_names
        ]
        obj._has_shared_edges = any(obj.edge_resources.values())
        obj._weighted_adjacency = OrderedDict()
        obj._native_adjacency = {}
        obj._lazy_adjacency = {}
        obj._rank_exit_bounds = {}
        obj._rank_incoming = {}
        obj._tie_priorities = {}
        obj._paths = OrderedDict()
        # Integer predecessor arrays are much smaller than string-keyed maps.
        # Retain several parameter sets without allowing this cache to grow
        # beyond roughly two million state entries (predecessor, distance, tie).
        obj._path_cache_limit = max(1, min(2048, 650_000 // len(obj._node_names)))
        obj.input_sha256 = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return obj

    @staticmethod
    def _overhead(value) -> float:
        if isinstance(value, bool):
            raise ValueError("invalid overhead")
        result = float(value)
        if result < 0 or not math.isfinite(result):
            raise ValueError("overhead must be finite and nonnegative")
        return result

    def path(self, src: str, dst: str, size_bytes: float = 1,
             resource_weight: float = 1.0) -> list[str]:
        source = self._node_indices.get(src, -1)
        destination = self._node_indices.get(dst, -1)
        rank_count = self._rank_count
        if not 0 <= source < rank_count or not 0 <= destination < rank_count or src == dst:
            raise ValueError("path endpoints must be different ranks")
        if not math.isfinite(size_bytes) or size_bytes <= 0:
            raise ValueError("invalid routing payload")
        if not math.isfinite(resource_weight) or resource_weight < 0:
            raise ValueError("invalid routing weight")
        # With zero switch overhead payload scales every edge equally and does
        # not affect the chosen path. Likewise an unused resource weight does
        # not distinguish routing problems. Reuse only genuinely equivalent
        # weights; positive per-switch overhead still depends on payload size.
        effective_size = size_bytes if self.switch_overhead_us else 1
        effective_weight = resource_weight if self._has_shared_edges else 0.0
        weight_key = (effective_size, effective_weight)
        key = (source, *weight_key)
        if key in self._paths and not isinstance(self._paths[key], tuple):
            self._paths.move_to_end(key)
            return self._lazy_path(self._paths[key], destination, src, dst)
        if key not in self._paths:
            if weight_key not in self._weighted_adjacency:
                scale = effective_size * 1e6
                overhead = self.switch_overhead_us
                self._weighted_adjacency[weight_key] = [
                    [(neighbor, scale * (inverse + effective_weight * resource_inverse)
                      + (overhead if neighbor >= rank_count else 0.0))
                     for neighbor, inverse, resource_inverse in neighbors]
                    for neighbors in self._indexed_adjacency
                ]
                if any(not math.isfinite(cost) for neighbors in self._weighted_adjacency[weight_key]
                       for _, cost in neighbors):
                    # Infinite costs would otherwise satisfy the direct-route
                    # lower-bound comparison as inf <= inf. Reject instead of
                    # returning a route with an unrepresentable timing model.
                    del self._weighted_adjacency[weight_key]
                    raise ValueError("routing costs exceed the finite numeric range")
                if native is not None and hasattr(native, "compile_lazy_graph"):
                    self._lazy_adjacency[weight_key] = native.compile_lazy_graph(self._weighted_adjacency[weight_key])
                incoming = [[] for _ in range(rank_count)]
                exits = [math.inf] * rank_count
                for node, neighbors in enumerate(self._weighted_adjacency[weight_key][rank_count:], rank_count):
                    for neighbor, cost in neighbors:
                        if neighbor < rank_count:
                            incoming[neighbor].append((node, cost))
                            exits[neighbor] = min(exits[neighbor], cost)
                self._rank_incoming[weight_key] = incoming
                # Ranks are terminal nodes. Search switches once, then evaluate
                # only a requested destination's exits instead of all ranks.
                self._weighted_adjacency[weight_key][rank_count:] = [
                    [(node, cost) for node, cost in neighbors if node >= rank_count]
                    for neighbors in self._weighted_adjacency[weight_key][rank_count:]]
                if native is not None and hasattr(native, "shortest_tree") and weight_key not in self._lazy_adjacency:
                    self._native_adjacency[weight_key] = native.compile_adjacency(self._weighted_adjacency[weight_key])
                self._rank_exit_bounds[weight_key] = exits
                if len(self._weighted_adjacency) > 8:
                    expired, _ = self._weighted_adjacency.popitem(last=False)
                    del self._rank_exit_bounds[expired]
                    del self._rank_incoming[expired]
                    self._native_adjacency.pop(expired, None)
                    self._lazy_adjacency.pop(expired, None)
            self._weighted_adjacency.move_to_end(weight_key)
            adjacency = self._weighted_adjacency[weight_key]
            if (src, dst) in self.edges:
                direct_cost = math.inf
                entry_bound = math.inf
                for neighbor, cost in adjacency[source]:
                    if neighbor == destination:
                        direct_cost = cost
                    elif neighbor >= rank_count and cost < entry_bound:
                        entry_bound = cost
                # Every indirect route needs both an ingress and an egress.
                # If their lower bound cannot beat the direct edge, avoid an
                # entire source-tree search (common for fast local mesh links).
                if direct_cost <= entry_bound + self._rank_exit_bounds[weight_key][destination]:
                    return [src, dst]
            if weight_key in self._lazy_adjacency:
                tree = native.lazy_tree(self._lazy_adjacency[weight_key], source, rank_count)
                if tree is not None:
                    self._paths[key] = tree
                    if len(self._paths) > self._path_cache_limit:
                        self._paths.popitem(last=False)
                    return self._lazy_path(tree, destination, src, dst)
                # The legacy heap handles zero-increment costs at extreme
                # floating-point scales. Compile this representation on demand.
                if weight_key not in self._native_adjacency:
                    self._native_adjacency[weight_key] = native.compile_adjacency(adjacency)
            node_count = len(self._node_names)
            distances = [math.inf] * node_count
            ties = [0] * node_count
            previous = [-1] * node_count
            if source not in self._tie_priorities:
                # A destination rank contributes the same priority to all its
                # routes, so only switch priorities need to be accumulated.
                self._tie_priorities[source] = [0] * rank_count + [
                    int.from_bytes(hashlib.sha256((src + "\0" + node).encode()).digest()[:4], "big")
                    for node in self._node_names[rank_count:]
                ]
            priority = self._tie_priorities[source]
            if weight_key in self._native_adjacency:
                previous, distances, ties = native.shortest_tree(
                    self._native_adjacency[weight_key], source, rank_count, priority)
            else:
                distances[source] = 0.0
                queue = []
                push, pop = heapq.heappush, heapq.heappop
                for neighbor, cost in adjacency[source]:
                    distances[neighbor] = cost
                    ties[neighbor] = priority[neighbor]
                    previous[neighbor] = source
                    if neighbor >= rank_count:
                        push(queue, (cost, ties[neighbor], neighbor))
                while queue:
                    cost, tie, node = pop(queue)
                    if cost != distances[node] or tie != ties[node]:
                        continue
                    for neighbor, weight in adjacency[node]:
                        next_cost = cost + weight
                        next_tie = tie + priority[neighbor]
                        if next_cost < distances[neighbor] or (
                            next_cost == distances[neighbor] and next_tie < ties[neighbor]
                        ):
                            distances[neighbor] = next_cost
                            ties[neighbor] = next_tie
                            previous[neighbor] = node
                            if neighbor >= rank_count:
                                push(queue, (next_cost, next_tie, neighbor))
            # Keep incoming edges with this search state: the weight cache can
            # evict independently while a source tree still has live queries.
            self._paths[key] = (previous, distances, ties, bytearray(rank_count),
                                self._rank_incoming[weight_key])
            if len(self._paths) > self._path_cache_limit:
                self._paths.popitem(last=False)
        self._paths.move_to_end(key)
        previous, distances, ties, resolved, incoming = self._paths[key]
        if not resolved[destination]:
            # Equal-cost/equal-priority exits retain the original Dijkstra pop
            # order. Direct routes were initialized before every switch pop.
            best = (distances[destination], ties[destination], -math.inf, -1, -1)
            for node, weight in incoming[destination]:
                if not math.isfinite(distances[node]):
                    continue
                choice = (distances[node] + weight, ties[node],
                          distances[node], ties[node], node)
                if choice < best:
                    best = choice
                    previous[destination] = node
            distances[destination], ties[destination] = best[:2]
            resolved[destination] = 1
        if previous[destination] < 0:
            raise ValueError(f"no switch-only route between {src} and {dst}")
        route = [destination]
        while route[-1] != source:
            route.append(previous[route[-1]])
        return [self._node_names[node] for node in reversed(route)]

    def _lazy_path(self, tree, destination, src, dst):
        route = native.lazy_path(tree, destination, self._switch_priority)
        if route is None:
            raise ValueError(f"no switch-only route between {src} and {dst}")
        return [self._node_names[i] for i in route]

    def _switch_priority(self, source, node):
        payload = (self._node_names[source] + "\0" + self._node_names[node]).encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")

    def summary(self) -> dict:
        """Compact input evidence for planning, without all paths or chunk lists."""
        capacities = list(self.edges.values())
        ranks = set(self.ranks)
        edge_types = {}
        for (src, dst), bw in self.edges.items():
            kind = ("rank" if src in ranks else "switch") + "->" + ("rank" if dst in ranks else "switch")
            edge_types.setdefault(kind, []).append(bw)
        domains = {}
        for rank in self.ranks:
            outgoing = tuple(dst for dst, _, _ in self.adjacency[rank] if dst in self.switches)
            domains.setdefault(outgoing, []).append(rank)
        return {
            "input_sha256": self.input_sha256,
            "rank_count": len(self.ranks), "switch_count": len(self.switches),
            "directed_edge_count": len(self.edges), "resource_count": len(self.resources),
            "min_edge_bytes_per_second": min(capacities, default=0),
            "max_edge_bytes_per_second": max(capacities, default=0),
            "startup_us": self.startup_us, "switch_overhead_us": self.switch_overhead_us,
            "edge_types": {kind: {"count": len(values), "min_bw": min(values), "max_bw": max(values),
                                  "mean_bw": sum(values) / len(values)} for kind, values in edge_types.items()},
            "one_way_edges": sum((dst, src) not in self.edges for src, dst in self.edges),
            "asymmetric_edges": sum(self.edges.get((dst, src), bw) != bw for (src, dst), bw in self.edges.items()),
            "rank_domains_total": len(domains),
            "rank_domains_sample": [{"rank_count": len(members), "ranks_sample": members[:4],
                                     "outgoing_switches": list(neighbors)}
                                    for neighbors, members in list(domains.items())[:16]],
            "edges_sample": [{"src": src, "dst": dst, "bw": bw,
                              "resources": list(self.edge_resources[src, dst])}
                             for (src, dst), bw in sorted(self.edges.items(), key=lambda item: item[1])[:16]],
            "tightest_resources": [
                {"id": name, "capacity_bytes_per_second": capacity,
                 "member_edges": sum(name in ids for ids in self.edge_resources.values()),
                 "members_sample": [{"src": src, "dst": dst} for (src, dst), ids in
                                    self.edge_resources.items() if name in ids][:4]}
                for name, capacity in sorted(self.resources.items(), key=lambda item: item[1])[:16]
            ],
        }
