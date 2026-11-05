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
        for edge in data.get("edges", []):
            src, dst = edge["src"], edge["dst"]
            if src not in obj.nodes or dst not in obj.nodes or src == dst:
                raise ValueError("edge references unknown or identical nodes")
            if not isinstance(edge.get("bidirectional", False), bool):
                raise ValueError("bidirectional must be boolean")
            capacity = bandwidth(edge["bw"])
            pairs = [(src, dst)]
            if edge.get("bidirectional", False):
                pairs.append((dst, src))
            for pair in pairs:
                obj.edges[pair] = obj.edges.get(pair, 0.0) + capacity
                bindings.setdefault(pair, set())
                if "resource_id" in edge:
                    bindings[pair].add(edge["resource_id"])
        obj.resources = {}
        for resource in data.get("resources", []):
            name = resource["id"]
            if not isinstance(name, str) or not name or name in obj.resources:
                raise ValueError("invalid or duplicate resource ID")
            obj.resources[name] = bandwidth(resource.get("capacity", resource.get("bw")))
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
        obj._paths = OrderedDict()
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
        if src not in self.ranks or dst not in self.ranks or src == dst:
            raise ValueError("path endpoints must be different ranks")
        if not math.isfinite(size_bytes) or size_bytes <= 0:
            raise ValueError("invalid routing payload")
        if not math.isfinite(resource_weight) or resource_weight < 0:
            raise ValueError("invalid routing weight")
        key = (src, size_bytes, resource_weight)
        if key not in self._paths:
            distances, previous = {src: (0.0, 0)}, {}
            priority = {node: int.from_bytes(hashlib.sha256((src + "\0" + node).encode()).digest()[:4], "big")
                        for node in self.nodes}
            queue = [(0.0, 0, src)]
            while queue:
                cost, tie, node = heapq.heappop(queue)
                if (cost, tie) != distances[node]:
                    continue
                # A reached rank is a destination, never an intermediate hop.
                if node != src and node not in self.switches:
                    continue
                for neighbor, edge_inverse, resource_inverse in self.adjacency[node]:
                    weight = size_bytes * 1e6 * (edge_inverse + resource_weight * resource_inverse)
                    if neighbor in self.switches:
                        weight += self.switch_overhead_us
                    candidate = (cost + weight, tie + priority[neighbor])
                    if candidate < distances.get(neighbor, (math.inf, 0)):
                        distances[neighbor] = candidate
                        previous[neighbor] = node
                        heapq.heappush(queue, (*candidate, neighbor))
            self._paths[key] = previous
            if len(self._paths) > 512:
                self._paths.popitem(last=False)
        self._paths.move_to_end(key)
        previous = self._paths[key]
        if dst not in previous:
            raise ValueError(f"no switch-only route between {src} and {dst}")
        route = [dst]
        while route[-1] != src:
            route.append(previous[route[-1]])
        return route[::-1]

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
