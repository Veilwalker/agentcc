"""Differential tests for native arithmetic, ownership and backend selection."""
import math
import random
import unittest
from unittest.mock import patch

from src import capacity_backend as backend, fabric_routes
from src.fabric_routes import CapacityCosts, PythonCapacityAssignment
from src.topology import Topology
from tests.test_solver import make_case


class BackendSelectionTests(unittest.TestCase):
    def test_python_never_imports_native_module(self):
        with patch.object(backend.importlib, 'import_module', side_effect=AssertionError):
            self.assertIsNone(backend.load_native('python'))

    def test_required_cpp_fails_visibly_while_auto_can_use_reference(self):
        with patch.object(backend.importlib, 'import_module', side_effect=ImportError):
            self.assertIsNone(backend.load_native('auto'))
            with self.assertRaisesRegex(RuntimeError, 'C\+\+ capacity extension unavailable'):
                backend.load_native('cpp')

    def test_invalid_mode_and_extension_version_are_rejected(self):
        with self.assertRaises(ValueError):
            backend.load_native('typo')
        with patch.object(backend.importlib, 'import_module', return_value=type('Module', (), {'API_VERSION': 0})):
            with self.assertRaisesRegex(RuntimeError, 'unsupported'):
                backend.load_native('cpp')


@unittest.skipIf(backend.native is None, 'build native extension before native differential tests')
class NativeCapacityTests(unittest.TestCase):
    def test_cached_graph_matches_original_compiler_bit_for_bit(self):
        rng = random.Random(1112)
        nodes = [f'节点{i}' for i in range(9)]
        edges = {(a, b): rng.uniform(1e6, 1e12) for a in nodes for b in nodes if a != b}
        pools = {f'pool{i}': rng.uniform(1e6, 1e12) for i in range(13)}
        members = {edge: rng.sample(list(pools), rng.randrange(5)) for edge in edges}
        indices = {('edge', e): i for i, e in enumerate(edges)}
        indices.update({('pool', p): len(edges) + i for i, p in enumerate(pools)})
        compiled = backend.native.make_capacity_graph(edges, members, pools, indices, .7, .13)
        for _ in range(1000):
            path = tuple(rng.sample(nodes, rng.randrange(2, 10)))
            expected = backend.native.compile_path(path, edges, members, pools, indices, .7, .13)
            self.assertEqual(expected, backend.native.compile_cached_path(compiled, path))
        # Multiplicity is not deduplicated, including repeated edges.
        path = (nodes[0], nodes[1], nodes[0], nodes[1])
        self.assertEqual(backend.native.compile_path(path, edges, members, pools, indices, .7, .13),
                         backend.native.compile_cached_path(compiled, path))

    def test_cached_graph_is_request_local_and_rejects_invalid_paths(self):
        edges = {('a', 'b'): 3e8, ('b', 'c'): 7e8}
        indices = {('edge', e): i for i, e in enumerate(edges)}
        members = {e: [] for e in edges}
        one = backend.native.make_capacity_graph(edges, members, {}, indices, 1., 2.)
        edges['a', 'b'] *= 2
        two = backend.native.make_capacity_graph(edges, members, {}, indices, 1., 2.)
        path = ('a', 'b', 'c')
        self.assertNotEqual(backend.native.compile_cached_path(one, path),
                            backend.native.compile_cached_path(two, path))
        for bad in [(), ('a',), ['a', 'b'], ('a', 'c'), ('c', 'b'), ('unknown', 'b')]:
            with self.subTest(path=bad), self.assertRaises((ValueError, KeyError)):
                backend.native.compile_cached_path(one, bad)
        with self.assertRaises(ValueError):
            backend.native.make_capacity_graph({('a', 'b'): 0}, {('a', 'b'): []}, {}, indices, 1., 2.)

    def assignments(self, resources=64):
        costs = type('Costs', (), {'indices': list(range(resources))})()
        return PythonCapacityAssignment(costs), backend.NativeCapacityAssignment(costs)

    def test_six_thousand_greedy_choices_and_ties_match_exactly(self):
        rng = random.Random(671)
        for _ in range(200):
            reference, native = self.assignments()
            for number in range(30):
                options = []
                for j in range(rng.randint(1, 16)):
                    terms = tuple((i, rng.choice([.001, .01, .125, 1., 2.5]))
                                  for i in rng.sample(range(64), rng.randint(1, 12)))
                    options.append((terms, rng.choice([0., .3, 1.3]), .1 + j / 10,
                                    (str(number), str(j))))
                if number % 3 == 0:
                    options.append(options[0][:-1] + (('tie', 'last'),))
                count, packet = rng.randint(1, 10000), rng.randint(1, 100)
                self.assertEqual(reference.choose(options, count, packet), native.choose(options, count, packet))
                self.assertEqual(reference.time_us, native.time_us)

    def test_path_compilation_counts_shared_resource_multiplicity(self):
        _, topology = make_case(4)
        topology['resources'] = [
            {'id': 'both', 'capacity': 2e8, 'edges': [{'src': 'r0', 'dst': 'S0'}, {'src': 'S0', 'dst': 'r1'}]},
            {'id': 'ingress', 'capacity': 4e8, 'edges': [{'src': 'r0', 'dst': 'S0'}]}]
        graph = Topology.from_data(topology)
        paths = [('r0', 'S0', 'r1'), ('r2', 'S0', 'r1')]
        native = CapacityCosts(graph)
        compiled = [native.compile(path) for path in paths]
        with patch.object(fabric_routes, 'native', None):
            reference = CapacityCosts(graph)
            self.assertEqual(compiled, [reference.compile(path) for path in paths])
        pool_index = native.indices['pool', 'both']
        self.assertIn((pool_index, 2 * 1e6 / 2e8), compiled[0][0])
        self.assertIs(native.compile(paths[0]), compiled[0])

    def test_native_round_state_is_not_shared(self):
        _, one = self.assignments()
        _, two = self.assignments()
        option = (((0, .125),), .3, .25, ('a', 'b'))
        one.choose([option], 100)
        self.assertEqual(two.time_us, 0.0)
        two.choose([option], 100)
        self.assertEqual(one.time_us, two.time_us)

    def test_invalid_native_arguments_raise_python_errors(self):
        _, native = self.assignments(4)
        cases = [(), (((), -1., .5, ('a', 'b')),),
                 ((((7, 1.),), 0., .5, ('a', 'b')),),
                 ((((0, math.nan),), 0., .5, ('a', 'b')),),
                 ((((0, 1.),), 0., math.inf, ('a', 'b')),), ((1, 2),)]
        for options in cases:
            with self.subTest(options=options), self.assertRaises(ValueError):
                native.choose(options, 1)
            self.assertEqual(native.time_us, 0.0)
        with self.assertRaises(ValueError):
            backend.native.make_assignment(-1)
        with self.assertRaises((OverflowError, MemoryError)):
            backend.native.make_assignment(2**63 - 1)


if __name__ == '__main__':
    unittest.main()
