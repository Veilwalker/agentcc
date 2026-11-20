"""Verify coordinate reconstruction without using rank labels or list order."""
import random
from types import SimpleNamespace
import unittest

from src.grid_axes import axes


def grid_components():
    cells = [[['node' + str((d*16 + y*4 + x)*17 % 71)
               for x in range(4)] for y in range(4)] for d in range(2)]
    ranks = [rank for block in cells for row in block for rank in row]
    random.Random(17).shuffle(ranks)
    edges = {}
    for block in cells:
        for y, row in enumerate(block):
            for x, rank in enumerate(row):
                for other in range(4):
                    if x != other:
                        edges[rank, row[other]] = 20.
                    if y != other:
                        edges[rank, block[other][x]] = 5.
    return cells, SimpleNamespace(ranks=ranks, edges=edges)


class PhysicalGridTests(unittest.TestCase):
    def test_strong_axis_and_domains_survive_opaque_rank_order(self):
        cells, graph = grid_components()
        a, b = cells[0][0][:2]
        del graph.edges[a,b]
        actual = axes(graph)
        self.assertIs(actual, axes(graph))
        self.assertEqual({frozenset(sum(block, [])) for block in actual},
                         {frozenset(sum(block, [])) for block in cells})
        self.assertEqual({frozenset(row) for block in actual for row in block},
                         {frozenset(row) for block in cells for row in block})

    def test_one_clique_per_domain_has_no_second_axis(self):
        cells, graph = grid_components()
        for block in cells:
            for y, row in enumerate(block):
                for x, rank in enumerate(row):
                    for other in range(4):
                        if other != y:
                            del graph.edges[rank, block[other][x]]
        self.assertIsNone(axes(graph))
