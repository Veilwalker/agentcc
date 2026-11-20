"""Optional two-axis tree search must skip an incomplete baseline."""
import unittest
from src import mesh_elbows


class MeshElbowTests(unittest.TestCase):
    def test_missing_or_empty_reference_waves_skip_the_candidate(self):
        for reference in ([], [(0., {})] * 6):
            with self.subTest(reference=reference):
                self.assertIsNone(mesh_elbows.improve(None, reference))
