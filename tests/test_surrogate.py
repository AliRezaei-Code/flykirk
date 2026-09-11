"""Surrogate connectome tests: determinism, scale, and honest labelling."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flykirk.connectome.surrogate import PROVENANCE, surrogate_connectome  # noqa: E402


class TestSurrogate(unittest.TestCase):
    def test_same_seed_same_graph(self):
        a = surrogate_connectome(n_neurons=400, seed=11)
        b = surrogate_connectome(n_neurons=400, seed=11)
        np.testing.assert_array_equal(a.indices, b.indices)
        np.testing.assert_array_equal(a.weights, b.weights)

    def test_different_seed_different_graph(self):
        a = surrogate_connectome(n_neurons=400, seed=1)
        b = surrogate_connectome(n_neurons=400, seed=2)
        self.assertFalse(np.array_equal(a.weights, b.weights))

    def test_neuron_count_is_exact(self):
        graph = surrogate_connectome(n_neurons=500, seed=3)
        self.assertEqual(graph.n_neurons, 500)
        self.assertEqual(len(set(graph.neurons.cell_class.tolist())) > 8, True)

    def test_marked_as_synthetic(self):
        graph = surrogate_connectome(n_neurons=200, seed=1)
        self.assertEqual(graph.source, PROVENANCE)
        self.assertEqual(graph.meta["provenance"], PROVENANCE)
        self.assertIn("not FlyWire", graph.meta["warning"])

    def test_root_ids_cannot_collide_with_flywire_ids(self):
        graph = surrogate_connectome(n_neurons=200, seed=1)
        self.assertTrue((graph.neurons.root_id < 0).all())

    def test_edge_budget_is_respected(self):
        graph = surrogate_connectome(n_neurons=2000, seed=5, max_edges=5000)
        self.assertLessEqual(graph.n_edges, 5000)
        self.assertTrue(graph.meta["edge_budget_hit"])

    def test_too_few_neurons_is_rejected(self):
        with self.assertRaises(ValueError):
            surrogate_connectome(n_neurons=3)

    def test_graph_has_both_signs(self):
        graph = surrogate_connectome(n_neurons=1000, seed=7)
        self.assertTrue((graph.neurons.sign < 0).sum() > 0)
        self.assertTrue((graph.neurons.sign > 0).sum() > 0)


if __name__ == "__main__":
    unittest.main()
