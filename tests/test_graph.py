"""Connectome graph tests: the sparse structure has to be exactly right."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flykirk.connectome.graph import (  # noqa: E402
    EXCITATORY,
    INHIBITORY,
    Connectome,
    NeuronTable,
    normalize_nt,
)


def tiny_table(nt=("acetylcholine", "gaba", "acetylcholine")) -> NeuronTable:
    n = len(nt)
    return NeuronTable(np.arange(n), ["c"] * n, ["c"] * n, ["intrinsic"] * n, list(nt))


class TestNormalizeNt(unittest.TestCase):
    def test_signs_follow_transmitter(self):
        self.assertEqual(normalize_nt("acetylcholine")[1], EXCITATORY)
        self.assertEqual(normalize_nt("glutamate")[1], EXCITATORY)
        self.assertEqual(normalize_nt("gaba")[1], INHIBITORY)
        self.assertEqual(normalize_nt("histamine")[1], INHIBITORY)

    def test_unknown_transmitter_keeps_the_edge(self):
        name, sign = normalize_nt(None)
        self.assertEqual(name, "unknown")
        self.assertEqual(sign, EXCITATORY)

    def test_aminergic_transmitters_are_flagged_modulatory(self):
        self.assertLess(abs(normalize_nt("dopamine")[1]), 1.0)
        self.assertGreater(normalize_nt("dopamine")[1], 0.0)


class TestConstruction(unittest.TestCase):
    def test_parallel_edges_are_summed(self):
        table = tiny_table()
        graph = Connectome.from_edgelist([0, 0, 0], [1, 1, 2], [2.0, 3.0, 1.0], table)
        self.assertEqual(graph.n_edges, 2)
        self.assertAlmostEqual(graph.n_synapses, 6.0)
        dense = graph.to_dense()
        self.assertAlmostEqual(dense[0, 1], 5.0)
        self.assertAlmostEqual(dense[0, 2], 1.0)

    def test_sign_comes_from_the_presynaptic_neuron(self):
        # neuron 1 is GABAergic, so its outgoing edge is negative
        table = tiny_table()
        graph = Connectome.from_edgelist([1], [0], [4.0], table)
        self.assertAlmostEqual(graph.signed_weights[0], -4.0)

    def test_out_of_range_targets_are_rejected(self):
        with self.assertRaises(ValueError):
            Connectome.from_edgelist([0], [9], [1.0], tiny_table())

    def test_indptr_length_is_checked(self):
        table = tiny_table()
        with self.assertRaises(ValueError):
            Connectome(table, np.zeros(3, dtype=np.int64), np.zeros(0, np.int32), np.zeros(0, np.float32))


class TestSynapticDrive(unittest.TestCase):
    def test_drive_lands_on_the_postsynaptic_neuron(self):
        """CSR rows are presynaptic, so the drive must be the column sum."""
        table = tiny_table()
        graph = Connectome.from_edgelist([0, 0, 2, 1], [1, 2, 1, 0], [5.0, 3.0, 2.0, 4.0], table)
        # 0 --(+5)--> 1, 0 --(+3)--> 2, 2 --(+2)--> 1, 1 --(-4)--> 0
        drive = graph.synaptic_drive(np.array([1.0, 0.0, 1.0], dtype=np.float32))
        np.testing.assert_allclose(drive, [0.0, 7.0, 3.0], atol=1e-6)

    def test_matches_dense_matrix_product(self):
        table = tiny_table(("acetylcholine", "gaba", "glutamate", "gaba"))
        graph = Connectome.from_edgelist([0, 1, 2, 3, 0], [1, 2, 3, 0, 3], [1.0, 2.0, 3.0, 4.0, 5.0], table)
        x = np.array([0.3, 0.9, 0.0, 0.5], dtype=np.float32)
        np.testing.assert_allclose(graph.synaptic_drive(x), graph.to_dense().T @ x, atol=1e-6)

    def test_wrong_activation_length_is_rejected(self):
        graph = Connectome.from_edgelist([0], [1], [1.0], tiny_table())
        with self.assertRaises(ValueError):
            graph.synaptic_drive(np.zeros(7, dtype=np.float32))


class TestQuerying(unittest.TestCase):
    def setUp(self):
        nt = ("acetylcholine", "gaba", "acetylcholine", "gaba")
        classes = ["Kenyon cell", "Kenyon cell", "DN", "DN"]
        supers = ["intrinsic", "intrinsic", "descending", "descending"]
        table = NeuronTable(np.arange(4), classes, classes, supers, list(nt))
        self.graph = Connectome.from_edgelist([0, 1, 2, 3], [1, 2, 3, 0], [1.0, 1.0, 1.0, 1.0], table)

    def test_mask_selects_by_annotation(self):
        np.testing.assert_array_equal(self.graph.mask(super_class="descending"), [False, False, True, True])
        np.testing.assert_array_equal(self.graph.mask(cell_class=("DN",)), [False, False, True, True])

    def test_subgraph_cuts_crossing_edges(self):
        sub = self.graph.subgraph(self.graph.mask(super_class="descending"))
        self.assertEqual(sub.n_neurons, 2)
        self.assertEqual(sub.n_edges, 1)  # 2 -> 3 survives, 3 -> 0 does not

    def test_degrees_count_synapses_not_edges(self):
        graph = Connectome.from_edgelist([0, 0], [1, 1], [2.0, 3.0], tiny_table())
        np.testing.assert_allclose(graph.out_synapses()[0], 5.0)
        np.testing.assert_allclose(graph.in_synapses()[1], 5.0)

    def test_to_dense_refuses_a_real_graph(self):
        with self.assertRaises(ValueError):
            self.graph.to_dense(limit=2)


class TestPersistence(unittest.TestCase):
    def test_roundtrip_preserves_everything_that_matters(self):
        table = tiny_table()
        graph = Connectome.from_edgelist([0, 0], [1, 2], [3.0, 4.0], table, source="unit-test", meta={"seed": 3})
        with tempfile.TemporaryDirectory() as tmp:
            path = graph.save(Path(tmp) / "g.npz")
            back = Connectome.load(path)
        self.assertEqual(back.source, "unit-test")
        self.assertEqual(back.meta["seed"], 3)
        self.assertEqual(back.n_neurons, graph.n_neurons)
        np.testing.assert_allclose(back.signed_weights, graph.signed_weights)


if __name__ == "__main__":
    unittest.main()
