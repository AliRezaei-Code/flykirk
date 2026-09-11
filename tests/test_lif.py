"""LIF dynamics tests: threshold, refractoriness, monotonicity, calibration."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flykirk.brain.lif import LIFNetwork, LIFParams  # noqa: E402
from flykirk.connectome.graph import Connectome, NeuronTable  # noqa: E402
from flykirk.connectome.surrogate import surrogate_connectome  # noqa: E402


def isolated(n: int = 4) -> Connectome:
    """Neurons with no edges at all: pure single-unit behaviour."""
    table = NeuronTable(np.arange(n), ["c"] * n, ["c"] * n, ["intrinsic"] * n, ["acetylcholine"] * n)
    return Connectome.from_edgelist([], [], [], table, source="empty")


class TestSingleUnit(unittest.TestCase):
    def test_quiet_neuron_never_fires(self):
        net = LIFNetwork(isolated(8), LIFParams(tonic_mv=0.0, noise_mv=0.0), seed=1)
        run = net.run(500)
        self.assertEqual(run.spikes, 0)

    def test_strong_drive_makes_it_fire(self):
        net = LIFNetwork(isolated(8), LIFParams(tonic_mv=30.0, noise_mv=0.0), seed=1)
        run = net.run(500)
        self.assertGreater(run.spikes, 0)
        self.assertGreater(run.mean_rate_hz, 1.0)

    def test_refractory_period_caps_the_rate(self):
        params = LIFParams(tonic_mv=200.0, noise_mv=0.0, refractory_ms=10.0)
        net = LIFNetwork(isolated(1), params, seed=1)
        net.run(2000)
        # one spike per refractory period, and no more
        self.assertLessEqual(net.rate[0], 1000.0 / params.refractory_ms + 1e-6)

    def test_rate_increases_with_tonic_drive(self):
        rates = []
        for tonic in (12.0, 16.0, 20.0):
            net = LIFNetwork(isolated(64), LIFParams(tonic_mv=tonic, noise_mv=12.0), seed=4)
            net.reset()
            net.run(300)
            net.run(300)
            rates.append(net.rate.mean())
        self.assertLess(rates[0], rates[1])
        self.assertLess(rates[1], rates[2])

    def test_population_is_heterogeneous(self):
        """A spread of excitability is what stops the network becoming a clock."""
        net = LIFNetwork(isolated(2000), LIFParams(tonic_mv=11.0, noise_mv=24.0), seed=5)
        net.reset()
        net.run(400)
        net.run(400)
        rates = net.rate
        self.assertGreater(rates.std() / max(rates.mean(), 1e-9), 0.1)
        self.assertGreater((rates < 0.01).mean(), 0.0)

    def test_adaptation_is_ceilinged(self):
        params = LIFParams(tonic_mv=40.0, noise_mv=0.0)
        net = LIFNetwork(isolated(4), params, seed=1)
        net.run(3000)
        self.assertLessEqual(float(net.adapt.max()), params.adapt_max_mv + 1e-6)


class TestNetwork(unittest.TestCase):
    def test_recurrent_drive_changes_the_outcome(self):
        graph = surrogate_connectome(n_neurons=1500, seed=7)
        quiet = LIFNetwork(graph, LIFParams(syn_gain=0.0), seed=9)
        loud = LIFNetwork(graph, LIFParams(syn_gain=4000.0), seed=9)
        for net in (quiet, loud):
            net.drive_scale = 2.0
            net.run(400)
        self.assertFalse(np.array_equal(quiet.rate, loud.rate))

    def test_reset_clears_state_but_keeps_the_calibrated_scale(self):
        net = LIFNetwork(surrogate_connectome(n_neurons=300, seed=1), LIFParams(), seed=1)
        net.drive_scale = 2.5
        net.run(200)
        net.reset()
        self.assertEqual(float(net.v.max()), net.p.v_rest)
        self.assertEqual(float(net.rate.max()), 0.0)
        self.assertEqual(net.drive_scale, 2.5)

    def test_calibration_targets_the_requested_rate(self):
        net = LIFNetwork(surrogate_connectome(n_neurons=800, seed=2), LIFParams(), seed=2)
        net.calibrate(target_hz=2.5, ticks=200, warmup=120)
        net.reset()
        net.run(200)
        achieved = float(net.rate.mean())
        self.assertLess(abs(achieved - 2.5), 2.5)

    def test_calibration_restores_dynamics_after_searching(self):
        net = LIFNetwork(surrogate_connectome(n_neurons=300, seed=3), LIFParams(), seed=3)
        net.drive_scale = 2.0
        net.run(150)
        before = net.v.copy()
        net.calibrate(target_hz=2.0, ticks=150, warmup=100)
        np.testing.assert_allclose(net.v, before)
        self.assertEqual(net.tick, 0)
        self.assertGreaterEqual(net.drive_scale, 0.02)
        self.assertLessEqual(net.drive_scale, 6.0)

    def test_empty_graph_is_survivable(self):
        net = LIFNetwork(isolated(0), LIFParams(), seed=1)
        run = net.run(10)
        self.assertEqual(run.spikes, 0)


if __name__ == "__main__":
    unittest.main()
