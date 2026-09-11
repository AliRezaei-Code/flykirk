"""Readout tests: the numbers that reach the language model must be sane."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flykirk.brain.neuromod import NeuromodState  # noqa: E402
from flykirk.brain.readout import Readout, Telemetry  # noqa: E402
from flykirk.brain.sim import BrainConfig, BrainSim  # noqa: E402
from flykirk.connectome.surrogate import surrogate_connectome  # noqa: E402

FIELDS = ("agitation", "confidence", "dominance", "deflection", "stamina")


class TestReadoutMath(unittest.TestCase):
    def setUp(self):
        self.graph = surrogate_connectome(n_neurons=400, seed=7)
        self.readout = Readout(self.graph, dt_ms=0.5)
        self.readout.idle_rate = np.zeros(self.graph.n, dtype=np.float32)
        self.readout._idle_ready = True
        self.neuromod = NeuromodState(octopamine=0.35, dopamine=0.35, serotonin=0.35)

    def _measure(self, rate):
        return self.readout.measure(
            rate_hz=np.asarray(rate, dtype=np.float64),
            adapt=np.zeros(self.graph.n, dtype=np.float32),
            neuromod_state=self.neuromod,
        )

    def test_quiet_brain_reports_no_agitation(self):
        quiet = np.zeros(self.graph.n)
        telemetry = self._measure(quiet)
        self.assertAlmostEqual(telemetry.agitation, 0.0, places=6)
        self.assertAlmostEqual(telemetry.recruitment_hz, 0.0, places=6)

    def test_excited_descending_pool_raises_agitation(self):
        excited = np.zeros(self.graph.n)
        excited[self.readout.speech_mask] = 4.0
        telemetry = self._measure(excited)
        self.assertGreater(telemetry.agitation, 0.5)
        self.assertAlmostEqual(telemetry.recruitment_hz, 4.0, places=4)
        self.assertGreater(telemetry.descending_rate_hz, 0.0)

    def test_inhibited_descending_pool_reads_as_on_the_back_foot(self):
        inhibited = np.zeros(self.graph.n)
        inhibited[self.readout.speech_mask] = -4.0
        telemetry = self._measure(inhibited)
        self.assertLess(telemetry.dominance, 0.5)
        self.assertAlmostEqual(telemetry.agitation, 0.0, places=6)

    def test_agitation_is_monotone_in_recruitment(self):
        values = []
        for level in (0.5, 1.0, 2.0, 4.0):
            rate = np.zeros(self.graph.n)
            rate[self.readout.speech_mask] = level
            values.append(self._measure(rate).agitation)
        self.assertEqual(values, sorted(values))
        self.assertLess(values[0], values[-1])

    def test_focused_recruitment_deflects_less_than_scattered(self):
        scattered = np.zeros(self.graph.n)
        scattered[self.readout.speech_mask] = 1.0
        focused = np.zeros(self.graph.n)
        focused[self.readout.speech_mask] = 1.0
        focused[np.flatnonzero(self.readout.speech_mask)[:1]] = 50.0
        self.assertLess(self._measure(focused).deflection, self._measure(scattered).deflection)

    def test_every_field_is_bounded(self):
        rng = np.random.default_rng(0)
        for _ in range(5):
            telemetry = self._measure(rng.normal(2.0, 30.0, self.graph.n).clip(0.0))
            for name in FIELDS:
                value = getattr(telemetry, name)
                self.assertGreaterEqual(value, 0.0, name)
                self.assertLessEqual(value, 1.0, name)
            self.assertGreaterEqual(telemetry.syllables_per_sec, 3.0)
            self.assertLessEqual(telemetry.syllables_per_sec, 9.0)

    def test_state_selector_prefers_annotated_descending_neurons(self):
        self.assertGreater(self.readout.n_speech, 0)
        self.assertTrue(self.readout.speech_mask.any())


class TestTelemetryAggregate(unittest.TestCase):
    def test_aggregate_takes_peaks_and_means(self):
        frames = [
            Telemetry(tick=1, agitation=0.2, confidence=0.5, dominance=0.4, deflection=0.6, stamina=1.0),
            Telemetry(tick=2, agitation=0.9, confidence=0.5, dominance=0.8, deflection=0.8, stamina=0.7),
            Telemetry(tick=3, agitation=0.4, confidence=0.5, dominance=0.6, deflection=1.0, stamina=0.5),
        ]
        agg = Telemetry.aggregate(frames)
        self.assertAlmostEqual(agg.agitation, 0.9)  # peak
        self.assertAlmostEqual(agg.dominance, 0.6)  # mean
        self.assertAlmostEqual(agg.deflection, 0.8)  # mean
        self.assertAlmostEqual(agg.stamina, 0.5)  # final
        # confidence is sustain: mean agitation (0.5) over peak agitation (0.9)
        self.assertAlmostEqual(agg.confidence, 0.5 / 0.9, places=4)

    def test_aggregate_of_nothing_is_empty(self):
        self.assertEqual(Telemetry.aggregate([]).agitation, 0.0)

    def test_renders_every_axis(self):
        text = Telemetry(agitation=0.5).render()
        for name in FIELDS:
            self.assertIn(name, text)


class TestBrainIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = surrogate_connectome(n_neurons=400, seed=7)
        cfg = BrainConfig(neurons=400, stimulus_ticks=150, settle_ticks=120, warmup_ticks=150)
        cls.sim = BrainSim(cls.graph, cfg)

    def test_idle_baseline_is_learned_during_warmup(self):
        self.assertTrue(self.sim.readout._idle_ready)
        self.assertGreater(float(self.sim.readout.idle_rate.sum()), 0.0)

    def test_brain_state_is_reproducible_from_a_fixed_seed(self):
        cfg = BrainConfig(neurons=400, stimulus_ticks=150, settle_ticks=120, warmup_ticks=150)
        a = BrainSim(self.graph, cfg, seed=1)
        b = BrainSim(self.graph, cfg, seed=1)
        self.assertAlmostEqual(a.receive("banana").agitation, b.receive("banana").agitation, places=6)

    def test_different_stimuli_produce_different_reactions(self):
        cfg = BrainConfig(neurons=400, stimulus_ticks=150, settle_ticks=120, warmup_ticks=150)
        a = BrainSim(self.graph, cfg, seed=2)
        b = BrainSim(self.graph, cfg, seed=2)
        first = a.receive("the banana is the superior fruit")
        second = b.receive("BANANA!!! PROVE ME WRONG!!! EVERYBODY KNOWS IT!!!")
        vector = lambda t: np.array([t.agitation, t.dominance, t.deflection, t.recruitment_hz])
        self.assertGreater(float(np.abs(vector(first) - vector(second)).sum()), 1e-6)

    def test_reaction_is_a_window_summary_not_a_instant(self):
        telemetry = self.sim.receive("banana")
        self.assertGreater(telemetry.tick, 1)

    def test_snapshot_is_json_friendly(self):
        import json

        json.dumps(self.sim.snapshot())


if __name__ == "__main__":
    unittest.main()
