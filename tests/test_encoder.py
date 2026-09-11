"""Sensory encoder tests: determinism, sparsity, and loudness."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flykirk.brain.encoder import SensoryEncoder  # noqa: E402


class TestEncoder(unittest.TestCase):
    def setUp(self):
        self.enc = SensoryEncoder(targets=np.arange(400), n_neurons=400, active_neurons=24, seed=3)

    def test_same_text_same_drive(self):
        a = self.enc.encode("the banana is the superior fruit")
        b = self.enc.encode("the banana is the superior fruit")
        np.testing.assert_array_equal(a, b)

    def test_different_text_different_drive(self):
        a = self.enc.encode("the banana is the superior fruit")
        b = self.enc.encode("monetary policy and interest rates")
        self.assertFalse(np.allclose(a, b))

    def test_drive_is_sparse(self):
        drive = self.enc.encode("the banana is the superior fruit and it is obviously better")
        self.assertLessEqual(int((drive > 0).sum()), 24)
        self.assertGreater(int((drive > 0).sum()), 0)

    def test_drive_is_graph_shaped(self):
        drive = self.enc.encode("banana")
        self.assertEqual(drive.shape[0], 400)

    def test_empty_text_produces_no_drive(self):
        self.assertEqual(float(np.abs(self.enc.encode("")).sum()), 0.0)

    def test_peak_is_scaled_by_the_texts_loudness(self):
        """``drive_mv`` is the ceiling; a quiet sentence sits below it."""
        quiet = self.enc.encode("banana")
        loud = self.enc.encode("BANANA!!!")
        self.assertAlmostEqual(float(loud.max()), self.enc.drive_mv * 1.6, places=4)
        self.assertGreaterEqual(float(quiet.max()), self.enc.drive_mv * 0.7)
        self.assertLess(float(quiet.max()), float(loud.max()))

    def test_louder_text_drives_harder(self):
        quiet = self.enc.encode("banana is a fruit")
        loud = self.enc.encode("BANANA IS A FRUIT!!! PROVE ME WRONG!!!")
        self.assertGreater(float(loud.max()), float(quiet.max()))

    def test_salience_ignores_plain_text(self):
        self.assertLess(SensoryEncoder.salience("the banana is a fruit"), 0.2)
        self.assertGreater(SensoryEncoder.salience("BANANA!!! WRONG!!! WRONG!!!"), 0.5)

    def test_pulse_decays_over_time(self):
        pulse = self.enc.pulse("banana", ticks=200, tau_ms=50.0)
        self.assertEqual(pulse.shape, (200, 400))
        self.assertGreater(float(np.abs(pulse[0]).sum()), float(np.abs(pulse[-1]).sum()))

    def test_content_words_excludes_stopwords(self):
        words = SensoryEncoder.content_words("The banana is obviously the superior fruit")
        self.assertNotIn("the", words)
        self.assertIn("banana", words)


if __name__ == "__main__":
    unittest.main()
