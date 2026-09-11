"""Persona and sampling tests: brain state must actually steer the prompt."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flykirk.brain.readout import Telemetry  # noqa: E402
from flykirk.llm.sampling import SamplingParams, from_telemetry  # noqa: E402
from flykirk.persona.kirk import (  # noqa: E402
    DEFAULT_REGISTER,
    PARODY_BANNER,
    build_system_prompt,
    register_for,
    stage_directions,
)


def telemetry(**overrides):
    base = dict(agitation=0.5, confidence=0.6, dominance=0.5, deflection=0.7, stamina=0.9, syllables_per_sec=5.5)
    base.update(overrides)
    return Telemetry(**base)


class TestRegister(unittest.TestCase):
    def test_default_register_is_registered(self):
        self.assertIs(register_for(None), DEFAULT_REGISTER)
        self.assertIs(register_for("campus-debate"), DEFAULT_REGISTER)

    def test_unknown_register_is_an_error(self):
        with self.assertRaises(KeyError):
            register_for("does-not-exist")


class TestStageDirections(unittest.TestCase):
    def test_agitated_fly_is_told_to_speed_up(self):
        notes = " ".join(stage_directions(telemetry(agitation=0.95)))
        self.assertIn("HOTTING UP", notes)

    def test_calm_fly_is_told_to_slow_down(self):
        notes = " ".join(stage_directions(telemetry(agitation=0.1)))
        self.assertIn("flat", notes)

    def test_losing_fly_is_told_to_pivot_not_concede(self):
        notes = " ".join(stage_directions(telemetry(dominance=0.1)))
        self.assertIn("back foot", notes)
        self.assertIn("pivot", notes)

    def test_scattered_fly_is_told_to_focus(self):
        notes = " ".join(stage_directions(telemetry(deflection=0.99)))
        self.assertIn("scattered", notes)

    def test_speaking_rate_appears_in_the_directions(self):
        notes = " ".join(stage_directions(telemetry(syllables_per_sec=7.2)))
        self.assertIn("7.2", notes)
        self.assertIn("very fast", notes)

    def test_every_state_produces_directions(self):
        self.assertGreaterEqual(len(stage_directions(telemetry())), 1)


class TestPrompt(unittest.TestCase):
    def test_prompt_carries_the_parody_banner(self):
        prompt = build_system_prompt(DEFAULT_REGISTER, telemetry(), "Resolved: the banana is the superior fruit.")
        self.assertIn(PARODY_BANNER, prompt)

    def test_prompt_contains_the_motion_and_opponent(self):
        prompt = build_system_prompt(
            DEFAULT_REGISTER, telemetry(), "Resolved: soup is not a meal.", opponent="FLY-2", turn=2, total_turns=3
        )
        self.assertIn("soup is not a meal", prompt)
        self.assertIn("FLY-2", prompt)
        self.assertIn("Turn 2 of 3", prompt)

    def test_prompt_contains_the_live_brain_readout(self):
        prompt = build_system_prompt(DEFAULT_REGISTER, telemetry(agitation=0.81, syllables_per_sec=6.4), "motion")
        self.assertIn("0.81", prompt)
        self.assertIn("6.4", prompt)
        self.assertIn("octopamine", prompt)

    def test_brain_state_visibly_changes_the_prompt(self):
        calm = build_system_prompt(DEFAULT_REGISTER, telemetry(agitation=0.1, dominance=0.9), "motion")
        hot = build_system_prompt(DEFAULT_REGISTER, telemetry(agitation=0.99, dominance=0.1), "motion")
        self.assertNotEqual(calm, hot)
        self.assertIn("back foot", hot)
        self.assertNotIn("back foot", calm)

    def test_prompt_states_the_claim_and_word_caps(self):
        prompt = build_system_prompt(DEFAULT_REGISTER, telemetry(), "motion")
        self.assertIn(str(DEFAULT_REGISTER.max_claims), prompt)
        self.assertIn(str(DEFAULT_REGISTER.max_words), prompt)


class TestSampling(unittest.TestCase):
    def test_payload_is_openai_shaped(self):
        payload = SamplingParams(seed=4).as_payload("model-x", [{"role": "user", "content": "hi"}])
        self.assertEqual(payload["model"], "model-x")
        self.assertEqual(payload["seed"], 4)
        self.assertFalse(payload["stream"])
        self.assertIn("temperature", payload)

    def test_seed_is_omitted_when_unset(self):
        self.assertNotIn("seed", SamplingParams().as_payload("m", []))

    def test_agitation_raises_temperature(self):
        calm = from_telemetry(telemetry(agitation=0.05))
        hot = from_telemetry(telemetry(agitation=1.0))
        self.assertGreater(hot.temperature, calm.temperature)

    def test_low_confidence_shortens_the_turn(self):
        steady = from_telemetry(telemetry(confidence=1.0))
        flinching = from_telemetry(telemetry(confidence=0.0))
        self.assertLess(flinching.max_tokens, steady.max_tokens)

    def test_deflection_raises_presence_penalty(self):
        focused = from_telemetry(telemetry(deflection=0.0))
        scattered = from_telemetry(telemetry(deflection=1.0))
        self.assertGreater(scattered.presence_penalty, focused.presence_penalty)

    def test_bounds_hold_at_the_extremes(self):
        for ag in (0.0, 1.0):
            for conf in (0.0, 1.0):
                params = from_telemetry(telemetry(agitation=ag, confidence=conf, deflection=1.0, stamina=0.0))
                self.assertGreaterEqual(params.temperature, 0.4)
                self.assertLessEqual(params.temperature, 1.35)
                self.assertGreaterEqual(params.top_p, 0.7)
                self.assertLessEqual(params.top_p, 1.0)
                self.assertGreaterEqual(params.max_tokens, 48)
                self.assertLessEqual(params.max_tokens, 400)
                self.assertGreaterEqual(params.frequency_penalty, 0.0)
                self.assertLessEqual(params.frequency_penalty, 1.2)


if __name__ == "__main__":
    unittest.main()
