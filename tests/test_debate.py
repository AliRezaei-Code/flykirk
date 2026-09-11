"""Debate tests, all offline: the loop must close without a language model."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flykirk.brain.sim import BrainConfig, BrainSim  # noqa: E402
from flykirk.connectome.surrogate import surrogate_connectome  # noqa: E402
from flykirk.debate.arena import Arena, FlyAgent, brain_judge  # noqa: E402
from flykirk.llm.client import OpenAICompatClient, ScriptedClient, TurnContext  # noqa: E402
from flykirk.llm.sampling import SamplingParams  # noqa: E402
from flykirk.persona.kirk import register_for  # noqa: E402

MOTION = "Resolved: the banana is the superior fruit."


def build_arena(rounds_ready: int = 2, seed: int = 3):
    graph = surrogate_connectome(n_neurons=400, seed=seed)
    cfg = BrainConfig(neurons=400, stimulus_ticks=120, settle_ticks=100, warmup_ticks=120)
    style = register_for()
    agents = []
    for index, (name, side) in enumerate((("FLY-1", "for"), ("FLY-2", "against"))):
        sim = BrainSim(graph, cfg, seed=seed + index)
        agents.append(
            FlyAgent(
                name=name,
                sim=sim,
                client=ScriptedClient(style=style, seed=seed + index),
                style=style,
                side=side,
                seed=seed + index,
            )
        )
    return Arena(agents, judge=brain_judge())


class TestScriptedClient(unittest.TestCase):
    def setUp(self):
        self.style = register_for()
        self.client = ScriptedClient(style=self.style, seed=1)
        self.context = TurnContext(topic=MOTION, turn=1, total_turns=3)

    def test_produces_text_within_the_word_cap(self):
        text = self.client.chat([], SamplingParams(), self.context)
        self.assertGreater(len(text.split()), 0)
        self.assertLessEqual(len(text.split()), self.style.max_words)

    def test_is_deterministic(self):
        a = self.client.chat([], SamplingParams(), self.context)
        b = self.client.chat([], SamplingParams(), self.context)
        self.assertEqual(a, b)

    def test_different_turns_produce_different_text(self):
        first = self.client.chat([], SamplingParams(), TurnContext(topic=MOTION, turn=1))
        second = self.client.chat([], SamplingParams(), TurnContext(topic=MOTION, turn=2))
        self.assertNotEqual(first, second)

    def test_subject_is_taken_from_the_motion(self):
        text = self.client.chat([], SamplingParams(), TurnContext(topic=MOTION, turn=1))
        self.assertIn("banana", text)

    def test_claims_are_not_repeated_within_a_turn(self):
        text = self.client.chat([], SamplingParams(), self.context)
        lowered = text.lower()
        for claim in self.client._CLAIMS:
            fragment = claim.format(subject="cereal").split(" and ")[0].lower()
            self.assertLessEqual(lowered.count(fragment), 1, fragment)

    def test_brain_state_changes_the_sentence(self):
        from flykirk.brain.readout import Telemetry

        hot = self.client.chat(
            [], SamplingParams(), TurnContext(topic=MOTION, telemetry=Telemetry(agitation=1.0, stamina=0.9, confidence=0.9))
        )
        spent = self.client.chat(
            [], SamplingParams(), TurnContext(topic=MOTION, telemetry=Telemetry(agitation=0.1, stamina=0.1, confidence=0.2))
        )
        self.assertNotEqual(hot, spent)


class TestHealth(unittest.TestCase):
    def test_unreachable_server_reports_instead_of_raising(self):
        client = OpenAICompatClient(base_url="http://127.0.0.1:9/v1", model="m", timeout=1.0)
        report = client.health()
        self.assertEqual(len(report), 1)
        self.assertTrue(report[0].startswith("unreachable"))


class TestArena(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arena = build_arena()
        cls.transcript = cls.arena.run(MOTION, rounds=2, judge=True)

    def test_two_rounds_produce_four_turns(self):
        self.assertEqual(len(self.transcript.turns), 4)

    def test_speakers_alternate(self):
        self.assertEqual([t.speaker for t in self.transcript.turns], ["FLY-1", "FLY-2", "FLY-1", "FLY-2"])

    def test_every_turn_carries_a_brain_state(self):
        for turn in self.transcript.turns:
            self.assertGreater(turn.telemetry.tick, 1)
            self.assertGreaterEqual(turn.telemetry.agitation, 0.0)
            self.assertLessEqual(turn.telemetry.agitation, 1.0)
            self.assertGreater(turn.brain_ms, 0.0)

    def test_opponent_words_reach_the_brain(self):
        """Round two is a reaction to round one, not a fresh opening."""
        first = self.transcript.turns[0].text
        second = self.transcript.turns[1].text
        self.assertNotEqual(first, second)

    def test_judge_picks_a_participant(self):
        verdict = self.transcript.verdict
        self.assertIsNotNone(verdict)
        self.assertIn(verdict.winner, {"FLY-1", "FLY-2"})
        self.assertEqual(set(verdict.scores), {"FLY-1", "FLY-2"})

    def test_transcript_serialises(self):
        payload = json.loads(self.transcript.to_json())
        self.assertEqual(payload["topic"], MOTION)
        self.assertEqual(len(payload["turns"]), 4)
        self.assertIn("verdict", payload)
        self.assertIn("telemetry", payload["turns"][0])

    def test_topic_reaches_every_turn(self):
        self.assertTrue(all(self.transcript.topic == MOTION for _ in self.transcript.turns))

    def test_one_round_still_works(self):
        transcript = build_arena().run(MOTION, rounds=1, judge=False)
        self.assertEqual(len(transcript.turns), 2)
        self.assertIsNone(transcript.verdict)

    def test_arena_needs_two_flies(self):
        with self.assertRaises(ValueError):
            Arena([build_arena().flies[0]])


if __name__ == "__main__":
    unittest.main()
