"""Debate orchestration: two flies, one motion, a closed sensory loop.

The important structural detail is that an opponent's *words* are fed back into
the other fly's sensory neurons. A turn is not a chat message appended to a
context window; it is a stimulus that lands on the brain, moves it, and the
resulting state writes the next prompt and sets the sampler. Mood carries across
turns, so a long debate genuinely degrades or inflames the participants.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..brain.readout import Telemetry
from ..brain.sim import BrainSim
from ..llm.client import ChatClient, TurnContext
from ..llm.sampling import SamplingParams, from_telemetry
from ..persona.kirk import PersonaStyle, build_system_prompt

__all__ = ["Turn", "Verdict", "DebateTranscript", "FlyAgent", "Arena"]


@dataclass
class Turn:
    round: int
    speaker: str
    side: str
    text: str
    telemetry: Telemetry
    sampling: SamplingParams
    brain_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round": self.round,
            "speaker": self.speaker,
            "side": self.side,
            "text": self.text,
            "brain_ms": round(self.brain_ms, 2),
            "sampling": asdict(self.sampling),
            "telemetry": self.telemetry.as_dict(),
        }


@dataclass
class Verdict:
    winner: str
    reason: str
    scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"winner": self.winner, "reason": self.reason, "scores": self.scores}


@dataclass
class DebateTranscript:
    topic: str
    flies: List[Dict[str, Any]]
    turns: List[Turn]
    verdict: Optional[Verdict] = None
    connectome: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "connectome": self.connectome,
            "flies": self.flies,
            "turns": [t.to_dict() for t in self.turns],
            "verdict": self.verdict.to_dict() if self.verdict else None,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


@dataclass
class FlyAgent:
    """One participant: a brain, a client, and a register."""

    name: str
    sim: BrainSim
    client: ChatClient
    style: PersonaStyle
    side: str = "for"
    seed: int = 0
    turn_count: int = 0

    def respond(
        self,
        topic: str,
        opponent_text: str = "",
        opponent: str = "",
        turn: int = 1,
        total_turns: int = 1,
        on_tick: Optional[Callable[[Any], None]] = None,
    ) -> Turn:
        started = time.perf_counter()
        # The opponent's argument is the stimulus. Hearing it is what moves the
        # brain before the fly opens its mouth.
        stimulus = opponent_text.strip() or topic
        telemetry = self.sim.receive(stimulus, on_tick=on_tick)
        self.sim.settle(max(40, self.sim.config.settle_ticks // 3), on_tick=on_tick)

        system = build_system_prompt(
            self.style,
            telemetry,
            topic=topic,
            opponent=opponent or None,
            side=self.side,
            turn=turn,
            total_turns=total_turns,
        )
        user = (
            f"{opponent} just said:\n\n{opponent_text}\n\nAnswer them."
            if opponent_text.strip()
            else f"Open the debate on: {topic}"
        )
        params = from_telemetry(telemetry, SamplingParams(seed=self.seed * 1000 + turn))
        context = TurnContext(
            topic=topic,
            side=self.side,
            opponent_text=opponent_text,
            telemetry=telemetry,
            turn=turn,
            total_turns=total_turns,
            opponent=opponent,
        )
        text = self.client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}], params, context
        )
        self.turn_count += 1
        return Turn(
            round=turn,
            speaker=self.name,
            side=self.side,
            text=_clean(text),
            telemetry=telemetry,
            sampling=params,
            brain_ms=(time.perf_counter() - started) * 1000.0,
        )


def _clean(text: str) -> str:
    """Strip the stage directions models love to add anyway."""
    out = text.strip()
    out = re.sub(r"^\s*\*+.*?\*+\s*", "", out, flags=re.S)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


class Arena:
    """Runs a debate between two flies and asks a judge to settle it."""

    def __init__(
        self,
        flies: List[FlyAgent],
        judge: Optional[ChatClient] = None,
        judge_model_name: str = "judge",
        on_turn: Optional[Callable[[Turn], None]] = None,
        on_tick: Optional[Callable[[str, Any], None]] = None,
    ) -> None:
        if len(flies) < 2:
            raise ValueError("a debate needs at least two flies")
        self.flies = flies
        self.judge = judge
        self.judge_model_name = judge_model_name
        self.on_turn = on_turn
        self.on_tick = on_tick
        self.turns: List[Turn] = []

    def run(self, topic: str, rounds: int = 3, judge: bool = True) -> DebateTranscript:
        for rnd in range(1, max(1, rounds) + 1):
            for index, fly in enumerate(self.flies):
                opponent = self.flies[(index + 1) % len(self.flies)]
                last_opponent_text = self._last_text_from(opponent.name)
                tick_hook = None
                if self.on_tick is not None:
                    tick_hook = lambda sample, _name=fly.name, _r=rnd: self.on_tick(_name, sample)  # noqa: E731
                turn = fly.respond(
                    topic=topic,
                    opponent_text=last_opponent_text,
                    opponent=opponent.name,
                    turn=rnd,
                    total_turns=rounds,
                    on_tick=tick_hook,
                )
                self.turns.append(turn)
                if self.on_turn is not None:
                    self.on_turn(turn)

        verdict = self.judge_debate(topic) if judge else None
        return DebateTranscript(
            topic=topic,
            flies=[
                {
                    "name": f.name,
                    "side": f.side,
                    "seed": f.seed,
                    "connectome": f.sim.connectome.source,
                }
                for f in self.flies
            ],
            turns=self.turns,
            verdict=verdict,
            connectome=self.flies[0].sim.connectome.source,
        )

    def _last_text_from(self, name: str) -> str:
        for turn in reversed(self.turns):
            if turn.speaker == name:
                return turn.text
        return ""

    # ------------------------------------------------------------------ judging

    def judge_debate(self, topic: str) -> Verdict:
        if self.judge is None or isinstance(self.judge, _BrainJudge):
            return self._brain_verdict()
        transcript = "\n\n".join(f"{t.speaker} ({t.side}): {t.text}" for t in self.turns)
        prompt = (
            "You are scoring a debate between two fruit flies. Judge only the debating, not whether "
            "you agree with the motion.\n"
            f"Motion: {topic}\n\nTranscript:\n{transcript}\n\n"
            "Reply with strict JSON only: "
            '{"winner": "<fly name>", "scores": {"<fly name>": 0-10, ...}, "reason": "<= 40 words"}'
        )
        try:
            raw = self.judge.chat(
                [
                    {"role": "system", "content": "You return only strict JSON."},
                    {"role": "user", "content": prompt},
                ],
                SamplingParams(temperature=0.2, top_p=0.9, max_tokens=220),
            )
            payload = _extract_json(raw)
            if payload and "winner" in payload:
                return Verdict(
                    winner=str(payload.get("winner", "unknown")),
                    reason=str(payload.get("reason", "")).strip(),
                    scores={str(k): float(v) for k, v in (payload.get("scores") or {}).items()},
                )
        except Exception:  # noqa: BLE001 - a broken judge falls back to the brain
            pass
        return self._brain_verdict()

    def _brain_verdict(self) -> Verdict:
        """Fall back to the brains themselves: the steadier, more dominant fly wins."""
        scores: Dict[str, float] = {}
        for fly in self.flies:
            own = [t.telemetry for t in self.turns if t.speaker == fly.name]
            if not own:
                scores[fly.name] = 0.0
                continue
            agitation = sum(t.agitation for t in own) / len(own)
            dominance = sum(t.dominance for t in own) / len(own)
            confidence = sum(t.confidence for t in own) / len(own)
            deflection = sum(t.deflection for t in own) / len(own)
            scores[fly.name] = round(10.0 * (0.30 * agitation + 0.30 * dominance + 0.40 * confidence) * (1.0 - 0.25 * deflection), 3)
        winner = max(scores, key=lambda k: scores[k]) if scores else "unknown"
        return Verdict(
            winner=winner,
            reason="",
            scores=scores,
        )


class _BrainJudge:
    """Marker type: asks the arena to score from brain telemetry alone."""


def _extract_json(raw: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def brain_judge() -> ChatClient:
    """A judge that is really just the flies' own telemetry."""
    judge = _BrainJudge()  # type: ignore[assignment]
    return judge  # type: ignore[return-value]
