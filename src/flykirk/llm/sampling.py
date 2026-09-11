"""Brain state -> sampler settings.

The brain does not just write the prompt; it also sets the knobs. An agitated
fly samples hotter and repeats itself less, a deflected fly wanders further off
topic, a flinching fly gets a shorter leash. This is the second half of the
coupling, and it is the half that makes two turns with identical text feel
different.

Every mapping is monotone and bounded, so a runaway brain cannot drive the
sampler somewhere degenerate.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Tuple

from ..brain.readout import Telemetry

__all__ = ["SamplingParams", "DEFAULT_SAMPLING", "from_telemetry"]


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.85
    top_p: float = 0.92
    frequency_penalty: float = 0.15
    presence_penalty: float = 0.25
    max_tokens: int = 160
    seed: Optional[int] = None
    stop: Tuple[str, ...] = ()
    #: Non-OpenAI knobs (``min_p``, ``repeat_penalty``) that llama.cpp's server
    #: accepts. Off by default because not every OpenAI-compatible backend
    #: tolerates unknown fields.
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_payload(self, model: str, messages: list, stream: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": round(self.temperature, 4),
            "top_p": round(self.top_p, 4),
            "frequency_penalty": round(self.frequency_penalty, 4),
            "presence_penalty": round(self.presence_penalty, 4),
            "max_tokens": int(self.max_tokens),
            "stream": bool(stream),
        }
        if self.seed is not None:
            payload["seed"] = int(self.seed)
        if self.stop:
            payload["stop"] = list(self.stop)
        payload.update(self.extra)
        return payload

    def describe(self) -> str:
        bits = [
            f"temp {self.temperature:.2f}",
            f"top_p {self.top_p:.2f}",
            f"freq {self.frequency_penalty:.2f}",
            f"pres {self.presence_penalty:.2f}",
            f"max {self.max_tokens}",
        ]
        return "  ".join(bits)


DEFAULT_SAMPLING = SamplingParams()


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def from_telemetry(telemetry: Telemetry, base: Optional[SamplingParams] = None) -> SamplingParams:
    """Derive sampler settings from a brain state."""
    b = base or DEFAULT_SAMPLING
    t = telemetry
    temperature = _clip(b.temperature + 0.55 * (t.agitation - 0.5), 0.40, 1.35)
    top_p = _clip(b.top_p + 0.10 * (t.agitation - 0.5), 0.70, 1.00)
    # Repetition is the tell of a position that is not holding, so a flinching
    # fly gets a heavier anti-repetition penalty -- and so is a scattered one.
    frequency_penalty = _clip(b.frequency_penalty + 0.50 * (1.0 - t.confidence), 0.0, 1.20)
    presence_penalty = _clip(b.presence_penalty + 0.45 * t.deflection, 0.0, 1.20)
    tokens = int(round(b.max_tokens * (0.65 + 0.70 * t.confidence) * (0.85 + 0.30 * t.stamina)))
    max_tokens = int(_clip(tokens, 48, 400))
    return replace(
        b,
        temperature=round(temperature, 4),
        top_p=round(top_p, 4),
        frequency_penalty=round(frequency_penalty, 4),
        presence_penalty=round(presence_penalty, 4),
        max_tokens=max_tokens,
    )
