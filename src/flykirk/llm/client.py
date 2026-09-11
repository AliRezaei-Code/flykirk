"""Language-model clients: a local OpenAI-compatible backend, and a scripted one.

No dependency on an HTTP library beyond the standard library, so installing
flykirk needs nothing but numpy. Point :class:`OpenAICompatClient` at any
OpenAI-compatible endpoint -- Ollama, llama.cpp's ``llama serve``, LM Studio:

    ollama serve                                   # http://localhost:11434/v1
    llama serve -hf LiquidAI/LFM2-700M-GGUF:Q4_K_M  # http://localhost:8080/v1

:class:`ScriptedClient` needs no model at all. It composes a turn out of the
register's own moves and the brain state, which makes ``--offline`` runs, tests
and CI deterministic.
"""

from __future__ import annotations

import json
import random
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence

from ..brain.readout import Telemetry
from ..persona.kirk import PersonaStyle, register_for
from .sampling import SamplingParams

__all__ = [
    "LLMError",
    "TurnContext",
    "ChatClient",
    "OpenAICompatClient",
    "ScriptedClient",
    "make_client",
]


class LLMError(RuntimeError):
    pass


@dataclass
class TurnContext:
    """Everything a client might need beyond the prompt itself."""

    topic: str
    side: str = "for"
    opponent_text: str = ""
    telemetry: Optional[Telemetry] = None
    turn: int = 1
    total_turns: int = 1
    opponent: str = ""


class ChatClient:
    name = "client"

    def chat(self, messages: List[Dict[str, str]], params: SamplingParams, context: Optional[TurnContext] = None) -> str:
        raise NotImplementedError

    def health(self) -> List[str]:
        return []


# --------------------------------------------------------------------- HTTP


class OpenAICompatClient(ChatClient):
    """Minimal OpenAI-compatible chat client."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "hf.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF:Q4_K_M",
        api_key: Optional[str] = None,
        timeout: float = 180.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = float(timeout)
        self.name = f"openai-compat:{self.model}"

    # ------------------------------------------------------------------ utils

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", "User-Agent": "flykirk/0.1"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request(self, path: str, payload: Optional[Dict[str, Any]] = None, method: str = "POST"):
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(url, data=data, headers=self._headers(), method=method)
        try:
            return urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:400]
            raise LLMError(f"{method} {url} -> HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(
                f"cannot reach {url} ({exc.reason}). Start a local server, e.g. "
                f"`ollama serve` or `llama serve -hf LiquidAI/LFM2-700M-GGUF:Q4_K_M`, "
                f"or run with --offline."
            ) from exc

    # ------------------------------------------------------------------- api

    def health(self) -> List[str]:
        try:
            with self._request("/models", method="GET") as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - health checks report, never raise
            return [f"unreachable: {exc}"]
        entries = payload.get("data") or payload.get("models") or []
        names = []
        for entry in entries:
            if isinstance(entry, dict):
                names.append(str(entry.get("id") or entry.get("name") or "?"))
        return names

    def chat(self, messages: List[Dict[str, str]], params: SamplingParams, context: Optional[TurnContext] = None) -> str:
        payload = params.as_payload(self.model, messages, stream=False)
        with self._request("/chat/completions", payload) as response:
            body = json.loads(response.read().decode("utf-8"))
        try:
            return str(body["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response shape: {str(body)[:400]}") from exc

    def stream(
        self, messages: List[Dict[str, str]], params: SamplingParams, context: Optional[TurnContext] = None
    ) -> Iterator[str]:
        payload = params.as_payload(self.model, messages, stream=True)
        with self._request("/chat/completions", payload) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    delta = json.loads(chunk)["choices"][0].get("delta", {})
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                piece = delta.get("content")
                if piece:
                    yield piece


# ------------------------------------------------------------------ scripted


@dataclass
class ScriptedClient(ChatClient):
    """Deterministic stand-in for a language model.

    Assembles a turn from the register's openers, transitions, rhetoricals and
    closers plus a claim about the motion's subject. The brain state drives the
    choices: an agitated fly drops its transitions and shortens its sentences, a
    deflected fly pivots to an unrelated claim, a tired fly cuts the closer.
    """

    style: PersonaStyle = field(default_factory=register_for)
    seed: int = 0
    name: str = "scripted"

    _SUBJECTS: Sequence[str] = (
        "the {subject}",
        "everything about the {subject}",
        "the entire case for the {subject}",
        "the people who defend the {subject}",
        "the so-called experts on the {subject}",
    )
    _CLAIMS: Sequence[str] = (
        "The {subject} is the single most overrated thing in this entire conversation.",
        "Nobody under thirty even thinks about the {subject}, and that tells you everything.",
        "Name one person who has actually looked at the {subject} and come away impressed. One.",
        "The {subject} only survives because everyone agreed to stop asking questions about it.",
        "The {subject} is a settled question and the other side knows it.",
        "Every argument for the {subject} falls apart the second you say it out loud.",
        "I have talked to a thousand people about the {subject} and not one could defend it.",
        "The {subject} is not a debate, it is a vibe, and the vibe is wrong.",
    )
    _PIVOTS: Sequence[str] = (
        "And that is before we even talk about the {subject},",
        "But forget all that -- what about the {subject}?",
        "Which brings me to the thing nobody will say about the {subject}:",
    )

    def chat(self, messages: List[Dict[str, str]], params: SamplingParams, context: Optional[TurnContext] = None) -> str:
        ctx = context or TurnContext(topic="Resolved: the banana is the superior fruit.")
        rng = random.Random((self.seed + 1) * 100003 + ctx.turn)
        subject = _subject_of(ctx.topic)

        def pick(items: Sequence[str]) -> str:
            return items[rng.randrange(len(items))]

        def fill(template: str) -> str:
            return template.format(subject=subject)

        telemetry: Telemetry = ctx.telemetry or Telemetry()
        # Draw claims without replacement: the register stacks distinct claims,
        # and repeating one within a turn reads as a bug rather than a tic.
        claim_pool = list(self._CLAIMS)
        rng.shuffle(claim_pool)

        def next_claim() -> str:
            return fill(claim_pool.pop()) if claim_pool else fill(pick(self._CLAIMS))

        pieces: List[str] = [fill(pick(self.style.openers))]

        if ctx.opponent_text:
            pieces.append(fill(pick(self.style.address_forms)).capitalize() + ", " + _needle(ctx.opponent_text))

        pieces.append(next_claim())
        if telemetry.agitation < 0.75:
            pieces.append(fill(pick(self.style.transitions)))
        pieces.append(next_claim())
        if telemetry.deflection >= 0.88:
            pieces.append(fill(pick(self._PIVOTS)))
            pieces.append(next_claim())
        else:
            pieces.append(fill(pick(self.style.rhetoricals)))
        if telemetry.stamina > 0.6 and telemetry.confidence > 0.5:
            pieces.append(pick(self.style.closers))

        text = " ".join(pieces)
        return _trim_words(text, self.style.max_words)

    def health(self) -> List[str]:
        return ["scripted (no model required)"]


def _subject_of(topic: str) -> str:
    text = topic.strip()
    for prefix in ("Resolved:", "resolved:", "RESOLVED:"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    for lead in ("the ", "The "):
        if text.startswith(lead):
            text = text[len(lead) :]
    text = text.rstrip(".")
    # The motion's predicate is not the subject; take the first noun-ish chunk.
    for separator in (" is ", " are ", " was ", " were "):
        if separator in text:
            text = text.split(separator)[0]
            break
    for tail in (" at best", " and you know it"):
        text = text.replace(tail, "")
    return text.strip() or "whole thing"


def _needle(opponent_text: str) -> str:
    """A short jabbing reference to whatever the opponent just said."""
    from ..brain.encoder import SensoryEncoder

    words = SensoryEncoder.content_words(opponent_text, limit=1)
    if not words:
        return "that is not an argument."
    return f"nobody is buying the {words[0]} thing."


def _trim_words(text: str, limit: int) -> str:
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit]).rstrip(",;:") + "."


def make_client(
    offline: bool = False,
    seed: int = 0,
    style: Optional[PersonaStyle] = None,
    **kwargs: Any,
) -> ChatClient:
    """Build the client the CLI asked for."""
    if offline:
        return ScriptedClient(style=style or register_for(), seed=seed)
    return OpenAICompatClient(**kwargs)
