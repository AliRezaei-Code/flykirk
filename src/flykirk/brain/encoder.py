"""Text as sensory drive.

The fly cannot read, so text never reaches it directly. Instead each token is
hashed onto a small, fixed, sparse set of sensory neurons, weighted by how loud
the token is (capitals, punctuation, digits), and the resulting vector is what
the brain actually receives. Two properties matter:

* **Deterministic.** The hash is ``blake2b``, not Python's randomised ``hash``,
  so the same sentence drives the same neurons in every process and every run.
* **Distributed.** Every token lights up several neurons, so semantically nearby
  input (shared words, shared morphology) produces a correlated pattern of
  activity rather than a single labelled line.
"""

from __future__ import annotations

import hashlib
import re
from typing import List, Optional, Sequence

import numpy as np

__all__ = ["SensoryEncoder"]

_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_STOPWORDS = frozenset(
    """a an and are as at be but by for from has have he her him his i if in is it its me my no not of on or our
    she so that the their them then there these they this to too us was we were what when where which who will with
    you your do does did done just very really gonna wanna okay ok""".split()
)


class SensoryEncoder:
    """Maps text onto spiking drive for a designated set of sensory neurons."""

    def __init__(
        self,
        targets: Optional[Sequence[int] | np.ndarray] = None,
        n_neurons: int = 0,
        neurons_per_token: int = 3,
        active_neurons: int = 96,
        drive_mv: float = 12.0,
        seed: int = 0,
    ) -> None:
        if targets is None:
            targets = np.arange(n_neurons, dtype=np.int64)
        self.targets = np.asarray(targets, dtype=np.int64).ravel()
        if self.targets.size == 0:
            raise ValueError("encoder needs at least one target neuron")
        self.neurons_per_token = max(1, int(neurons_per_token))
        #: How many sensory neurons a stimulus is allowed to recruit. A sparse
        #: code is the whole point: if every stimulus lit up the same fraction
        #: of the sensory sheet at the same amplitude, the brain downstream
        #: could not tell two sentences apart.
        self.active_neurons = max(1, int(active_neurons))
        self.drive_mv = float(drive_mv)
        self.seed = int(seed)
        # Drive vectors are always graph-shaped so they can be added straight to
        # the membrane potential, even when sensory neurons are a small block.
        self.size = int(max(n_neurons, int(self.targets.max()) + 1))

    # ------------------------------------------------------------------ hashing

    def _bucket(self, token: str, salt: int) -> int:
        digest = hashlib.blake2b(f"{self.seed}:{salt}:{token}".encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big")

    # ------------------------------------------------------------------ features

    @staticmethod
    def content_words(text: str, limit: int = 8) -> List[str]:
        """Longest non-stopword tokens, most distinctive first. Used by the
        offline persona writer so it has something concrete to talk about."""
        words = [w.lower() for w in _WORD.findall(text)]
        keep = [w for w in words if w not in _STOPWORDS and len(w) > 2]
        keep.sort(key=len, reverse=True)
        seen: List[str] = []
        for w in keep:
            if w not in seen:
                seen.append(w)
            if len(seen) >= limit:
                break
        return seen

    @staticmethod
    def salience(text: str) -> float:
        """0..1 loudness of a string: caps, exclamation, questions, digits."""
        if not text:
            return 0.0
        letters = sum(1 for c in text if c.isalpha())
        caps = sum(1 for c in text if c.isupper())
        caps_ratio = caps / letters if letters else 0.0
        bangs = text.count("!")
        asks = text.count("?")
        digits = sum(1 for c in text if c.isdigit())
        raw = (
            1.4 * caps_ratio
            + 0.18 * min(bangs, 5)
            + 0.10 * min(asks, 4)
            + 0.05 * min(digits, 6)
            + 0.15 * min(len(text) / 200.0, 1.0)
        )
        return float(np.clip(raw, 0.0, 1.0))

    def _features(self, text: str) -> List[tuple]:
        words = [w.lower() for w in _WORD.findall(text)]
        flat = re.sub(r"[^a-z0-9]", "", text.lower())
        features: List[tuple] = [(w, 1.0) for w in words]
        features.extend((f"{a}_{b}", 1.4) for a, b in zip(words, words[1:]))
        features.extend((flat[i : i + 4], 0.5) for i in range(max(0, len(flat) - 3)))
        return features

    # ------------------------------------------------------------------ encoding

    def encode(self, text: str) -> np.ndarray:
        """Return a graph-shaped drive vector in mV.

        Tokens accumulate evidence on sensory neurons; only the top
        ``active_neurons`` survive, scaled so the strongest sits at
        ``drive_mv`` and multiplied by the text's loudness. Two different
        sentences therefore recruit two different populations of receptors --
        which is the only reason the downstream brain can respond differently
        to them.
        """
        raw = np.zeros(self.size, dtype=np.float32)
        if not text:
            return raw
        emphasis = float(np.clip(0.7 + 0.9 * self.salience(text), 0.7, 1.6))
        for token, weight in self._features(text):
            for salt in range(self.neurons_per_token):
                bucket = self._bucket(token, salt)
                idx = int(bucket % self.targets.size)
                raw[self.targets[idx]] += weight

        touched = int((raw > 0).sum())
        if touched == 0:
            return raw
        k = min(self.active_neurons, touched)
        top = np.argpartition(raw, -k)[-k:] if k < raw.shape[0] else np.flatnonzero(raw > 0)
        values = raw[top]
        peak = float(values.max())
        drive = np.zeros_like(raw)
        if peak > 0:
            drive[top] = values / peak * self.drive_mv * emphasis
        return drive

    def pulse(self, text: str, ticks: int, tau_ms: float = 110.0, dt_ms: float = 0.5) -> np.ndarray:
        """A decaying ``(ticks, size)`` drive: the stimulus is an event, not a
        permanent condition."""
        base = self.encode(text)
        t = np.arange(ticks, dtype=np.float32) * dt_ms
        envelope = np.exp(-t / max(tau_ms, 1e-3)).astype(np.float32)
        return envelope[:, None] * base[None, :]
