"""Descending-neuron population -> a handful of numbers the persona can use.

There is no homunculus in the fly, so the readout is deliberately mechanical.
What the language model gets is not "a thought" but population statistics
measured off the descending neurons -- the last stage of the brain before the
ventral nerve cord, and the closest thing here to a mouth.

The measurement is a **deviation from the fly's own idle state**, not an
absolute level. Absolute levels are dominated by whatever global arousal the
network happens to be in, which is the same for every input; the deviation is
what carries the content. So the readout learns an idle baseline while nothing
is happening and reports how a stimulus moved the pool away from it:

``agitation``   mean rectified recruitment of the pool, in Hz above idle
``dominance``   excitatory versus inhibitory recruitment -- attack or retreat
``deflection``  entropy of the recruitment pattern: a diffuse pool has no line
``confidence``  how sustained the recruitment was, peak versus average
``stamina``     how much adaptation has accumulated
``syllables_per_sec``  dominant frequency of the descending pool's population
                rate. This is the joke that writes itself: the fly's own
                oscillation sets how fast it talks.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..connectome.graph import Connectome

__all__ = ["Telemetry", "Readout"]

_TRACE = 1024


@dataclass
class Telemetry:
    tick: int = 0
    population_rate_hz: float = 0.0
    mean_rate_hz: float = 0.0
    peak_rate_hz: float = 0.0
    active_fraction: float = 0.0
    descending_rate_hz: float = 0.0
    recruitment_hz: float = 0.0
    agitation: float = 0.0
    confidence: float = 0.0
    dominance: float = 0.5
    deflection: float = 0.0
    stamina: float = 1.0
    syllables_per_sec: float = 5.0
    neuromod: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def aggregate(cls, frames: "List[Telemetry]") -> "Telemetry":
        """Collapse a stimulus window into one reaction reading.

        Peaks where the peak is the point (how hard the fly came out swinging),
        means where the mean is the point. ``confidence`` is the ratio of
        average to peak recruitment: a response that holds up over the window is
        a conviction, one that spikes and dies is a flinch.
        """
        if not frames:
            return cls()
        peak_frame = max(frames, key=lambda f: f.agitation)
        last = frames[-1]
        mean_agitation = float(np.mean([f.agitation for f in frames]))
        peak_agitation = peak_frame.agitation
        confidence = mean_agitation / peak_agitation if peak_agitation > 1e-6 else 0.0
        return cls(
            tick=last.tick,
            population_rate_hz=float(np.mean([f.population_rate_hz for f in frames])),
            mean_rate_hz=float(np.mean([f.mean_rate_hz for f in frames])),
            peak_rate_hz=max(f.peak_rate_hz for f in frames),
            active_fraction=float(np.mean([f.active_fraction for f in frames])),
            descending_rate_hz=max(f.descending_rate_hz for f in frames),
            recruitment_hz=float(np.mean([f.recruitment_hz for f in frames])),
            agitation=peak_agitation,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            dominance=float(np.mean([f.dominance for f in frames])),
            deflection=float(np.mean([f.deflection for f in frames])),
            stamina=last.stamina,
            syllables_per_sec=float(np.mean([f.syllables_per_sec for f in frames])),
            neuromod=dict(last.neuromod),
        )

    def render(self, width: int = 22) -> str:
        """Compact ASCII readout, the thing you actually watch while debugging."""

        def bar(value: float) -> str:
            filled = int(round(max(0.0, min(1.0, value)) * width))
            return "#" * filled + "." * (width - filled)

        return "\n".join(
            [
                f"  descending pool {self.descending_rate_hz:7.2f} Hz"
                f"   recruitment {self.recruitment_hz:6.2f} Hz"
                f"   population {self.population_rate_hz:7.2f} Hz",
                f"  agitation   {bar(self.agitation)} {self.agitation:4.2f}",
                f"  confidence  {bar(self.confidence)} {self.confidence:4.2f}",
                f"  dominance   {bar(self.dominance)} {self.dominance:4.2f}",
                f"  deflection  {bar(self.deflection)} {self.deflection:4.2f}",
                f"  stamina     {bar(self.stamina)} {self.stamina:4.2f}",
                f"  speech      {self.syllables_per_sec:4.2f} syllables/s"
                f"   OA {self.neuromod.get('octopamine', 0):.2f}"
                f"  DA {self.neuromod.get('dopamine', 0):.2f}"
                f"  5HT {self.neuromod.get('serotonin', 0):.2f}",
            ]
        )


class Readout:
    """Measures population statistics off a network's current state."""

    #: Recruitment (Hz per descending neuron, above idle) at which the pool
    #: counts as half-agitated. This is a measurement gain, and it has to match
    #: the scale the dynamics actually produce: a fly brain's descending pool
    #: moves by a few tenths of a hertz between stimuli, not by tens.
    HALF_RECRUITED_HZ = 0.25

    def __init__(
        self,
        connectome: Connectome,
        dt_ms: float = 0.5,
        speech_super_class: str = "descending",
        speech_cell_class: Optional[str] = None,
        idle_tau_ms: float = 400.0,
    ) -> None:
        self.conn = connectome
        self.dt_ms = dt_ms
        if speech_cell_class:
            mask = connectome.mask(cell_class=speech_cell_class)
        else:
            mask = connectome.mask(super_class=speech_super_class)
        if not mask.any():
            # Fall back to the highest-out-degree neurons: whatever is talking,
            # it is certainly shouting at something downstream.
            out = connectome.out_synapses()
            cutoff = float(np.quantile(out, 0.99)) if out.size else 0.0
            mask = out >= cutoff
            if not mask.any():
                mask = np.ones(connectome.n, dtype=bool)
        self.speech_mask = mask
        self.n_speech = int(mask.sum())

        weights = connectome.weights
        pre_sign = connectome.signed_weights
        rows = connectome.row_of_nnz
        self._pos, self._neg = pre_sign > 0, pre_sign < 0
        self._rows = rows
        self._targets = connectome.indices
        self._weights = weights

        self.idle_rate = np.zeros(connectome.n, dtype=np.float32)
        self._idle_alpha = 1.0 - float(np.exp(-dt_ms / max(idle_tau_ms, 1e-6)))
        self._idle_ready = False
        self._dn_trace = np.zeros(_TRACE, dtype=np.float32)
        self._dn_i = 0
        self._dn_n = 0

    # ------------------------------------------------------------------ maths

    def _balance(self, rate_hz: np.ndarray) -> tuple:
        """Excitatory and inhibitory drive arriving at the descending pool."""
        if self._rows.size:
            exc_edge = self._weights[self._pos] * rate_hz[self._targets[self._pos]]
            inh_edge = self._weights[self._neg] * rate_hz[self._targets[self._neg]]
            exc = np.bincount(self._rows[self._pos], weights=exc_edge, minlength=self.conn.n)
            inh = np.bincount(self._rows[self._neg], weights=inh_edge, minlength=self.conn.n)
        else:
            exc = np.zeros(self.conn.n)
            inh = np.zeros(self.conn.n)
        return float(exc[self.speech_mask].sum()), float(inh[self.speech_mask].sum())

    def _speaking_rate(self, agitation: float) -> float:
        """Dominant oscillation of the descending pool -> syllables per second.

        Band-limited to 2-120 Hz because anything outside that is either drift
        or the simulation step. Mapped linearly onto 4.0-7.8 syl/s, the range of
        a fast talker, with a nudge from agitation so the corpus stays lively.
        """
        if self._dn_n < 64:
            return 5.0 + 1.5 * agitation
        trace = self._dn_trace[: self._dn_n] if self._dn_n < _TRACE else np.roll(self._dn_trace, -self._dn_i)
        x = trace.astype(np.float64) - float(trace.mean())
        if not np.any(x):
            return 5.0 + 1.5 * agitation
        spectrum = np.abs(np.fft.rfft(x * np.hanning(x.size)))
        freqs = np.fft.rfftfreq(x.size, d=self.dt_ms / 1000.0)
        band = (freqs >= 2.0) & (freqs <= 120.0)
        if not band.any():
            return 5.0 + 1.5 * agitation
        peak = float(freqs[band][int(np.argmax(spectrum[band]))])
        # Log-scaled so the 2-120 Hz band spreads across the output range
        # instead of clustering at the bottom of it.
        shaped = math.log1p(peak) / math.log1p(60.0)
        return float(np.clip(3.8 + 3.2 * shaped + 0.8 * agitation, 3.6, 8.2))

    # ------------------------------------------------------------------ update

    def measure(
        self,
        rate_hz: np.ndarray,
        adapt: np.ndarray,
        neuromod_state: Any,
        pop_trace_hz: Optional[np.ndarray] = None,
        tick: int = 0,
        update_idle: bool = False,
    ) -> Telemetry:
        rate = np.asarray(rate_hz, dtype=np.float64)
        speech = rate[self.speech_mask]
        dn_mean = float(speech.mean()) if speech.size else 0.0

        # Idle baseline: only the quiet path is allowed to move it, so a big
        # stimulus cannot drag the reference along with it.
        if update_idle:
            self.idle_rate += self._idle_alpha * (rate - self.idle_rate)
            self._idle_ready = True
        baseline = self.idle_rate[self.speech_mask].astype(np.float64) if self._idle_ready else np.zeros_like(speech)
        delta = speech - baseline

        pos = float(np.clip(delta, 0.0, None).sum())
        neg = float(np.clip(-delta, 0.0, None).sum())
        recruitment = pos / max(self.n_speech, 1)
        agitation = float(recruitment / (recruitment + self.HALF_RECRUITED_HZ))
        if pos + neg > 1e-9:
            dominance = float((pos - neg) / (pos + neg + 1e-9) + 1.0) / 2.0
        else:
            dominance = 0.5

        magnitude = np.abs(delta)
        total = float(magnitude.sum())
        if magnitude.size > 1 and total > 1e-9:
            p = magnitude / total
            entropy = float(-(p * np.log(p + 1e-12)).sum() / np.log(magnitude.size))
        else:
            entropy = 1.0
        deflection = float(np.clip(entropy, 0.0, 1.0))

        if adapt.size:
            stamina = float(np.clip(1.0 - float(adapt.mean()) / 12.0, 0.0, 1.0))
        else:
            stamina = 1.0
        if hasattr(neuromod_state, "stamina"):
            stamina = float(np.clip(0.6 * stamina + 0.4 * neuromod_state.stamina(), 0.0, 1.0))

        self._dn_trace[self._dn_i] = dn_mean
        self._dn_i = (self._dn_i + 1) % _TRACE
        self._dn_n = min(self._dn_n + 1, _TRACE)
        syllables = self._speaking_rate(agitation)

        trace = np.asarray(pop_trace_hz if pop_trace_hz is not None else np.zeros(0), dtype=np.float64)
        neuromod = neuromod_state.as_dict() if hasattr(neuromod_state, "as_dict") else {}
        return Telemetry(
            tick=tick,
            population_rate_hz=float(trace[-1]) if trace.size else 0.0,
            mean_rate_hz=float(rate.mean()) if rate.size else 0.0,
            peak_rate_hz=float(rate.max()) if rate.size else 0.0,
            active_fraction=float((rate > 1.0).mean()) if rate.size else 0.0,
            descending_rate_hz=dn_mean,
            recruitment_hz=recruitment,
            agitation=agitation,
            # Per-frame confidence is the instantaneous sustain proxy; the window
            # aggregate replaces it with mean-over-peak.
            confidence=agitation,
            dominance=dominance,
            deflection=deflection,
            stamina=stamina,
            syllables_per_sec=syllables,
            neuromod=neuromod,
        )

    def describe(self) -> str:
        return (
            f"readout: {self.n_speech:,} descending neurons "
            f"({self.n_speech / max(self.conn.n, 1) * 100:.2f}% of the graph), "
            f"idle baseline {'ready' if self._idle_ready else 'not yet learned'}"
        )
