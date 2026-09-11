"""Neuromodulator state: the fly's mood, and it really does drift.

Three slow variables ride on top of the fast spiking dynamics:

``octopamine``  arousal and aggression; rises with activity and novelty.
``dopamine``    salience and reward prediction error; rises on surprise.
``serotonin``   satiation and patience; integrates activity over seconds.

Octopamine raises synaptic gain, serotonin lowers it. That single loop is why a
fly that has been arguing for eight rounds gets slower and calmer rather than
louder forever, and it is the reason ``--rounds`` changes the character of a
debate rather than just its length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

__all__ = ["NeuromodParams", "NeuromodState", "NeuromodulatorSystem"]


@dataclass
class NeuromodParams:
    baseline: float = 0.35
    tau_octopamine_ms: float = 900.0
    tau_dopamine_ms: float = 400.0
    tau_serotonin_ms: float = 6000.0
    tau_expectation_ms: float = 1500.0
    octopamine_rate_weight: float = 0.55
    octopamine_novelty_weight: float = 0.30
    dopamine_error_weight: float = 0.40
    dopamine_salience_weight: float = 0.25
    serotonin_sustain_weight: float = 0.45

    def __post_init__(self) -> None:
        if self.baseline <= 0.0 or self.baseline >= 1.0:
            raise ValueError("baseline must be in (0, 1)")


@dataclass
class NeuromodState:
    octopamine: float
    dopamine: float
    serotonin: float
    expected_rate_hz: float = 0.0
    prediction_error: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "octopamine": round(self.octopamine, 4),
            "dopamine": round(self.dopamine, 4),
            "serotonin": round(self.serotonin, 4),
            "prediction_error": round(self.prediction_error, 4),
        }

    def synaptic_gain(self, baseline: float = 0.35) -> float:
        """Multiplier on synaptic drive: aroused flies shout, sated flies mumble."""
        return float(
            np.clip(
                0.55
                + 1.35 * (self.octopamine - baseline)
                - 0.55 * (self.serotonin - baseline)
                + 0.25 * (self.dopamine - baseline),
                0.25,
                2.50,
            )
        )

    def aggression(self, baseline: float = 0.35) -> float:
        """0..1 scalar used by the persona layer to pick a register."""
        return float(
            np.clip(
                0.5
                + 1.20 * (self.octopamine - baseline)
                + 0.90 * (self.dopamine - baseline)
                - 0.80 * (self.serotonin - baseline),
                0.0,
                1.0,
            )
        )

    def stamina(self, baseline: float = 0.35) -> float:
        return float(np.clip(1.0 - 1.6 * max(0.0, self.serotonin - baseline), 0.0, 1.0))


class NeuromodulatorSystem:
    """Integrates spiking activity into slow modulatory concentrations."""

    def __init__(self, params: NeuromodParams | None = None, dt_ms: float = 0.5):
        self.p = params or NeuromodParams()
        self.dt_ms = dt_ms
        base = self.p.baseline
        self.state = NeuromodState(octopamine=base, dopamine=base, serotonin=base)
        self._expected = 0.0
        self._sustain = 0.0

    def reset(self) -> None:
        base = self.p.baseline
        self.state = NeuromodState(octopamine=base, dopamine=base, serotonin=base)
        self._expected = 0.0
        self._sustain = 0.0

    @staticmethod
    def _saturate(rate_hz: float, half: float = 20.0) -> float:
        return rate_hz / (rate_hz + half) if rate_hz > 0 else 0.0

    def step(
        self,
        population_rate_hz: float,
        novelty: float = 0.0,
        salience: float = 0.0,
        ticks: int = 1,
    ) -> NeuromodState:
        p = self.p
        dt = self.dt_ms * max(1, ticks) / 1000.0
        r = self._saturate(population_rate_hz)

        alpha_exp = 1.0 - np.exp(-self.dt_ms * max(1, ticks) / p.tau_expectation_ms)
        self._expected += alpha_exp * (population_rate_hz - self._expected)
        error = population_rate_hz - self._expected
        norm_error = self._saturate(abs(error), half=8.0)

        alpha_sus = 1.0 - np.exp(-self.dt_ms * max(1, ticks) / p.tau_serotonin_ms)
        self._sustain += alpha_sus * (r - self._sustain)

        target_oa = p.baseline + p.octopamine_rate_weight * r + p.octopamine_novelty_weight * float(novelty)
        target_da = p.baseline + p.dopamine_error_weight * norm_error + p.dopamine_salience_weight * float(salience)
        target_5ht = p.baseline + p.serotonin_sustain_weight * self._sustain

        s = self.state
        s.octopamine += (target_oa - s.octopamine) * dt / (p.tau_octopamine_ms / 1000.0)
        s.dopamine += (target_da - s.dopamine) * dt / (p.tau_dopamine_ms / 1000.0)
        s.serotonin += (target_5ht - s.serotonin) * dt / (p.tau_serotonin_ms / 1000.0)
        s.octopamine = float(np.clip(s.octopamine, 0.0, 1.0))
        s.dopamine = float(np.clip(s.dopamine, 0.0, 1.0))
        s.serotonin = float(np.clip(s.serotonin, 0.0, 1.0))
        s.expected_rate_hz = float(self._expected)
        s.prediction_error = float(error)
        return s
