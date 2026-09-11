"""Leaky integrate-and-fire dynamics over a fixed connectome.

Two decisions make this work across connectomes that differ by three orders of
magnitude in size:

* **Drive is normalised by in-degree.** ``syn_gain`` is mV delivered to a neuron
  whose entire presynaptic pool fires at once, so the same parameters behave the
  same way on a 4,000-neuron surrogate and on the 139,255-neuron FlyWire graph.
* **Gain is calibrated, not guessed.** :meth:`LIFNetwork.calibrate` scales the
  global gain until spontaneous activity lands on a target mean rate, which is
  what keeps the model in the asynchronous-irregular regime instead of
  saturating or going silent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np

from ..connectome.graph import Connectome

__all__ = ["LIFParams", "LIFNetwork", "LIFRun"]

_HISTORY = 1024


@dataclass
class LIFParams:
    """Membrane and synapse constants. Potentials in mV, times in ms."""

    dt_ms: float = 0.5
    tau_ms: float = 20.0
    v_rest: float = -65.0
    v_reset: float = -70.0
    v_thresh: float = -50.0
    v_floor: float = -90.0
    v_ceil: float = 10.0
    refractory_ms: float = 2.0
    #: mV delivered when the whole presynaptic pool is active for one tau.
    syn_gain: float = 120.0
    #: Constant depolarising bias. Sits the median neuron a couple of sigma
    #: below threshold so that the population straddles threshold and firing
    #: rate is graded by each neuron's own excitability. Push it far above
    #: threshold and every neuron becomes identical and the whole network locks
    #: into one global oscillation.
    tonic_mv: float = 11.0
    #: Standard deviation of per-neuron excitability. Homogeneous populations
    #: respond to a rising tonic drive as one all-or-nothing switch; a spread of
    #: thresholds is what makes the rate a smooth, monotonic function of drive.
    bias_sd_mv: float = 3.5
    #: Membrane noise is low-passed by tau/dt ~ 40, so this is a current
    #: perturbation: sigma_v ~ noise*sqrt(dt/(2*tau)).
    noise_mv: float = 24.0
    #: Per-spike increment of the adaptation current, and its ceiling. The
    #: ceiling matters: without it, steady-state adaptation is
    #: increment * rate * tau_adapt, which at any real firing rate is large
    #: enough to shut the whole population down -- and large enough to replace
    #: noise as the thing that sets firing rate, which collapses the population
    #: onto a single global clock.
    adaptation_mv: float = 1.0
    adapt_max_mv: float = 5.0
    tau_adapt_ms: float = 200.0
    tau_rate_ms: float = 60.0


@dataclass
class LIFRun:
    """Result of :meth:`LIFNetwork.run`."""

    pop_trace_hz: np.ndarray
    mean_rate_hz: float
    peak_rate_hz: float
    active_fraction: float
    spikes: int
    ticks: int

    def __repr__(self) -> str:
        return (
            f"<LIFRun mean={self.mean_rate_hz:.2f} Hz peak={self.peak_rate_hz:.1f} Hz "
            f"active={self.active_fraction * 100:.2f}% spikes={self.spikes}>"
        )


class LIFNetwork:
    """Vectorised LIF network bound to a :class:`Connectome`."""

    def __init__(self, connectome: Connectome, params: Optional[LIFParams] = None, seed: int = 0):
        self.conn = connectome
        self.p = params or LIFParams()
        self.rng = np.random.default_rng(seed)
        n = connectome.n
        self.n = n
        self.v = np.full(n, self.p.v_rest, dtype=np.float32)
        self.refractory = np.zeros(n, dtype=np.float32)
        self.adapt = np.zeros(n, dtype=np.float32)
        self.spikes = np.zeros(n, dtype=np.float32)
        self.rate = np.zeros(n, dtype=np.float32)
        self.gain_scale = 1.0
        self.drive_scale = 1.0
        # Fixed per-neuron excitability, drawn once so resets are reproducible.
        self.bias = self.rng.normal(0.0, self.p.bias_sd_mv, n).astype(np.float32) if n else np.zeros(0, np.float32)
        in_syn = connectome.in_synapses()
        self.in_scale = 1.0 / max(float(in_syn.mean()) if n else 1.0, 1.0)
        self._hist = np.zeros(_HISTORY, dtype=np.float32)
        self._hist_i = 0
        self._hist_n = 0
        self.tick = 0

    # ------------------------------------------------------------------ state

    def reset(self, keep_gain: bool = True) -> None:
        self.v.fill(self.p.v_rest)
        self.refractory.fill(0.0)
        self.adapt.fill(0.0)
        self.spikes.fill(0.0)
        self.rate.fill(0.0)
        self._hist.fill(0.0)
        self._hist_i = 0
        self._hist_n = 0
        self.tick = 0
        if not keep_gain:
            self.gain_scale = 1.0

    def pop_trace(self) -> np.ndarray:
        """Population firing rate (Hz) over the recent past, oldest first."""
        if self._hist_n < _HISTORY:
            return self._hist[: self._hist_n].copy()
        return np.roll(self._hist, -self._hist_i)

    def last_population_hz(self) -> float:
        """Population firing rate (Hz) at the most recent step."""
        if self._hist_n == 0:
            return 0.0
        return float(self._hist[(self._hist_i - 1) % _HISTORY])

    # ---------------------------------------------------------------- dynamics

    def step(self, external: Optional[np.ndarray] = None, gain: float = 1.0) -> np.ndarray:
        """Advance one time step. Returns the spike vector."""
        p = self.p
        drive = self.conn.synaptic_drive(self.spikes) * self.in_scale * p.syn_gain * self.gain_scale * gain
        if external is not None:
            drive = drive + external
        noise = self.rng.normal(0.0, p.noise_mv, self.n)
        tonic = p.tonic_mv * self.drive_scale + self.bias
        self.v += (
            -(self.v - p.v_rest) + tonic + drive.astype(np.float32) + noise
        ) * (p.dt_ms / p.tau_ms)
        self.v -= self.adapt
        np.clip(self.v, p.v_floor, p.v_ceil, out=self.v)

        fired = (self.refractory <= 0.0) & (self.v >= p.v_thresh)
        self.v[fired] = p.v_reset
        n_fired = int(fired.sum())
        if n_fired:
            # Jittered refractoriness: without it the population fires in
            # near-lockstep volleys, which makes the readout oscillate with the
            # simulation step instead of with the network.
            self.refractory[fired] = p.refractory_ms * self.rng.uniform(0.7, 1.4, n_fired)
        np.subtract(self.refractory, p.dt_ms, out=self.refractory)
        np.clip(self.refractory, 0.0, None, out=self.refractory)

        self.adapt *= np.exp(-p.dt_ms / p.tau_adapt_ms)
        self.adapt += fired * p.adaptation_mv
        np.clip(self.adapt, 0.0, p.adapt_max_mv, out=self.adapt)

        self.spikes = fired.astype(np.float32)
        alpha = 1.0 - np.exp(-p.dt_ms / p.tau_rate_ms)
        self.rate += alpha * (self.spikes * (1000.0 / p.dt_ms) - self.rate)

        pop_hz = float(self.spikes.mean() * (1000.0 / p.dt_ms)) if self.n else 0.0
        self._hist[self._hist_i] = pop_hz
        self._hist_i = (self._hist_i + 1) % _HISTORY
        self._hist_n = min(self._hist_n + 1, _HISTORY)
        self.tick += 1
        return self.spikes

    def run(
        self,
        ticks: int,
        external: Optional[np.ndarray] = None,
        gain: float = 1.0,
        callback: Optional[Callable[[int, np.ndarray], None]] = None,
    ) -> LIFRun:
        """Run ``ticks`` steps.

        ``external`` may be a single drive vector (held constant) or a
        ``(ticks, n)`` array for time-varying input.
        """
        trace = np.zeros(ticks, dtype=np.float32)
        spike_total = 0.0
        for t in range(ticks):
            ext = external
            if external is not None and external.ndim == 2:
                ext = external[t]
            spikes = self.step(ext, gain=gain)
            spike_total += float(spikes.sum())
            trace[t] = self._hist[(self._hist_i - 1) % _HISTORY]
            if callback is not None:
                callback(t, spikes)
        rate = self.rate
        active = float((rate > 1.0).mean()) if self.n else 0.0
        return LIFRun(
            pop_trace_hz=trace,
            mean_rate_hz=float(rate.mean()) if self.n else 0.0,
            peak_rate_hz=float(rate.max()) if self.n else 0.0,
            active_fraction=active,
            spikes=int(spike_total),
            ticks=ticks,
        )

    # --------------------------------------------------------------- calibration

    @staticmethod
    def _norm_isf(p: float) -> float:
        """Inverse survival function of the standard normal, by bisection.

        Kept local so the package stays numpy-only; scipy would be a heavy
        dependency for one call.
        """
        p = float(np.clip(p, 1e-12, 0.5))
        lo, hi = 0.0, 12.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if 0.5 * math.erfc(mid / math.sqrt(2.0)) > p:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def calibrate(
        self,
        target_hz: float = 4.0,
        ticks: int = 250,
        warmup: int = 150,
        gain: float = 1.0,
    ) -> float:
        """Tune ``drive_scale`` so spontaneous activity matches ``target_hz``.

        Membrane noise is low-passed by the leak, so the membrane potential
        makes roughly one independent draw per tau. That gives a closed-form
        prior -- sit the resting equilibrium ``z`` standard deviations below
        threshold so the Gaussian tail hits the target rate -- but the closed
        form ignores adaptation, and adaptation makes the rate-vs-drive curve
        non-monotone at high drive. So the prior only says where to look, and a
        bracketed scan plus one local refinement does the finding. About a
        dozen short probes; state is restored afterwards so calibration never
        contaminates the first real stimulus.
        """
        snapshot = (
            self.v.copy(),
            self.refractory.copy(),
            self.adapt.copy(),
            self.spikes.copy(),
            self.rate.copy(),
            self._hist.copy(),
            self._hist_i,
            self._hist_n,
        )
        p = self.p
        sigma_v = p.noise_mv * math.sqrt(p.dt_ms / (2.0 * p.tau_ms))
        tau_eff_s = p.tau_ms / 1000.0
        target_prob = float(np.clip(target_hz * tau_eff_s, 1e-6, 0.5))
        z = self._norm_isf(target_prob)
        mu = p.v_thresh - z * sigma_v
        prior = float(np.clip((mu - p.v_rest) / max(p.tonic_mv, 1e-6), 0.02, 6.0))

        def evaluate(scale: float) -> float:
            self.drive_scale = float(np.clip(scale, 0.02, 6.0))
            # Each probe starts from rest: without this, adaptation accumulated
            # by the previous probe contaminates the measured rate.
            self.reset()
            self.run(warmup)
            self.run(ticks, gain=gain)
            return float(self.rate.mean()) if self.n else 0.0

        try:
            # The closed form ignores adaptation, and adaptation turns the
            # rate-vs-drive curve non-monotone at high drive. So it is used as
            # a prior for where to look, and a bracketed scan does the finding.
            multipliers = (0.40, 0.60, 0.80, 1.00, 1.30, 1.70, 2.20, 3.00, 4.00)
            scales = sorted({float(np.clip(prior * m, 0.02, 6.0)) for m in multipliers})
            scored = [(abs(evaluate(s) - target_hz), s) for s in scales]
            _, best = min(scored)
            # One local refinement around the winner.
            for delta in (0.85, 0.93, 1.07, 1.18):
                probe = float(np.clip(best * delta, 0.02, 6.0))
                if abs(evaluate(probe) - target_hz) < abs(evaluate(best) - target_hz):
                    best = probe
            self.drive_scale = best
        finally:
            (
                self.v,
                self.refractory,
                self.adapt,
                self.spikes,
                self.rate,
                self._hist,
                self._hist_i,
                self._hist_n,
            ) = snapshot
            self.tick = 0
        return self.drive_scale

    def with_params(self, **changes: float) -> "LIFNetwork":
        """Return a clone sharing the connectome but with edited parameters."""
        clone = LIFNetwork(self.conn, replace(self.p, **changes), seed=int(self.rng.integers(2**31)))
        clone.gain_scale = self.gain_scale
        clone.drive_scale = self.drive_scale
        return clone
