"""The assembled brain: connectome plus dynamics plus a readout.

``BrainSim.receive(text)`` is the whole loop in one call -- text becomes sensory
drive, drive propagates through the graph, neuromodulators integrate the result,
and the descending pool is measured into a :class:`Telemetry`. That object is
the *only* thing the persona layer is allowed to see.

The brain also spends time doing nothing, and that matters: :meth:`settle` lets
the readout learn the fly's idle firing pattern, so :meth:`receive` can report
how a stimulus moved the brain *away from its own baseline* rather than
reporting a global arousal level that is identical for every input.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..connectome.graph import Connectome
from .encoder import SensoryEncoder
from .lif import LIFNetwork, LIFParams
from .neuromod import NeuromodParams, NeuromodulatorSystem
from .readout import Readout, Telemetry

__all__ = ["BrainConfig", "BrainSim", "TickSample"]


def _select_monitored(connectome: Connectome, count: int) -> np.ndarray:
    """Pick the neurons the live raster displays.

    Stratified across sensory, central and descending pools so the raster shows
    a signal entering the brain and leaving it, rather than 96 random cells.
    Deterministic, so the same raster channels mean the same thing every run.
    """
    if connectome.n == 0:
        return np.zeros(0, dtype=np.int64)
    count = max(1, min(int(count), connectome.n))
    groups = [
        connectome.mask(super_class="sensory"),
        connectome.mask(super_class="central"),
        connectome.mask(super_class="descending"),
    ]
    rng = np.random.default_rng(0)
    per_group = max(1, count // len(groups))
    picked: List[np.ndarray] = []
    for mask in groups:
        idx = np.flatnonzero(mask)
        if idx.size:
            take = min(per_group, idx.size)
            picked.append(rng.choice(idx, size=take, replace=False))
    if not picked:
        return np.arange(count, dtype=np.int64)
    out = np.unique(np.concatenate(picked))
    if out.shape[0] > count:
        out = out[:count]
    return out.astype(np.int64)


@dataclass
class TickSample:
    """One instrumented read of the running network, for live display.

    This is deliberately raw: which monitored neurons fired, plus the slow
    quantities. Anything the UI derives (rates, percentages) it derives from
    the same numbers the persona layer sees.
    """

    tick: int
    sim_ms: float
    population_hz: float
    descending_hz: float
    spikes: int
    active_fraction: float
    neuromod: Dict[str, float]
    #: Indices into ``monitored`` that spiked on this step.
    fired: np.ndarray

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "sim_ms": round(self.sim_ms, 3),
            "population_hz": round(self.population_hz, 3),
            "descending_hz": round(self.descending_hz, 3),
            "spikes": self.spikes,
            "active_fraction": round(self.active_fraction, 5),
            "neuromod": {k: round(v, 4) for k, v in self.neuromod.items()},
            "fired": self.fired.tolist(),
        }


@dataclass
class BrainConfig:
    """Knobs for building a brain. All defaults chosen to run on a laptop CPU."""

    neurons: int = 4000
    seed: int = 7
    dt_ms: float = 0.5
    max_edges: int = 2_000_000
    target_rate_hz: float = 2.5
    #: Off by default: the analytical defaults already sit the population on
    #: its threshold, and calibration targets a mean rate, which is the wrong
    #: objective -- a high mean rate means every neuron is suprathreshold and
    #: identical. Turn it on to retune for an unusual connectome.
    calibrate: bool = False
    stimulus_ticks: int = 260
    settle_ticks: int = 200
    warmup_ticks: int = 400
    encoder_gain_mv: float = 12.0
    active_sensory_neurons: int = 96
    #: exponential decay of the stimulus pulse, in ms
    stimulus_tau_ms: float = 110.0
    #: Neurons whose spikes are streamed to the live UI. The raster is a fixed
    #: stratified sample rather than every neuron: 139k channels is neither
    #: renderable nor honest about what a reader can follow.
    monitor_count: int = 96
    #: Emit one sample every this many simulation steps. At dt=0.5 ms this is
    #: one sample per 4 ms of brain time.
    sample_every: int = 8

    def replace(self, **changes: Any) -> "BrainConfig":
        return replace(self, **changes)


class BrainSim:
    """A single fly's brain, ready to be offended."""

    def __init__(self, connectome: Connectome, config: Optional[BrainConfig] = None, seed: Optional[int] = None) -> None:
        self.config = config or BrainConfig()
        self.seed = self.config.seed if seed is None else int(seed)
        self.connectome = connectome

        params = LIFParams(dt_ms=self.config.dt_ms)
        self.net = LIFNetwork(connectome, params, seed=self.seed)
        self.neuromod = NeuromodulatorSystem(NeuromodParams(), dt_ms=self.config.dt_ms)

        sensory = connectome.mask(super_class="sensory")
        if not sensory.any():
            # Fall back to the most heavily targeted neurons so a graph with no
            # sensory annotation still has somewhere for text to land.
            in_syn = connectome.in_synapses()
            sensory = in_syn >= (np.quantile(in_syn, 0.90) if in_syn.size else 0.0)
        targets = np.flatnonzero(sensory)
        self.encoder = SensoryEncoder(
            targets=targets,
            n_neurons=connectome.n,
            active_neurons=self.config.active_sensory_neurons,
            drive_mv=self.config.encoder_gain_mv,
            seed=self.seed,
        )
        self.readout = Readout(connectome, dt_ms=self.config.dt_ms)
        self.monitored = _select_monitored(connectome, self.config.monitor_count)
        self.history: List[Telemetry] = []
        if self.config.calibrate and connectome.n:
            self.net.calibrate(target_hz=self.config.target_rate_hz)
        if connectome.n and self.config.warmup_ticks:
            # Establish the idle baseline before anyone says anything.
            self._advance(self.config.warmup_ticks, collect=True, update_idle=True)
            self.history.clear()

    # -------------------------------------------------------------- lifecycle

    def reset(self) -> None:
        """Clear the fast dynamics. Mood and the learned idle baseline persist:
        a fly that has calmed down has not forgotten what calm looks like."""
        self.net.reset()
        self.neuromod.reset()
        self.history.clear()

    def describe(self) -> str:
        return "\n".join(
            [
                self.connectome.describe(),
                "",
                self.readout.describe(),
                f"dynamics     dt={self.config.dt_ms} ms  drive_scale={self.net.drive_scale:.3f}  "
                f"syn_gain={self.net.p.syn_gain:.0f} mV  target={self.config.target_rate_hz} Hz",
                f"sensory      {self.encoder.targets.size:,} input neurons, "
                f"top {self.encoder.active_neurons} recruited per stimulus",
            ]
        )

    # ---------------------------------------------------------------- dynamics

    def step(self, external: Optional[np.ndarray] = None) -> np.ndarray:
        gain = self.neuromod.state.synaptic_gain(self.neuromod.p.baseline)
        return self.net.step(external, gain=gain)

    def _advance(
        self,
        ticks: int,
        external: Optional[np.ndarray] = None,
        novelty: float = 0.0,
        salience: float = 0.0,
        collect: bool = False,
        update_idle: bool = False,
        on_tick: Optional[Callable[[TickSample], None]] = None,
    ) -> List[Telemetry]:
        pulse = external is not None and external.ndim == 2
        frames: List[Telemetry] = []
        every = max(1, self.config.sample_every)
        monitored_fired = np.zeros(self.monitored.shape[0], dtype=bool)
        accumulated = 0
        for t in range(ticks):
            ext = external[t] if pulse else external
            spikes = self.step(ext)
            accumulated += int(spikes.sum())
            if on_tick is not None:
                monitored_fired |= spikes[self.monitored] > 0
                if t % every == every - 1:
                    on_tick(
                        TickSample(
                            tick=self.net.tick,
                            sim_ms=self.net.tick * self.config.dt_ms,
                            population_hz=self.net.last_population_hz(),
                            descending_hz=float(self.net.rate[self.readout.speech_mask].mean())
                            if self.readout.n_speech
                            else 0.0,
                            spikes=accumulated,
                            active_fraction=float((self.net.rate > 1.0).mean()) if self.net.n else 0.0,
                            neuromod=self.neuromod.state.as_dict(),
                            fired=np.flatnonzero(monitored_fired),
                        )
                    )
                    monitored_fired = np.zeros(self.monitored.shape[0], dtype=bool)
                    accumulated = 0
            decay = float(np.exp(-t * self.config.dt_ms / max(self.config.stimulus_tau_ms, 1e-3))) if pulse else 1.0
            self.neuromod.step(
                self.net.last_population_hz(),
                novelty=novelty * decay,
                salience=salience * decay,
            )
            if collect:
                frames.append(self.observe(update_idle=update_idle))
        return frames

    def observe(self, update_idle: bool = False) -> Telemetry:
        """Measure the current state without advancing it."""
        return self.readout.measure(
            rate_hz=self.net.rate,
            adapt=self.net.adapt,
            neuromod_state=self.neuromod.state,
            pop_trace_hz=self.net.pop_trace(),
            tick=self.net.tick,
            update_idle=update_idle,
        )

    def receive(self, text: str, ticks: Optional[int] = None, on_tick: Optional[Callable[[TickSample], None]] = None) -> Telemetry:
        """Deliver text as a stimulus and summarise the reaction.

        The returned :class:`Telemetry` is a window summary (peak agitation,
        mean scatter, final mood), not a single instant -- a reaction is a
        transient, and sampling only its tail would report the adapted
        aftermath instead of the outburst.
        """
        ticks = self.config.stimulus_ticks if ticks is None else int(ticks)
        salience = SensoryEncoder.salience(text)
        pulse = self.encoder.pulse(text, ticks, tau_ms=self.config.stimulus_tau_ms, dt_ms=self.config.dt_ms)
        frames = self._advance(
            ticks,
            external=pulse,
            novelty=min(1.0, 0.35 + salience),
            salience=salience,
            collect=True,
            update_idle=False,
            on_tick=on_tick,
        )
        telemetry = Telemetry.aggregate(frames)
        self.history.append(telemetry)
        return telemetry

    def settle(self, ticks: Optional[int] = None, on_tick: Optional[Callable[[TickSample], None]] = None) -> Telemetry:
        """Let the network run with no stimulus.

        Mood decays, adaptation clears, and the readout's idle baseline keeps
        learning -- so a long silence really does change how the next sentence
        lands.
        """
        ticks = self.config.settle_ticks if ticks is None else int(ticks)
        frames = self._advance(ticks, collect=True, update_idle=True, on_tick=on_tick)
        telemetry = frames[-1] if frames else self.observe(update_idle=True)
        self.history.append(telemetry)
        return telemetry

    # ---------------------------------------------------------------- builders

    @classmethod
    def from_source(cls, source: str = "surrogate", config: Optional[BrainConfig] = None, **kwargs: Any) -> "BrainSim":
        """Build a brain straight from a named data source.

        Imported lazily so the connectome fetchers stay optional.
        """
        from ..connectome.fetch import load_or_build_connectome

        cfg = config or BrainConfig()
        graph = load_or_build_connectome(
            source=source,
            neurons=cfg.neurons,
            seed=cfg.seed,
            max_edges=cfg.max_edges,
            **kwargs,
        )
        return cls(graph, cfg)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "tick": self.net.tick,
            "drive_scale": round(float(self.net.drive_scale), 4),
            "telemetry": self.observe().as_dict(),
        }
