"""Command line interface.

    flykirk doctor
    flykirk fetch annotations
    flykirk brain --text "banana"
    flykirk debate --rounds 3 --offline
    flykirk debate --topic "Resolved: soup is not a meal."
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence

from . import __version__
from .brain.sim import BrainConfig, BrainSim
from .connectome.fetch import (
    ZENODO_FEATHER,
    ZENODO_RECORD,
    annotations_path,
    fetch_annotations,
    fetch_connectome,
    load_or_build_connectome,
)
from .debate.arena import Arena, FlyAgent, Turn, brain_judge
from .llm.client import OpenAICompatClient, make_client
from .paths import data_dir, ensure_dir
from .persona.kirk import PARODY_BANNER, REGISTERS, default_motions, register_for

__all__ = ["main", "build_parser"]

DEFAULT_MODEL = "hf.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF:Q4_K_M"
DEFAULT_BASE_URL = "http://localhost:11434/v1"


# --------------------------------------------------------------------- output


class Palette:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def cyan(self, text: str) -> str:
        return self._wrap("36", text)

    def yellow(self, text: str) -> str:
        return self._wrap("33", text)

    def magenta(self, text: str) -> str:
        return self._wrap("35", text)

    def green(self, text: str) -> str:
        return self._wrap("32", text)


def _palette() -> Palette:
    return Palette(sys.stdout.isatty() and not os.environ.get("NO_COLOR"))


# ------------------------------------------------------------------ argument parsing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flykirk",
        description="A Drosophila connectome drives a local Liquid AI SLM in campus-debate parody.",
        epilog=PARODY_BANNER,
    )
    parser.add_argument("--version", action="version", version=f"flykirk {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # doctor ---------------------------------------------------------------
    doc = sub.add_parser("doctor", help="check the environment and cached data")
    doc.add_argument("--base-url", default=os.environ.get("FLYKIRK_BASE_URL", DEFAULT_BASE_URL))
    doc.add_argument("--model", default=os.environ.get("FLYKIRK_MODEL", DEFAULT_MODEL))

    # fetch ----------------------------------------------------------------
    fetch = sub.add_parser("fetch", help="download and build connectome data")
    fetch.add_argument("what", choices=["annotations", "connectome"])
    fetch.add_argument("--source", default="surrogate", choices=["surrogate", "zenodo", "neuprint", "annotations"])
    fetch.add_argument("--neurons", type=int, default=4000)
    fetch.add_argument("--seed", type=int, default=7)
    fetch.add_argument("--max-edges", type=int, default=2_000_000)
    fetch.add_argument("--min-synapses", type=int, default=3)
    fetch.add_argument("--force", action="store_true")
    fetch.add_argument("--data")

    # graph ----------------------------------------------------------------
    graph = sub.add_parser("graph", help="describe the connectome in use")
    graph.add_argument("--source", default="surrogate")
    graph.add_argument("--neurons", type=int, default=4000)
    graph.add_argument("--seed", type=int, default=7)
    graph.add_argument("--data")
    graph.add_argument("--refresh", action="store_true")

    # brain ----------------------------------------------------------------
    brain = sub.add_parser("brain", help="stimulate one brain with text and read it out")
    brain.add_argument("--text", required=True)
    brain.add_argument("--source", default="surrogate")
    brain.add_argument("--neurons", type=int, default=4000)
    brain.add_argument("--seed", type=int, default=7)
    brain.add_argument("--ticks", type=int, default=None)
    brain.add_argument("--settle", type=int, default=None)
    brain.add_argument("--calibrate", action="store_true")
    brain.add_argument("--data")
    brain.add_argument("--json", action="store_true")

    # debate ---------------------------------------------------------------
    debate = sub.add_parser("debate", help="run a debate between two flies")
    debate.add_argument("--topic", action="append", default=[])
    debate.add_argument("--random-topic", action="store_true", help="pick a random absurd motion")
    debate.add_argument("--rounds", type=int, default=3)
    debate.add_argument("--source", default="surrogate")
    debate.add_argument("--neurons", type=int, default=4000)
    debate.add_argument("--seed", type=int, default=7)
    debate.add_argument("--data")
    debate.add_argument("--register", default="campus-debate")
    debate.add_argument("--offline", action="store_true", help="no model required")
    debate.add_argument("--base-url", default=os.environ.get("FLYKIRK_BASE_URL", DEFAULT_BASE_URL))
    debate.add_argument("--model", default=os.environ.get("FLYKIRK_MODEL", DEFAULT_MODEL))
    debate.add_argument("--api-key", default=os.environ.get("FLYKIRK_API_KEY"))
    debate.add_argument("--timeout", type=float, default=180.0)
    debate.add_argument("--judge-model", default=None, help="separate model for the judge")
    debate.add_argument("--no-judge", action="store_true")
    debate.add_argument("--show-brain", action="store_true", help="print the brain readout each turn")
    debate.add_argument("--json-out", default=None)

    # personas -------------------------------------------------------------
    sub.add_parser("personas", help="list available registers")

    return parser


# ------------------------------------------------------------------ helpers


def _brain_config(args: argparse.Namespace) -> BrainConfig:
    return BrainConfig(
        neurons=getattr(args, "neurons", 4000),
        seed=getattr(args, "seed", 7),
        calibrate=bool(getattr(args, "calibrate", False)),
    )


def _load_graph(args: argparse.Namespace):
    cfg = _brain_config(args)
    return load_or_build_connectome(
        source=getattr(args, "source", "surrogate"),
        root=Path(args.data) if getattr(args, "data", None) else None,
        refresh=bool(getattr(args, "refresh", False)),
        neurons=cfg.neurons,
        seed=cfg.seed,
        max_edges=cfg.max_edges,
    )


def _print_brain(title: str, telemetry, pal: "Palette") -> None:
    print(pal.dim(f"  ── {title} ──"))
    print(telemetry.render())


# ------------------------------------------------------------------ commands


def cmd_doctor(args: argparse.Namespace) -> int:
    pal = _palette()
    print(pal.bold(f"flykirk {__version__}"))
    print(f"  python        {sys.version.split()[0]}")
    try:
        import numpy

        print(f"  numpy         {numpy.__version__}")
    except ImportError:  # pragma: no cover
        print(pal.yellow("  numpy         MISSING - pip install numpy"))

    root = ensure_dir(data_dir())
    print(f"  data dir      {root}")
    ann = annotations_path(root)
    print(f"  annotations   {'cached, %.1f MB' % (ann.stat().st_size / 1e6) if ann.exists() else 'not downloaded'}")
    graph = root / "connectome.npz"
    print(f"  connectome    {'cached, %.1f MB' % (graph.stat().st_size / 1e6) if graph.exists() else 'not built'}")
    feather = root / ZENODO_FEATHER
    print(
        f"  flywire raw   {'cached, %.1f MB' % (feather.stat().st_size / 1e6) if feather.exists() else 'not downloaded'}"
    )
    print(pal.dim(f"  (full connectivity: https://zenodo.org/records/{ZENODO_RECORD})"))

    if args.base_url:
        client = OpenAICompatClient(base_url=args.base_url, model=args.model, timeout=10.0)
        models = client.health()
        if models and not models[0].startswith("unreachable"):
            print(pal.green(f"  model server  ok, {len(models)} model(s)"))
            for name in models[:5]:
                print(f"                - {name}")
            if args.model not in models:
                print(pal.yellow(f"  note          {args.model!r} is not in the server's model list"))
        else:
            print(pal.yellow(f"  model server  {models[0] if models else 'unreachable'}"))
            print(pal.dim("                start one with: ollama serve"))
            print(pal.dim("                or: llama serve -hf LiquidAI/LFM2-700M-GGUF:Q4_K_M"))
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    pal = _palette()
    root = ensure_dir(Path(args.data) if args.data else data_dir())
    if args.what == "annotations":
        table = fetch_annotations(root, force=args.force)
        print(pal.green(f"annotations: {len(table):,} neurons -> {annotations_path(root)}"))
        for name, count in table.counts("super_class", limit=10):
            print(f"  {name:<24}{count:>9,}")
        return 0
    graph = fetch_connectome(
        source=args.source,
        root=root,
        neurons=args.neurons,
        seed=args.seed,
        max_edges=args.max_edges,
        min_synapses=args.min_synapses,
        force=args.force,
    )
    print(pal.green(f"connectome built -> {root / 'connectome.npz'}"))
    print(graph.describe())
    return 0


def cmd_graph(args: argparse.Namespace) -> int:
    graph = _load_graph(args)
    print(graph.describe())
    return 0


def cmd_brain(args: argparse.Namespace) -> int:
    pal = _palette()
    graph = _load_graph(args)
    cfg = _brain_config(args).replace(
        stimulus_ticks=args.ticks or BrainConfig.stimulus_ticks,
        settle_ticks=args.settle or BrainConfig.settle_ticks,
    )
    sim = BrainSim(graph, cfg)
    sim.settle(cfg.settle_ticks)
    before = sim.observe()
    telemetry = sim.receive(args.text)
    if args.json:
        print(json.dumps({"before": before.as_dict(), "reaction": telemetry.as_dict()}, indent=2))
        return 0

    print(pal.bold("flykirk brain"))
    print(pal.dim(f"  {graph.source} — {graph.n_neurons:,} neurons, {graph.n_edges:,} connections"))
    print()
    print(f"  stimulus      {args.text!r}")
    print(pal.dim("  the fly does not read: the text is hashed onto sensory neurons and the"))
    print(pal.dim("  brain does the rest. Below is what the descending pool actually did."))
    print()
    _print_brain("idle, before the stimulus", before, pal)
    print()
    _print_brain("reaction to the stimulus", telemetry, pal)
    print()
    print(pal.cyan("  the reaction is measured as deviation from the fly's own idle baseline."))
    return 0


def _pick_topic(args: argparse.Namespace) -> str:
    motions = list(args.topic)
    if args.random_topic or not motions:
        import random

        motions = [random.Random(args.seed).choice(default_motions())]
    return motions[0]


def cmd_debate(args: argparse.Namespace) -> int:
    pal = _palette()
    style = register_for(args.register)
    graph = _load_graph(args)
    cfg = _brain_config(args)
    topic = _pick_topic(args)

    fly_a = BrainSim(graph, cfg, seed=cfg.seed)
    fly_b = BrainSim(graph, cfg, seed=cfg.seed + 1)
    fly_a.settle(cfg.settle_ticks)
    fly_b.settle(cfg.settle_ticks)

    client_a = make_client(
        offline=args.offline, seed=cfg.seed, style=style, base_url=args.base_url,
        model=args.model, api_key=args.api_key, timeout=args.timeout,
    )
    client_b = make_client(
        offline=args.offline, seed=cfg.seed + 1, style=style, base_url=args.base_url,
        model=args.model, api_key=args.api_key, timeout=args.timeout,
    )
    judge = None
    if not args.no_judge:
        if args.offline:
            judge = brain_judge()
        else:
            judge = OpenAICompatClient(
                base_url=args.base_url,
                model=args.judge_model or args.model,
                api_key=args.api_key,
                timeout=args.timeout,
            )

    agents = [
        FlyAgent(name="FLY-1", sim=fly_a, client=client_a, style=style, side="for", seed=cfg.seed),
        FlyAgent(name="FLY-2", sim=fly_b, client=client_b, style=style, side="against", seed=cfg.seed + 1),
    ]

    def on_turn(turn: Turn) -> None:
        print(pal.bold(f"\n{turn.speaker}") + pal.dim(f"  (round {turn.round})"))
        if args.show_brain:
            print(turn.telemetry.render())
            print(pal.dim(f"  sampler: {turn.sampling.describe()}"))
        print(pal.cyan(f"  {turn.text}"))

    arena = Arena(agents, judge=judge, on_turn=on_turn)

    print(pal.bold("flykirk"))
    print(f"  motion    {topic}")
    print(f"  brains    {graph.source} — {graph.n_neurons:,} neurons, {graph.n_edges:,} connections")
    print(f"  speakers  {client_a.name} / {client_b.name}")
    print(pal.dim(f"  {PARODY_BANNER}"))

    transcript = arena.run(topic, rounds=args.rounds, judge=not args.no_judge)

    if transcript.verdict:
        verdict = transcript.verdict
        print(pal.bold("\nverdict"))
        print(f"  winner  {verdict.winner}")
        if verdict.reason:
            print(f"  reason  {verdict.reason}")
        if verdict.scores:
            for name, score in sorted(verdict.scores.items(), key=lambda kv: -kv[1]):
                print(f"  {name}  {score:.2f}")

    if args.json_out:
        Path(args.json_out).write_text(transcript.to_json())
        print(pal.dim(f"\nwrote {args.json_out}"))
    return 0


def cmd_personas(_: argparse.Namespace) -> int:
    for key, style in REGISTERS.items():
        print(f"{key}  ({style.display_name})")
        print(f"  {style.tagline}")
        print(f"  max {style.max_claims} claims, {style.max_words} words per turn")
    print()
    print(PARODY_BANNER)
    return 0


_COMMANDS = {
    "doctor": cmd_doctor,
    "fetch": cmd_fetch,
    "graph": cmd_graph,
    "brain": cmd_brain,
    "debate": cmd_debate,
    "personas": cmd_personas,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _COMMANDS[args.command](args)
    except KeyboardInterrupt:  # pragma: no cover
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports, never tracebacks
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
