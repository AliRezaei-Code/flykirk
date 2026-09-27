"""Live UI server: a local dashboard for a running fly brain.

Standard library only -- ``http.server`` plus Server-Sent Events -- so ``flykirk
serve`` needs nothing that ``flykirk`` does not already need. The HUD panels
read real quantities: the dopamine figure is the neuromodulator concentration,
the spike counter is the cumulative spike count, the raster is the actual spike
train of 96 stratified neurons, and the sampler readout is the sampler that is
actually being used.

    flykirk serve                    # http://127.0.0.1:8765
    flykirk serve --source zenodo    # the real 139k-neuron brain
"""

from __future__ import annotations

import json
import queue
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

from .brain.sim import BrainConfig, BrainSim, TickSample
from .connectome.fetch import load_or_build_connectome
from .debate.arena import Arena, FlyAgent, Turn, brain_judge
from .llm.client import OpenAICompatClient, make_client
from .persona.kirk import default_motions, register_for

__all__ = ["run_server", "Hub", "ServerState"]

WEB_ROOT = Path(__file__).resolve().parent / "web"

_SAFE_SUFFIXES = {".html", ".css", ".js", ".svg", ".png", ".ico", ".webmanifest"}
_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".webmanifest": "application/manifest+json",
}


class DebateStopped(RuntimeError):
    """Raised inside the simulation to unwind a stopped debate."""


# --------------------------------------------------------------------- pub/sub


class Hub:
    """Fan-out of JSON events to connected browsers.

    Subscribers get a bounded queue: a browser that stops reading falls behind
    and drops samples rather than growing the server's memory without limit.
    """

    def __init__(self, backlog: int = 32, max_queue: int = 512) -> None:
        self._subscribers: List[queue.Queue] = []
        self._lock = threading.Lock()
        self._backlog = backlog
        self._max_queue = max_queue
        self._recent: List[Dict[str, Any]] = []
        self.dropped = 0

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=self._max_queue)
        with self._lock:
            self._subscribers.append(q)
            recent = list(self._recent)
        for event in recent:
            try:
                q.put_nowait(event)
            except queue.Full:  # pragma: no cover - only on a very slow client
                pass
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def publish(self, event: Dict[str, Any]) -> None:
        with self._lock:
            self._recent.append(event)
            if len(self._recent) > self._backlog:
                self._recent = self._recent[-self._backlog :]
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(event)
            except queue.Full:
                self.dropped += 1

    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)


# ----------------------------------------------------------------- server state


@dataclass
class ServerOptions:
    source: str = "surrogate"
    neurons: int = 1500
    seed: int = 7
    offline: bool = False
    base_url: str = "http://localhost:11434/v1"
    model: str = "fm2"
    api_key: Optional[str] = None
    register: str = "kirk"
    data: Optional[str] = None
    #: Milliseconds to sleep between samples. The brain runs far faster than
    #: real time; pacing is what makes the raster watchable.
    pace_ms: float = 25.0


@dataclass
class ServerState:
    options: ServerOptions
    hub: Hub = field(default_factory=Hub)
    connectome: Any = None
    error: Optional[str] = None
    running: bool = False
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: Optional[threading.Thread] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # ------------------------------------------------------------------ setup

    def load(self) -> None:
        """Build the connectome in the background so the page can render first."""
        self.hub.publish({"type": "status", "state": "loading", "detail": f"building {self.options.source} connectome"})
        try:
            graph = load_or_build_connectome(
                source=self.options.source,
                root=Path(self.options.data) if self.options.data else None,
                neurons=self.options.neurons,
                seed=self.options.seed,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            self.error = str(exc)
            self.hub.publish({"type": "status", "state": "error", "detail": self.error})
            return
        self.connectome = graph
        self.hub.publish({"type": "status", "state": "ready", "detail": f"{graph.n_neurons:,} neurons"})
        self.hub.publish(self.hello_payload())

    def hello_payload(self) -> Dict[str, Any]:
        graph = self.connectome
        if graph is None:
            return {"type": "hello", "ready": False}
        cfg = self._brain_config()
        probe = BrainSim(graph, cfg, seed=self.options.seed)
        channels = [
            {
                "i": int(i),
                "super_class": str(graph.neurons.super_class[i]),
                "cell_class": str(graph.neurons.cell_class[i]),
            }
            for i in probe.monitored.tolist()
        ]
        return {
            "type": "hello",
            "ready": True,
            "source": graph.source,
            "neurons": graph.n_neurons,
            "connections": graph.n_edges,
            "synapses": int(graph.n_synapses),
            "inhibitory_fraction": float((graph.signed_weights < 0).mean()) if graph.n_edges else 0.0,
            "channels": channels,
            "monitored": int(probe.monitored.shape[0]),
            "sample_every": cfg.sample_every,
            "dt_ms": cfg.dt_ms,
            "motions": default_motions(),
        }

    def _brain_config(self) -> BrainConfig:
        return BrainConfig(neurons=self.options.neurons, seed=self.options.seed)

    # ------------------------------------------------------------------ debate

    def start_debate(self, params: Dict[str, Any]) -> bool:
        with self._lock:
            if self.running:
                return False
            self.running = True
            self._stop.clear()
            self._thread = threading.Thread(target=self._run_debate, args=(params,), daemon=True)
            self._thread.start()
            return True

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------ runner

    def _run_debate(self, params: Dict[str, Any]) -> None:
        hub = self.hub
        try:
            if self.connectome is None:
                if self.error:
                    raise RuntimeError(self.error)
                self.load()
                if self.connectome is None:
                    raise RuntimeError(self.error or "connectome unavailable")

            graph = self.connectome
            cfg = self._brain_config()
            topic = str(params.get("topic") or default_motions()[0])
            rounds = max(1, min(int(params.get("rounds", 2)), 8))
            offline = bool(params.get("offline", self.options.offline))
            pace = max(0.0, float(params.get("pace_ms", self.options.pace_ms))) / 1000.0
            style = register_for(str(params.get("register") or self.options.register))

            hub.publish({"type": "status", "state": "running", "detail": topic})
            hub.publish({"type": "debate_start", "topic": topic, "rounds": rounds, "offline": offline})

            fly_a = BrainSim(graph, cfg, seed=self.options.seed)
            fly_b = BrainSim(graph, cfg, seed=self.options.seed + 1)
            self._idle(fly_a, hub, pace, "FLY-1", 0)
            self._idle(fly_b, hub, pace, "FLY-2", 0)

            client_kwargs = dict(
                base_url=self.options.base_url,
                model=str(params.get("model") or self.options.model),
                api_key=self.options.api_key,
                timeout=180.0,
            )
            agents = [
                FlyAgent(
                    name="FLY-1",
                    sim=fly_a,
                    client=make_client(offline=offline, seed=self.options.seed, style=style, **client_kwargs),
                    style=style,
                    side="for",
                    seed=self.options.seed,
                ),
                FlyAgent(
                    name="FLY-2",
                    sim=fly_b,
                    client=make_client(offline=offline, seed=self.options.seed + 1, style=style, **client_kwargs),
                    style=style,
                    side="against",
                    seed=self.options.seed + 1,
                ),
            ]
            judge = brain_judge() if (offline or params.get("no_judge")) else OpenAICompatClient(**client_kwargs)

            def on_tick(name: str, sample: TickSample) -> None:
                if self._stop.is_set():
                    raise DebateStopped()
                hub.publish({"type": "tick", "fly": name, "round": sample.tick and 0 or 0, **sample.as_dict()})
                if pace:
                    time.sleep(pace)

            def on_turn(turn: Turn) -> None:
                hub.publish(
                    {
                        "type": "turn",
                        "fly": turn.speaker,
                        "side": turn.side,
                        "round": turn.round,
                        "text": turn.text,
                        "brain_ms": round(turn.brain_ms, 1),
                        "telemetry": turn.telemetry.as_dict(),
                        "sampling": {
                            "temperature": turn.sampling.temperature,
                            "top_p": turn.sampling.top_p,
                            "frequency_penalty": turn.sampling.frequency_penalty,
                            "presence_penalty": turn.sampling.presence_penalty,
                            "max_tokens": turn.sampling.max_tokens,
                        },
                    }
                )

            arena = Arena(agents, judge=judge, on_turn=on_turn, on_tick=on_tick)
            transcript = arena.run(topic, rounds=rounds, judge=not params.get("no_judge", False))

            if transcript.verdict:
                hub.publish(
                    {
                        "type": "verdict",
                        "winner": transcript.verdict.winner,
                        "reason": transcript.verdict.reason,
                        "scores": transcript.verdict.scores,
                    }
                )
            hub.publish({"type": "done", "turns": len(transcript.turns)})
        except DebateStopped:
            hub.publish({"type": "status", "state": "stopped", "detail": "stopped by operator"})
            hub.publish({"type": "done", "turns": 0})
        except Exception as exc:  # noqa: BLE001 - reported to the UI, never a traceback
            self.error = str(exc)
            hub.publish({"type": "status", "state": "error", "detail": str(exc)})
            hub.publish({"type": "error", "detail": str(exc)})
            hub.publish({"type": "done", "turns": 0})
        finally:
            with self._lock:
                self.running = False
            hub.publish({"type": "status", "state": "idle", "detail": "ready"})

    def _idle(self, sim: BrainSim, hub: Hub, pace: float, name: str, rnd: int) -> None:
        """Let the fly be quiet before the debate, streaming its idle brain."""

        def on_tick(sample: TickSample) -> None:
            if self._stop.is_set():
                raise DebateStopped()
            hub.publish({"type": "tick", "fly": name, "round": rnd, "idle": True, **sample.as_dict()})
            if pace:
                time.sleep(pace)

        sim.settle(sim.config.settle_ticks, on_tick=on_tick)


# ----------------------------------------------------------------------- HTTP


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "flykirk"

    state: ServerState  # injected on the server class

    # ------------------------------------------------------------------ utils

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 - stdlib signature
        pass  # the HUD is the log

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return {}
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8")) or {}
        except (ValueError, UnicodeDecodeError):
            return {}

    # ------------------------------------------------------------------ routes

    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return self._serve_file("index.html")
        if path == "/api/status":
            state = self.state
            return self._send_json(
                {
                    "ready": state.connectome is not None,
                    "running": state.running,
                    "error": state.error,
                    "source": getattr(state.connectome, "source", None),
                    "neurons": getattr(state.connectome, "n_neurons", 0),
                    "subscribers": state.hub.subscriber_count(),
                    "dropped": state.hub.dropped,
                }
            )
        if path == "/api/events":
            return self._serve_events()
        if path.startswith("/static/"):
            return self._serve_file(path[len("/static/") :])
        self.send_error(404, "not found")

    def do_POST(self) -> None:  # noqa: N802 - stdlib signature
        path = self.path.split("?", 1)[0]
        if path == "/api/debate":
            params = self._read_json()
            if self.state.running:
                return self._send_json({"ok": False, "error": "a debate is already running"}, 409)
            self.state.start_debate(params)
            return self._send_json({"ok": True}, 202)
        if path == "/api/stop":
            self.state.stop()
            return self._send_json({"ok": True})
        if path == "/api/reload":
            threading.Thread(target=self.state.load, daemon=True).start()
            return self._send_json({"ok": True}, 202)
        self.send_error(404, "not found")

    # ----------------------------------------------------------------- serving

    def _serve_file(self, name: str) -> None:
        candidate = (WEB_ROOT / name).resolve()
        if WEB_ROOT.resolve() not in candidate.parents and candidate != WEB_ROOT.resolve():
            self.send_error(403, "forbidden")
            return
        if candidate.suffix.lower() not in _SAFE_SUFFIXES or not candidate.is_file():
            self.send_error(404, "not found")
            return
        body = candidate.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", _CONTENT_TYPES.get(candidate.suffix.lower(), "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_events(self) -> None:
        hub = self.state.hub
        q = hub.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            while True:
                try:
                    event = q.get(timeout=10.0)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                self.wfile.write(f"data: {json.dumps(event, default=str)}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            hub.unsubscribe(q)
            self.close_connection = True


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def run_server(
    options: ServerOptions,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
    load: bool = True,
) -> int:
    """Serve the dashboard until interrupted."""
    state = ServerState(options=options)
    handler = type("BoundHandler", (_Handler,), {"state": state})
    httpd = _Server((host, port), handler)
    actual_port = httpd.server_address[1]
    url = f"http://{host}:{actual_port}/"
    print(f"flykirk ui  ->  {url}")
    print(f"  connectome  {options.source} (building in background)")
    print(f"  speaker     {'scripted (offline)' if options.offline else options.model}")
    print("  ctrl-c to stop")
    if load:
        threading.Thread(target=state.load, daemon=True).start()
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.shutdown()
        httpd.server_close()
    return 0
