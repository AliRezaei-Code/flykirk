"""Download and construct real Drosophila connectome data.

Three sources, in increasing order of fidelity:

``surrogate``  local block model, no network, seconds (see :mod:`.surrogate`).
``annotations`` FlyWire 783 neuron annotations (~32 MB TSV, 139,255 neurons).
``zenodo``     FlyWire 783 proofread connections (~850 MB feather) -- the real
               wiring diagram, built against the annotations above.
``neuprint``   live queries against a neuPrint instance (hemibrain / MaleCNS).

Everything is cached under :func:`flykirk.paths.data_dir` and nothing large is
written inside the repository.
"""

from __future__ import annotations

import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ..paths import data_dir, ensure_dir
from .graph import Connectome, NeuronTable, normalize_nt
from .surrogate import PROVENANCE, surrogate_connectome

__all__ = [
    "FLYWIRE_ANNOTATIONS_URL",
    "ZENODO_RECORD",
    "download",
    "annotations_path",
    "fetch_annotations",
    "load_annotations",
    "connectome_from_feather",
    "connectome_from_neuprint",
    "fetch_connectome",
    "load_or_build_connectome",
]

FLYWIRE_ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/"
    "supplemental_files/Supplemental_file1_neuron_annotations.tsv"
)

ZENODO_RECORD = "10676866"
ZENODO_FEATHER = "proofread_connections_783.feather"
ZENODO_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{ZENODO_FEATHER}?download=1"

PRE_CANDIDATES = ("pre", "pre_id", "pre_root_id", "pre_pt_root_id", "source", "from")
POST_CANDIDATES = ("post", "post_id", "post_root_id", "post_pt_root_id", "target", "to")
WEIGHT_CANDIDATES = ("syn_count", "synapses", "weight", "n_syn", "count", "size")


# --------------------------------------------------------------------- download


def download(
    url: str,
    dest: Path,
    force: bool = False,
    progress: bool = True,
    chunk: int = 1 << 20,
) -> Path:
    """Stream ``url`` to ``dest`` with resume support.

    Interrupted transfers leave ``dest.part`` behind and pick up from there via
    a Range request, which matters for the 850 MB feather.
    """
    dest = Path(dest)
    ensure_dir(dest.parent)
    part = dest.with_suffix(dest.suffix + ".part")
    if dest.exists() and not force:
        if progress:
            print(f"  cached  {dest} ({dest.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)
        return dest
    if force and part.exists():
        part.unlink()

    offset = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "flykirk/0.1"})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            total = response.headers.get("Content-Length")
            total = int(total) + offset if total else None
            mode = "ab" if offset and response.status == 206 else "wb"
            if mode == "wb":
                offset = 0
            started = time.time()
            with open(part, mode) as fh:
                while True:
                    block = response.read(chunk)
                    if not block:
                        break
                    fh.write(block)
                    offset += len(block)
                    if progress and total:
                        pct = offset / total * 100
                        rate = offset / max(time.time() - started, 1e-6) / 1e6
                        print(
                            f"\r  {dest.name}: {pct:5.1f}%  {offset / 1e6:7.1f}/{total / 1e6:.1f} MB  {rate:5.1f} MB/s",
                            end="",
                            file=sys.stderr,
                            flush=True,
                        )
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and part.exists():
            pass  # already complete
        else:
            raise
    if progress:
        print("", file=sys.stderr)
    part.replace(dest)
    return dest


# ------------------------------------------------------------------ annotations


def annotations_path(root: Optional[Path] = None) -> Path:
    return (root or data_dir()) / "annotations.npz"


def fetch_annotations(
    root: Optional[Path] = None,
    force: bool = False,
    progress: bool = True,
) -> NeuronTable:
    """Download and cache the FlyWire 783 neuron annotation table."""
    root = ensure_dir(root or data_dir())
    tsv = root / "Supplemental_file1_neuron_annotations.tsv"
    download(FLYWIRE_ANNOTATIONS_URL, tsv, force=force, progress=progress)
    table = _parse_annotations_tsv(tsv)
    np.savez_compressed(annotations_path(root), **table.to_dict())
    (root / "annotations.meta.json").write_text(
        json.dumps(
            {
                "source": FLYWIRE_ANNOTATIONS_URL,
                "neurons": len(table),
                "super_classes": table.counts("super_class", limit=50),
                "neurotransmitters": _nt_counts(table),
            },
            indent=2,
        )
    )
    return table


def _nt_counts(table: NeuronTable) -> List[Tuple[str, int]]:
    values, counts = np.unique(table.nt, return_counts=True)
    order = np.argsort(counts)[::-1]
    return [(str(values[i]), int(counts[i])) for i in order]


def _parse_annotations_tsv(tsv: Path) -> NeuronTable:
    """Read the FlyWire annotation TSV, keeping only the columns we use."""
    needed = ("root_id", "super_class", "cell_class", "cell_type", "top_nt")
    records: List[Dict[str, Any]] = []
    soma_xyz: List[Tuple[float, float, float]] = []
    with open(tsv, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        missing = [c for c in needed if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"annotation table is missing columns: {missing}")
        for row in reader:
            root = (row.get("root_id") or "").strip()
            if not root:
                continue
            records.append(
                {
                    "root_id": int(root),
                    "super_class": row.get("super_class") or "unknown",
                    "cell_class": row.get("cell_class") or "unknown",
                    "cell_type": row.get("cell_type") or "unknown",
                    "nt": row.get("top_nt") or "unknown",
                }
            )
            soma_xyz.append(
                (
                    _float(row.get("soma_x")),
                    _float(row.get("soma_y")),
                    _float(row.get("soma_z")),
                )
            )
    table = NeuronTable.from_records(records)
    table.soma = np.asarray(soma_xyz, dtype=np.float32)
    return table


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_annotations(root: Optional[Path] = None) -> NeuronTable:
    path = annotations_path(root)
    if not path.exists():
        raise FileNotFoundError(
            f"no cached annotations at {path}; run `flykirk fetch annotations`"
        )
    with np.load(path, allow_pickle=False) as data:
        return NeuronTable.from_dict({k: data[k] for k in data.files})


# ------------------------------------------------------------------ connectivity


def _pick(names: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lowered = {n.lower(): n for n in names}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def connectome_from_feather(
    feather: Path,
    neurons: NeuronTable,
    min_synapses: int = 3,
    limit_edges: Optional[int] = None,
    include_classes: Optional[Sequence[str]] = None,
    progress: bool = True,
) -> Connectome:
    """Build a Connectome from the FlyWire proofread-connections feather.

    Read batch-by-batch through the Arrow IPC reader so peak memory stays near
    the size of the retained edges rather than the size of the file. Edges whose
    endpoints are not in the annotation table are dropped, as are edges below
    ``min_synapses``.
    """
    try:
        import pyarrow as pa  # noqa: F401
        import pyarrow.ipc as pa_ipc
    except ImportError as exc:  # pragma: no cover - exercised only without pyarrow
        raise RuntimeError(
            "reading the Zenodo feather needs pyarrow: pip install 'flykirk[data]'"
        ) from exc

    index = neurons.index
    class_ok = None
    if include_classes:
        wanted = set(include_classes)
        class_ok = {
            int(r): (c in wanted) for r, c in zip(neurons.root_id.tolist(), neurons.cell_class.tolist())
        }

    pre_parts: List[np.ndarray] = []
    post_parts: List[np.ndarray] = []
    weight_parts: List[np.ndarray] = []
    kept = 0
    dropped_unmapped = 0
    rows_seen = 0

    reader = pa_ipc.open_file(str(feather))
    names = [field.name for field in reader.schema]
    pre_col = _pick(names, PRE_CANDIDATES)
    post_col = _pick(names, POST_CANDIDATES)
    weight_col = _pick(names, WEIGHT_CANDIDATES)
    if not pre_col or not post_col:
        raise ValueError(f"could not find pre/post columns in {names}")

    for batch_index in range(reader.num_record_batches):
        batch = reader.get_batch(batch_index).to_pydict()
        pre_ids = np.asarray(batch[pre_col], dtype=np.int64)
        post_ids = np.asarray(batch[post_col], dtype=np.int64)
        if weight_col:
            weight = np.asarray(batch[weight_col], dtype=np.float32)
        else:
            weight = np.ones(pre_ids.shape[0], dtype=np.float32)
        rows_seen += pre_ids.shape[0]

        pre_ix = np.fromiter((index.get(int(x), -1) for x in pre_ids), dtype=np.int64, count=pre_ids.shape[0])
        post_ix = np.fromiter((index.get(int(x), -1) for x in post_ids), dtype=np.int64, count=post_ids.shape[0])
        keep = (pre_ix >= 0) & (post_ix >= 0) & (weight >= min_synapses)
        dropped_unmapped += int((~((pre_ix >= 0) & (post_ix >= 0))).sum())
        pre_ix, post_ix, weight = pre_ix[keep], post_ix[keep], weight[keep]
        if class_ok is not None and pre_ix.size:
            allowed = np.fromiter(
                (class_ok.get(int(i), False) for i in np.concatenate([pre_ix, post_ix])),
                dtype=bool,
                count=pre_ix.size * 2,
            )
            keep2 = allowed[: pre_ix.shape[0]] & allowed[pre_ix.shape[0] :]
            pre_ix, post_ix, weight = pre_ix[keep2], post_ix[keep2], weight[keep2]
        pre_parts.append(pre_ix)
        post_parts.append(post_ix)
        weight_parts.append(weight)
        kept += pre_ix.shape[0]
        if progress:
            print(
                f"\r  edges kept {kept:,}  scanned {rows_seen:,}  dropped {dropped_unmapped:,}",
                end="",
                file=sys.stderr,
                flush=True,
            )
        if limit_edges and kept >= limit_edges:
            break
    if progress:
        print("", file=sys.stderr)

    if not pre_parts:
        raise ValueError("no edges survived filtering")
    pre = np.concatenate(pre_parts)
    post = np.concatenate(post_parts)
    weight = np.concatenate(weight_parts)
    if limit_edges and pre.shape[0] > limit_edges:
        pre, post, weight = pre[:limit_edges], post[:limit_edges], weight[:limit_edges]

    columns, counts = np.unique(pre, return_counts=True)
    return Connectome.from_edgelist(
        pre,
        post,
        weight,
        neurons,
        source=f"flywire-783/proofread:{feather.name}",
        meta={
            "provenance": "flywire-783",
            "records_scanned": rows_seen,
            "edges_kept": int(pre.shape[0]),
            "min_synapses": min_synapses,
            "unmapped_dropped": dropped_unmapped,
            "active_presynaptic_neurons": int(columns.shape[0]),
        },
    )


def connectome_from_neuprint(
    dataset: str = "hemibrain",
    token: Optional[str] = None,
    server: Optional[str] = None,
    min_weight: int = 5,
) -> Connectome:
    """Query a neuPrint instance for neuronal connectivity.

    Requires ``pip install 'flykirk[neuprint]'`` and a neuPrint auth token
    (see https://neuprint.janelia.org -> Account).
    """
    try:
        from neuprint import Client, fetch_neurons, fetch_simple_connections
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "neuPrint access needs neuprint-python: pip install 'flykirk[neuprint]'"
        ) from exc

    client = Client(server or f"https://neuprint.janelia.org", dataset=dataset, token=token)
    neurons_df, _ = fetch_neurons(client)

    ids = neurons_df["bodyId"].to_numpy()
    index = {int(b): i for i, b in enumerate(ids)}
    table = NeuronTable(
        root_id=ids.astype(np.int64),
        cell_type=neurons_df.get("type", np.array(["unknown"] * len(ids))).astype(str),
        cell_class=neurons_df.get("cellType", np.array(["unknown"] * len(ids))).astype(str),
        super_class=np.array(["unknown"] * len(ids)),
        nt=neurons_df.get("predictedNt", np.array(["unknown"] * len(ids))).astype(str),
    )

    conn = fetch_simple_connections(client, min_weight=min_weight)
    pre = np.fromiter((index.get(int(x), -1) for x in conn["bodyId_pre"]), dtype=np.int64, count=len(conn))
    post = np.fromiter((index.get(int(x), -1) for x in conn["bodyId_post"]), dtype=np.int64, count=len(conn))
    weight = conn["weight"].to_numpy(dtype=np.float32)
    keep = (pre >= 0) & (post >= 0)
    return Connectome.from_edgelist(
        pre[keep],
        post[keep],
        weight[keep],
        table,
        source=f"neuprint:{dataset}",
        meta={"provenance": "neuprint", "dataset": dataset, "min_weight": min_weight},
    )


# ------------------------------------------------------------------ entry points


def fetch_connectome(
    source: str = "surrogate",
    root: Optional[Path] = None,
    neurons: int = 4000,
    seed: int = 7,
    max_edges: int = 2_000_000,
    min_synapses: int = 3,
    include_classes: Optional[Sequence[str]] = None,
    force: bool = False,
    progress: bool = True,
    **kwargs: Any,
) -> Connectome:
    """Build a connectome from ``source`` and cache it as ``connectome.npz``."""
    root = ensure_dir(root or data_dir())
    source = source.lower()
    if source in {PROVENANCE, "surrogate", "synthetic"}:
        graph = surrogate_connectome(neurons=neurons, seed=seed, max_edges=max_edges)
    elif source in {"annotations", "flywire", "zenodo"}:
        table = fetch_annotations(root, force=force, progress=progress)
        if source == "annotations":
            raise ValueError(
                "the annotations table contains no connectivity; "
                "use --source zenodo for the wiring diagram"
            )
        feather = download(ZENODO_URL, root / ZENODO_FEATHER, force=force, progress=progress)
        graph = connectome_from_feather(
            feather,
            table,
            min_synapses=min_synapses,
            limit_edges=kwargs.get("limit_edges"),
            include_classes=include_classes,
            progress=progress,
        )
    elif source == "neuprint":
        graph = connectome_from_neuprint(**kwargs)
    else:
        raise ValueError(f"unknown connectome source {source!r}")
    graph.save(root / "connectome.npz")
    return graph


def load_or_build_connectome(
    source: str = "surrogate",
    root: Optional[Path] = None,
    refresh: bool = False,
    **kwargs: Any,
) -> Connectome:
    """Load the cached connectome if it matches ``source``, else build it."""
    root = ensure_dir(root or data_dir())
    cache = root / "connectome.npz"
    if cache.exists() and not refresh:
        graph = Connectome.load(cache)
        if source in {PROVENANCE, "surrogate", "synthetic"} and graph.source == PROVENANCE:
            return graph
        if source not in {PROVENANCE, "surrogate", "synthetic"} and graph.source.startswith("flywire"):
            return graph
    return fetch_connectome(source=source, root=root, progress=True, **kwargs)


def iter_edges(graph: Connectome) -> Iterator[Tuple[int, int, float]]:
    """Yield ``(pre_index, post_index, weight)`` for every stored edge."""
    for i in range(graph.n):
        lo, hi = graph.indptr[i], graph.indptr[i + 1]
        for j in range(lo, hi):
            yield i, int(graph.indices[j]), float(graph.weights[j])
