"""Sparse, signed, synapse-weighted connectome graph.

Nodes are neurons keyed by FlyWire ``root_id``. Edges are chemical synapses
aggregated per (pre, post) pair and weighted by synapse count -- Drosophila
synapses are polyadic, so one anatomical connection routinely carries tens of
release sites, and that count is the natural weight.

Storage is CSR (``indptr`` / ``indices`` / ``weights``) plus a precomputed
``row_of_nnz`` vector, which lets synaptic input be computed with a single
``np.bincount`` and keeps the whole thing numpy-only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "Connectome",
    "NeuronTable",
    "normalize_nt",
    "EXCITATORY",
    "INHIBITORY",
    "MODULATORY",
]

EXCITATORY = 1.0
INHIBITORY = -1.0
#: Aminergic / nitrergic neurons are neuromodulatory: they are kept in the
#: graph but contribute a weak, sign-preserving drive rather than a fast
#: excitatory or inhibitory postsynaptic current.
MODULATORY = 0.35

# Longest keys first so "acetylcholine" wins over a hypothetical "ach" prefix.
_NT_TABLE: Tuple[Tuple[str, str, float], ...] = (
    ("acetylcholine", "acetylcholine", EXCITATORY),
    ("glutamate", "glutamate", EXCITATORY),
    ("histamine", "histamine", INHIBITORY),
    ("gaba", "gaba", INHIBITORY),
    ("glycine", "glycine", INHIBITORY),
    ("dopamine", "dopamine", MODULATORY),
    ("serotonin", "serotonin", MODULATORY),
    ("octopamine", "octopamine", MODULATORY),
    ("tyramine", "tyramine", MODULATORY),
    ("nitric oxide", "nitric oxide", MODULATORY),
    ("ach", "acetylcholine", EXCITATORY),
    ("glu", "glutamate", EXCITATORY),
    ("gaba/gly", "gaba", INHIBITORY),
    ("gly", "glycine", INHIBITORY),
)

_UNKNOWN_NT = frozenset({"", "none", "nan", "unknown", "?", "unk", "nt_unknown", "no_nt", "not_predicted"})


def normalize_nt(raw: Any) -> Tuple[str, float]:
    """Map a FlyWire ``top_nt`` string onto ``(canonical_name, sign)``.

    Unpredicted or unrecognised transmitters fall back to excitatory drive at
    unit sign; that is the overwhelmingly common case in the fly brain and keeps
    the model from silently deleting edges.
    """
    if raw is None:
        return "unknown", EXCITATORY
    text = str(raw).strip().lower()
    if text in _UNKNOWN_NT:
        return "unknown", EXCITATORY
    for needle, name, sign in _NT_TABLE:
        if needle in text:
            return name, sign
    return text, EXCITATORY


class NeuronTable:
    """Column of per-neuron annotations aligned with the graph's node order."""

    __slots__ = ("root_id", "cell_type", "cell_class", "super_class", "nt", "sign", "soma")

    def __init__(
        self,
        root_id: np.ndarray,
        cell_type: np.ndarray,
        cell_class: np.ndarray,
        super_class: np.ndarray,
        nt: np.ndarray,
        sign: Optional[np.ndarray] = None,
        soma: Optional[np.ndarray] = None,
    ) -> None:
        self.root_id = np.asarray(root_id, dtype=np.int64)
        n = self.root_id.shape[0]
        self.cell_type = _str_col(cell_type, n, "unknown")
        self.cell_class = _str_col(cell_class, n, "unknown")
        self.super_class = _str_col(super_class, n, "unknown")
        self.nt = _str_col(nt, n, "unknown")
        if sign is None:
            sign = np.array([normalize_nt(x)[1] for x in self.nt], dtype=np.float32)
        self.sign = np.asarray(sign, dtype=np.float32)
        self.soma = None if soma is None else np.asarray(soma, dtype=np.float32)

    def __len__(self) -> int:
        return int(self.root_id.shape[0])

    def __repr__(self) -> str:
        return f"<NeuronTable n={len(self)} types={len(set(self.cell_type.tolist()))}>"

    @property
    def index(self) -> Dict[int, int]:
        return {int(k): i for i, k in enumerate(self.root_id.tolist())}

    def subtable(self, mask: np.ndarray) -> "NeuronTable":
        mask = np.asarray(mask, dtype=bool)
        return NeuronTable(
            self.root_id[mask],
            self.cell_type[mask],
            self.cell_class[mask],
            self.super_class[mask],
            self.nt[mask],
            self.sign[mask],
            None if self.soma is None else self.soma[mask],
        )

    def counts(self, field_name: str = "cell_class", limit: int = 15) -> list:
        col = getattr(self, field_name)
        values, counts = np.unique(col, return_counts=True)
        order = np.argsort(counts)[::-1][:limit]
        return [(str(values[i]), int(counts[i])) for i in order]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_id": self.root_id,
            "cell_type": self.cell_type,
            "cell_class": self.cell_class,
            "super_class": self.super_class,
            "nt": self.nt,
            "sign": self.sign,
            "soma": self.soma if self.soma is not None else np.zeros((0, 3), dtype=np.float32),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "NeuronTable":
        soma = raw.get("soma")
        soma = None if soma is None or np.asarray(soma).size == 0 else np.asarray(soma)
        return cls(
            raw["root_id"],
            raw["cell_type"],
            raw["cell_class"],
            raw["super_class"],
            raw["nt"],
            raw.get("sign"),
            soma,
        )

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, Any]]) -> "NeuronTable":
        rows = list(records)
        if not rows:
            return cls(np.zeros(0, np.int64), [], [], [], [])
        nt = [r.get("nt", "unknown") for r in rows]
        return cls(
            np.array([int(r["root_id"]) for r in rows], dtype=np.int64),
            [str(r.get("cell_type", "unknown")) for r in rows],
            [str(r.get("cell_class", "unknown")) for r in rows],
            [str(r.get("super_class", "unknown")) for r in rows],
            nt,
            np.array([normalize_nt(x)[1] for x in nt], dtype=np.float32),
        )


def _str_col(values: Any, n: int, fill: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=object) if not isinstance(values, np.ndarray) else values
    if arr.shape[0] != n:
        raise ValueError(f"column length {arr.shape[0]} != neuron count {n}")
    return np.array([fill if v is None else str(v) for v in arr.tolist()], dtype="U128")


@dataclass
class Connectome:
    """A directed, signed, synapse-weighted wiring diagram."""

    neurons: NeuronTable
    indptr: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    source: str = "unknown"
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.indptr = np.ascontiguousarray(self.indptr, dtype=np.int64)
        self.indices = np.ascontiguousarray(self.indices, dtype=np.int32)
        self.weights = np.ascontiguousarray(self.weights, dtype=np.float32)
        n = len(self.neurons)
        if self.indptr.shape[0] != n + 1:
            raise ValueError(f"indptr has length {self.indptr.shape[0]}, expected {n + 1}")
        if self.indices.shape != self.weights.shape:
            raise ValueError("indices and weights disagree in length")
        if self.indices.size and (self.indices.min() < 0 or self.indices.max() >= n):
            raise ValueError("edge target out of range")
        # Row index per stored edge: precomputed because it is needed on every
        # synaptic-current call and recomputing it means materialising a repeat.
        self.row_of_nnz = np.repeat(np.arange(n, dtype=np.int32), np.diff(self.indptr))
        # Presynaptic sign folded into the weight, so the hot loop is one bincount.
        self.signed_weights = (self.weights * self.neurons.sign[self.row_of_nnz]).astype(np.float32)

    # ------------------------------------------------------------------ basics

    def __len__(self) -> int:
        return len(self.neurons)

    def __repr__(self) -> str:
        return (
            f"<Connectome n={self.n_neurons} edges={self.n_edges} "
            f"synapses={self.n_synapses} source={self.source!r}>"
        )

    @property
    def n(self) -> int:
        return len(self.neurons)

    @property
    def n_neurons(self) -> int:
        return len(self.neurons)

    @property
    def n_edges(self) -> int:
        return int(self.indices.shape[0])

    @property
    def n_synapses(self) -> float:
        return float(self.weights.sum())

    @property
    def n_bytes(self) -> int:
        return int(self.indptr.nbytes + self.indices.nbytes + self.weights.nbytes + self.row_of_nnz.nbytes)

    # ------------------------------------------------------------- construction

    @staticmethod
    def from_edgelist(
        pre: Sequence[int] | np.ndarray,
        post: Sequence[int] | np.ndarray,
        weight: Optional[Sequence[float] | np.ndarray] = None,
        neurons: Optional[NeuronTable] = None,
        source: str = "edgelist",
        meta: Optional[Dict[str, Any]] = None,
    ) -> "Connectome":
        """Aggregate parallel edges and build CSR.

        Duplicate (pre, post) pairs are summed, which is exactly what happens
        when individual synapses are collapsed into connections.
        """
        pre = np.asarray(pre, dtype=np.int64).ravel()
        post = np.asarray(post, dtype=np.int64).ravel()
        if pre.shape != post.shape:
            raise ValueError("pre and post must be the same length")
        if neurons is None:
            raise ValueError("neurons is required")
        n = len(neurons)
        if weight is None:
            weight = np.ones(pre.shape[0], dtype=np.float32)
        weight = np.asarray(weight, dtype=np.float32).ravel()
        if weight.shape != pre.shape:
            raise ValueError("weight must match pre")

        if pre.size:
            if pre.min() < 0 or pre.max() >= n or post.min() < 0 or post.max() >= n:
                raise ValueError("edgelist node ids out of range")
            key = pre * np.int64(n) + post
            ukey, inverse = np.unique(key, return_inverse=True)
            summed = np.bincount(inverse, weights=weight.astype(np.float64), minlength=ukey.shape[0])
            rows = (ukey // n).astype(np.int64)
            cols = (ukey % n).astype(np.int32)
            indptr = np.searchsorted(rows, np.arange(n + 1, dtype=np.int64), side="left")
        else:
            indptr = np.zeros(n + 1, dtype=np.int64)
            cols = np.zeros(0, dtype=np.int32)
            summed = np.zeros(0, dtype=np.float64)

        return Connectome(
            neurons=neurons,
            indptr=indptr,
            indices=cols,
            weights=summed.astype(np.float32),
            source=source,
            meta=dict(meta or {}),
        )

    # ------------------------------------------------------------- querying

    def mask(
        self,
        cell_class: Optional[Union[str, Sequence[str]]] = None,
        super_class: Optional[Union[str, Sequence[str]]] = None,
        cell_type: Optional[Union[str, Sequence[str]]] = None,
        nt: Optional[Union[str, Sequence[str]]] = None,
    ) -> np.ndarray:
        """Boolean mask over neurons selecting the given annotation values.

        Within one field the listed values are OR-ed; across fields they are
        AND-ed. Unmatched masks are all-False rather than an error.
        """
        out = np.ones(self.n, dtype=bool)
        for column, wanted in (
            (self.neurons.cell_class, cell_class),
            (self.neurons.super_class, super_class),
            (self.neurons.cell_type, cell_type),
            (self.neurons.nt, nt),
        ):
            if wanted is None:
                continue
            values = (wanted,) if isinstance(wanted, str) else tuple(wanted)
            out &= np.isin(column, np.asarray(values, dtype=column.dtype))
        return out

    def subgraph(self, mask: np.ndarray) -> "Connectome":
        """Induced subgraph on ``mask``; edges crossing the boundary are cut."""
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != self.n:
            raise ValueError("mask length must equal neuron count")
        keep = np.flatnonzero(mask)
        remap = np.full(self.n, -1, dtype=np.int64)
        remap[keep] = np.arange(keep.shape[0], dtype=np.int64)
        if keep.size == 0:
            return Connectome.from_edgelist([], [], None, self.neurons.subtable(mask), self.source + "+sub", self.meta)
        edge_ok = (remap[self.row_of_nnz] >= 0) & (remap[self.indices] >= 0)
        new_pre = remap[self.row_of_nnz[edge_ok]]
        new_post = remap[self.indices[edge_ok]]
        new_w = self.weights[edge_ok]
        meta = dict(self.meta)
        meta["induced_from"] = self.source
        return Connectome.from_edgelist(
            new_pre, new_post, new_w, self.neurons.subtable(mask), self.source + "+sub", meta
        )

    def out_degree(self) -> np.ndarray:
        return np.diff(self.indptr).astype(np.int32)

    def in_degree(self) -> np.ndarray:
        return np.bincount(self.indices, minlength=self.n).astype(np.int32)

    def out_synapses(self) -> np.ndarray:
        return np.bincount(self.row_of_nnz, weights=self.weights, minlength=self.n)

    def in_synapses(self) -> np.ndarray:
        return np.bincount(self.indices, weights=self.weights, minlength=self.n)

    # ------------------------------------------------------------- dynamics hook

    def synaptic_drive(self, spikes: np.ndarray) -> np.ndarray:
        """Signed synaptic drive arriving at each neuron for one time step.

        ``spikes`` is a per-neuron activation (0/1 for spikes, or a rate in
        [0, 1]). Presynaptic transmitter sign is already folded into the
        weights, so inhibitory sources subtract.

        CSR rows are presynaptic, so the drive landing on neuron ``j`` is the
        sum over edges whose *target* is ``j`` -- that is a column sum, binned
        by ``indices`` rather than by ``row_of_nnz``.
        """
        x = np.asarray(spikes, dtype=np.float32)
        if x.shape[0] != self.n:
            raise ValueError("activation length must equal neuron count")
        contrib = self.signed_weights * x[self.row_of_nnz]
        return np.bincount(self.indices, weights=contrib.astype(np.float64), minlength=self.n)

    def to_dense(self, limit: int = 64) -> np.ndarray:
        """Dense signed matrix; refuses to blow up on real connectomes."""
        if self.n > limit:
            raise ValueError(f"refusing to densify {self.n} neurons (limit={limit})")
        dense = np.zeros((self.n, self.n), dtype=np.float32)
        for i in range(self.n):
            lo, hi = self.indptr[i], self.indptr[i + 1]
            dense[i, self.indices[lo:hi]] = self.signed_weights[lo:hi]
        return dense

    # ------------------------------------------------------------- persistence

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.neurons.to_dict()
        payload.update(
            {
                "indptr": self.indptr,
                "indices": self.indices,
                "weights": self.weights,
                "source": np.array(self.source),
                "meta": np.array(json.dumps(self.meta, default=str)),
            }
        )
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Connectome":
        with np.load(Path(path), allow_pickle=False) as data:
            neurons = NeuronTable.from_dict({k: data[k] for k in ("root_id", "cell_type", "cell_class", "super_class", "nt", "sign", "soma")})
            meta = json.loads(str(data["meta"]))
            return cls(
                neurons=neurons,
                indptr=data["indptr"],
                indices=data["indices"],
                weights=data["weights"],
                source=str(data["source"]),
                meta=meta,
            )

    # ------------------------------------------------------------- reporting

    def describe(self, top: int = 12) -> str:
        lines = [
            f"connectome      {self.n_neurons:,} neurons, {self.n_edges:,} connections, "
            f"{self.n_synapses:,.0f} synapses",
            f"source          {self.source}",
            f"csr footprint   {self.n_bytes / 1e6:.1f} MB",
        ]
        if self.meta:
            for k, v in sorted(self.meta.items()):
                lines.append(f"meta.{k:<10} {v}")
        out_deg = self.out_degree()
        in_deg = self.in_degree()
        lines.append(
            f"out-degree      mean {out_deg.mean():.1f}  p50 {np.median(out_deg):.0f}  max {out_deg.max() if self.n else 0}"
        )
        lines.append(
            f"in-degree       mean {in_deg.mean():.1f}  p50 {np.median(in_deg):.0f}  max {in_deg.max() if self.n else 0}"
        )
        lines.append(f"inhibitory      {(self.neurons.sign < 0).mean() * 100:.1f}% of neurons")
        lines.append("")
        lines.append(f"{'super_class':<22}{'neurons':>10}{'out synapses':>14}")
        for name, count in self.neurons.counts("super_class", limit=top):
            m = self.neurons.super_class == name
            lines.append(f"{name:<22}{count:>10,}{self.out_synapses()[m].sum():>14,.0f}")
        return "\n".join(lines)


# Backwards-compatible helper used by the CLI before a graph exists.
def empty_connectome(n: int = 0) -> Connectome:
    table = NeuronTable(np.zeros(n, np.int64), [], [], [], [])
    return Connectome.from_edgelist([], [], None, table, "empty")
