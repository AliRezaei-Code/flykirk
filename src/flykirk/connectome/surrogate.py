"""Structurally plausible surrogate connectome for offline development.

This is **not** the FlyWire connectome. It is a degree-controlled block model
that reproduces the coarse statistics that matter for the dynamics: the fly's
superclass composition, its transmitter mix, its sparsity, and its heavy-tailed
synapse counts. Every artifact it produces is tagged
``provenance="synthetic-surrogate"`` so it can never be mistaken for real data.

Run ``flykirk fetch annotations`` and ``flykirk fetch connectome --source zenodo``
for the real thing.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .graph import Connectome, NeuronTable

__all__ = ["surrogate_connectome", "PROVENANCE"]

PROVENANCE = "synthetic-surrogate"

#: (super_class, cell_class, relative share, transmitter).
#: Shares are illustrative of FlyWire's superclass composition, not measured.
BLOCKS: Tuple[Tuple[str, str, float, str], ...] = (
    ("sensory", "photoreceptor", 0.190, "histamine"),
    ("sensory", "ORN", 0.045, "acetylcholine"),
    ("sensory", "GRN", 0.030, "acetylcholine"),
    ("sensory", "JO mechanosensory", 0.055, "acetylcholine"),
    ("visual_projection", "T4/T5", 0.070, "acetylcholine"),
    ("visual_projection", "T2/T3", 0.040, "acetylcholine"),
    ("visual_centrifugal", "CT1", 0.020, "acetylcholine"),
    ("ascending", "AN", 0.045, "acetylcholine"),
    ("central", "Kenyon cell", 0.075, "acetylcholine"),
    ("central", "MBON", 0.010, "acetylcholine"),
    ("central", "DAN", 0.008, "dopamine"),
    ("central", "PAM", 0.008, "acetylcholine"),
    ("central", "APL", 0.002, "gaba"),
    ("central", "ALPN", 0.020, "acetylcholine"),
    ("central", "LN", 0.015, "gaba"),
    ("central", "CX PB/EB/NO", 0.055, "acetylcholine"),
    ("central", "optic lobe intrinsic", 0.140, "acetylcholine"),
    ("central", "lateral horn", 0.010, "acetylcholine"),
    ("descending", "DN", 0.022, "acetylcholine"),
    ("descending", "DN modulatory", 0.004, "octopamine"),
    ("endocrine", "ring gland / CC", 0.003, "acetylcholine"),
    ("motor", "motor neuron", 0.020, "glutamate"),
)

#: (pre superclass, post superclass) -> anatomical connection probability.
SUPER_RULES: Dict[Tuple[str, str], float] = {
    ("sensory", "central"): 0.0055,
    ("sensory", "visual_projection"): 0.0060,
    ("sensory", "descending"): 0.0016,
    ("sensory", "ascending"): 0.0020,
    ("visual_projection", "central"): 0.0045,
    ("visual_projection", "descending"): 0.0022,
    ("visual_projection", "visual_centrifugal"): 0.0018,
    ("visual_centrifugal", "visual_projection"): 0.0040,
    ("ascending", "central"): 0.0040,
    ("ascending", "descending"): 0.0020,
    ("central", "central"): 0.0016,
    ("central", "descending"): 0.0055,
    ("central", "endocrine"): 0.0045,
    ("central", "ascending"): 0.0012,
    ("central", "motor"): 0.0008,
    ("descending", "central"): 0.0030,
    ("descending", "motor"): 0.0060,
    ("descending", "endocrine"): 0.0040,
    ("endocrine", "central"): 0.0025,
    ("motor", "central"): 0.0006,
}

#: Class-level overrides where the block model's defaults would be obviously wrong.
CLASS_RULES: Dict[Tuple[str, str], float] = {
    ("Kenyon cell", "MBON"): 0.0100,
    ("Kenyon cell", "APL"): 0.0300,
    ("APL", "Kenyon cell"): 0.1200,
    ("DAN", "Kenyon cell"): 0.0060,
    ("PAM", "Kenyon cell"): 0.0060,
    ("ALPN", "Kenyon cell"): 0.0080,
    ("ALPN", "lateral horn"): 0.0060,
    ("ORN", "ALPN"): 0.0600,
    ("ORN", "LN"): 0.0200,
    ("LN", "ALPN"): 0.0400,
    ("CX PB/EB/NO", "DN"): 0.0120,
    ("CX PB/EB/NO", "CX PB/EB/NO"): 0.0045,
    ("MBON", "CX PB/EB/NO"): 0.0060,
    ("DN", "DN"): 0.0020,
}

#: Mean synapse count per connection, by presynaptic cell class. Where a class
#: is absent the global mean is used. Heavy-tailed by design: the fly's wiring
#: is dominated by a small number of very strong connections.
SYNAPSE_MEAN: Dict[str, float] = {
    "photoreceptor": 9.0,
    "ORN": 7.0,
    "GRN": 6.0,
    "JO mechanosensory": 5.0,
    "Kenyon cell": 3.5,
    "MBON": 28.0,
    "APL": 22.0,
    "DAN": 6.0,
    "PAM": 6.0,
    "ALPN": 14.0,
    "LN": 12.0,
    "CX PB/EB/NO": 16.0,
    "DN": 24.0,
    "T4/T5": 11.0,
    "T2/T3": 9.0,
    "optic lobe intrinsic": 8.0,
    "lateral horn": 12.0,
    "motor neuron": 30.0,
}
DEFAULT_SYNAPSE_MEAN = 8.0

#: Multiplies every connection probability. The FlyWire brain averages roughly
#: 36 anatomical connections per neuron (about 360 synapses); at the reduced
#: neuron counts flykirk runs at by default, matching *density* instead of
#: *degree* would leave the graph too sparse for activity to spread more than a
#: synapse or two. This multiplier restores a plausible degree.
DENSITY = 3.0


def _allocate(weights: Sequence[float], total: int) -> List[int]:
    """Largest-remainder apportionment so block counts sum to exactly ``total``."""
    raw = np.asarray(weights, dtype=np.float64)
    raw = raw / raw.sum() * total
    base = np.floor(raw).astype(np.int64)
    remainder = total - int(base.sum())
    if remainder > 0:
        order = np.argsort(-(raw - base))
        base[order[:remainder]] += 1
    return [int(x) for x in base]


def _rule(pre_super: str, pre_class: str, post_super: str, post_class: str) -> float:
    if (pre_class, post_class) in CLASS_RULES:
        return min(CLASS_RULES[(pre_class, post_class)] * DENSITY, 0.35)
    base = SUPER_RULES.get((pre_super, post_super), 0.0)
    return min(base * DENSITY, 0.35)


def surrogate_connectome(
    n_neurons: int = 4000,
    seed: int = 7,
    max_edges: int = 2_000_000,
    synapse_scale: float = 1.0,
) -> Connectome:
    """Build a block-model connectome with ``n_neurons`` nodes.

    ``n_neurons`` defaults to a size that runs a full debate in seconds on a
    laptop CPU. Pass 139255 to model the whole FlyWire brain -- at that size the
    edge budget, not the neuron count, is the binding constraint.
    """
    if n_neurons < len(BLOCKS):
        raise ValueError(f"n_neurons must be at least {len(BLOCKS)}")
    rng = np.random.default_rng(seed)

    counts = _allocate([b[2] for b in BLOCKS], n_neurons)
    super_class: List[str] = []
    cell_class: List[str] = []
    nt: List[str] = []
    for block_id, ((sup, cls, _share, transmitter), count) in enumerate(zip(BLOCKS, counts)):
        super_class.extend([sup] * count)
        cell_class.extend([cls] * count)
        nt.extend([transmitter] * count)

    # Root ids are negative so a surrogate neuron can never collide with a
    # FlyWire root id in a downstream join.
    root_id = -np.arange(1, n_neurons + 1, dtype=np.int64)
    table = NeuronTable(
        root_id=root_id,
        cell_type=np.array(cell_class, dtype=object),
        cell_class=np.array(cell_class, dtype=object),
        super_class=np.array(super_class, dtype=object),
        nt=np.array(nt, dtype=object),
    )

    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    block_slice = [(int(starts[b]), int(starts[b] + counts[b])) for b in range(len(BLOCKS))]

    pre_parts: List[np.ndarray] = []
    post_parts: List[np.ndarray] = []
    weight_parts: List[np.ndarray] = []
    budget = max_edges
    for pre_block, (pre_sup, pre_cls, _s, _t) in enumerate(BLOCKS):
        lo_pre, hi_pre = block_slice[pre_block]
        n_pre = hi_pre - lo_pre
        if n_pre <= 0:
            continue
        for post_block, (post_sup, post_cls, _s2, _t2) in enumerate(BLOCKS):
            if budget <= 0:
                break
            lo_post, hi_post = block_slice[post_block]
            n_post = hi_post - lo_post
            if n_post <= 0:
                continue
            p = _rule(pre_sup, pre_cls, post_sup, post_cls)
            if p <= 0.0:
                continue
            k = int(round(p * n_pre * n_post))
            if k <= 0:
                continue
            # Cap fan-out per postsynaptic neuron; a real connectome is sparse
            # and a block model left uncapped produces star-shaped targets.
            k = min(k, 40 * n_post, budget)
            pre_idx = rng.integers(lo_pre, hi_pre, size=k, endpoint=False)
            post_idx = rng.integers(lo_post, hi_post, size=k, endpoint=False)
            mean = SYNAPSE_MEAN.get(pre_cls, DEFAULT_SYNAPSE_MEAN) * synapse_scale
            syn = np.clip(
                np.rint(rng.lognormal(mean=np.log(max(mean, 0.5)), sigma=1.05, size=k)), 1, 400
            ).astype(np.float32)
            pre_parts.append(pre_idx)
            post_parts.append(post_idx)
            weight_parts.append(syn)
            budget -= k
        if budget <= 0:
            break

    if pre_parts:
        pre = np.concatenate(pre_parts)
        post = np.concatenate(post_parts)
        weight = np.concatenate(weight_parts)
    else:  # pragma: no cover - only reachable with a degenerate rule table
        pre = post = weight = np.zeros(0, dtype=np.int64)

    return Connectome.from_edgelist(
        pre,
        post,
        weight,
        table,
        source=PROVENANCE,
        meta={
            "provenance": PROVENANCE,
            "seed": seed,
            "n_neurons": n_neurons,
            "blocks": len(BLOCKS),
            "warning": "synthetic block model, not FlyWire",
            "edge_budget_hit": bool(budget <= 0),
        },
    )
