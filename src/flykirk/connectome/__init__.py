"""Connectome layer: the wiring diagram the brain runs on."""

from .graph import (
    EXCITATORY,
    INHIBITORY,
    MODULATORY,
    Connectome,
    NeuronTable,
    normalize_nt,
)
from .surrogate import surrogate_connectome

__all__ = [
    "Connectome",
    "NeuronTable",
    "normalize_nt",
    "surrogate_connectome",
    "EXCITATORY",
    "INHIBITORY",
    "MODULATORY",
]
