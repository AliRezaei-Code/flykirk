"""flykirk -- a connectome-driven fly brain that argues like a campus debater.

The pipeline, end to end:

    opponent text
        -> SensoryEncoder          (text becomes sensory-neuron drive)
        -> LIFNetwork              (spiking spread over a FlyWire-derived graph)
        -> NeuromodulatorSystem    (octopamine / dopamine / serotonin state)
        -> Readout                 (descending-neuron population -> Telemetry)
        -> PersonaStyle + Sampling (telemetry becomes register and sampler knobs)
        -> local Liquid AI SLM     (LFM2 / LFM2.5 served by Ollama or llama.cpp)
        -> a fly that interrupts itself

Everything above the language model is a real dynamical system. The language
model is the only part that knows any English. This is a parody: see README.
"""

__version__ = "0.1.0"

from .connectome import Connectome, NeuronTable
from .brain import BrainSim, Telemetry
from .persona import PersonaStyle, register_for

__all__ = [
    "__version__",
    "Connectome",
    "NeuronTable",
    "BrainSim",
    "Telemetry",
    "PersonaStyle",
    "register_for",
]
