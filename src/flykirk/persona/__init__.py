"""Persona layer: turning a brain state into a debating register."""

from .kirk import (
    DEFAULT_REGISTER,
    REGISTERS,
    PersonaStyle,
    build_system_prompt,
    register_for,
    stage_directions,
)

__all__ = [
    "PersonaStyle",
    "DEFAULT_REGISTER",
    "REGISTERS",
    "register_for",
    "build_system_prompt",
    "stage_directions",
]
