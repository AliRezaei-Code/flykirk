"""Filesystem locations for flykirk data products.

Everything large and downloaded lives outside the repo. Resolution order:

1. ``$FLYKIRK_DATA``
2. ``$XDG_CACHE_HOME/flykirk``
3. ``~/.cache/flykirk``
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["data_dir", "cache_dir", "ensure_dir"]


def cache_dir() -> Path:
    override = os.environ.get("FLYKIRK_DATA")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "flykirk"


#: Alias kept for readability at call sites that mean "the big downloaded stuff".
data_dir = cache_dir


def ensure_dir(path: os.PathLike | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
