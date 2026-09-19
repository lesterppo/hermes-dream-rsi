"""Runnable start page plus import-path bootstrap.

Drops the distribution root on ``sys.path`` so LLM-written policies that do
``from see.policy.api import ...`` (the paper's import path) resolve, then
re-exports the public surface.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

__version__ = "1.0.0"
__all__ = ["tree", "grid", "simulator", "policy_api", "agents", "tasks",
           "online", "dream", "loop", "traces"]
