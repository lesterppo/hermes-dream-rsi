from __future__ import annotations

from typing import Any, Dict, List

from .base import EvalOutcome, Task
from .circle_packing import CirclePackingTask
from .hermes_hotpath import HermesHotPathTask
from .lasso_path import LassoPathTask

TASKS: Dict[str, Any] = {
    "circle_packing": CirclePackingTask,
    "lasso_path": LassoPathTask,
    "hermes_hotpath": HermesHotPathTask,
}


def build_task(spec: str, **kwargs: Any) -> Task:
    """``circle_packing``, ``circle_packing:n=12``, ``lasso_path:n=900,p=240``."""
    if spec in TASKS:
        return TASKS[spec](**kwargs)
    base, _, rest = spec.partition(":")
    if base not in TASKS:
        raise KeyError(f"unknown task {base!r}; known: {sorted(TASKS)}")
    kw: Dict[str, Any] = dict(kwargs)
    # split on commas, but a comma escaped as \, belongs to the value
    # (spec strings travel through shells, so quoting cannot be relied on)
    chunks, buf, i = [], "", 0
    while i < len(rest):
        ch = rest[i]
        if ch == "\\" and i + 1 < len(rest):
            buf += rest[i + 1]
            i += 2
            continue
        if ch == ",":
            chunks.append(buf)
            buf = ""
        else:
            buf += ch
        i += 1
    chunks.append(buf)
    for chunk in chunks:
        if not chunk.strip():
            continue
        key, _, val = chunk.partition("=")
        val = val.strip()
        # numeric parameters stay numbers; everything else (paths, names) is text
        if val.lstrip("-").isdigit():
            kw[key.strip()] = int(val)
        elif val.lower() in ("true", "false"):
            kw[key.strip()] = val.lower() == "true"
        else:
            kw[key.strip()] = val
    return TASKS[base](**kw)


def task_names() -> List[str]:
    return sorted(TASKS)


__all__ = ["Task", "EvalOutcome", "build_task", "task_names",
           "CirclePackingTask", "LassoPathTask", "HermesHotPathTask", "TASKS"]
