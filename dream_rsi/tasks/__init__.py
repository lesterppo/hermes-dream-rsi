from __future__ import annotations

from typing import Any, Dict, List

from .base import EvalOutcome, Task
from .circle_packing import CirclePackingTask
from .lasso_path import LassoPathTask

TASKS: Dict[str, Any] = {
    "circle_packing": CirclePackingTask,
    "lasso_path": LassoPathTask,
}


def build_task(spec: str, **kwargs: Any) -> Task:
    """``circle_packing``, ``circle_packing:n=12``, ``lasso_path:n=900,p=240``."""
    if spec in TASKS:
        return TASKS[spec](**kwargs)
    base, _, rest = spec.partition(":")
    if base not in TASKS:
        raise KeyError(f"unknown task {base!r}; known: {sorted(TASKS)}")
    kw: Dict[str, Any] = dict(kwargs)
    for chunk in rest.split(","):
        if not chunk.strip():
            continue
        key, _, val = chunk.partition("=")
        kw[key.strip()] = int(val)
    return TASKS[base](**kw)


def task_names() -> List[str]:
    return sorted(TASKS)


__all__ = ["Task", "EvalOutcome", "build_task", "task_names",
           "CirclePackingTask", "LassoPathTask", "TASKS"]
