"""Grid planning (Appendix B.2 "Required next-cycle grid planning").

`plan_grid` runs *before* a new live grid is created.  It never inspects a
current episode's outcomes; it may only use prefix-safe facts carried in
GridPlanningContext (completed earlier live manifests, worker/hard caps and the
replay structural support fields).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class GridPlan:
    branch_count: int
    refine_count: int
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def clamp(self, ctx: "GridPlanningContext") -> "GridPlan":
        w = max(1, min(int(self.branch_count), int(ctx.hard_max_branch_count),
                       int(ctx.worker_cap) or 1))
        r = max(0, min(int(self.refine_count), int(ctx.hard_max_refine_count)))
        return GridPlan(w, r, self.reason)


@dataclass
class GridPlanningContext:
    """Prefix-safe facts available to plan_grid."""

    history: List[Dict[str, Any]] = field(default_factory=list)
    worker_cap: int = 4
    fallback_branch_count: int = 4
    fallback_refine_count: int = 3
    hard_max_branch_count: int = 32
    hard_max_refine_count: int = 16
    trace_branch_count: Optional[int] = None
    trace_refine_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def in_replay_support(self, plan: GridPlan) -> bool:
        """Replay cannot reward a plan wider/deeper than the frozen trace."""
        if self.trace_branch_count is None or self.trace_refine_count is None:
            return True
        return (plan.branch_count <= self.trace_branch_count
                and plan.refine_count <= self.trace_refine_count)
