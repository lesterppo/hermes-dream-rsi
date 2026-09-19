"""Baseline policy: parallel refining (the paper's controlled baseline).

Paper Sec. 4: "it launches multiple independent exploration workspaces in
parallel, with each workspace maintaining its own local discovery trajectory and
repeatedly refining its current candidate based on the history accumulated within
that workspace."  Its policy is fixed across recursive rounds; Dream-RSI starts
from exactly this policy in round 1 and then improves it by dreaming.
"""
from __future__ import annotations

from typing import Any, Dict, List

from dream_rsi.grid import GridPlan, GridPlanningContext
from dream_rsi.policy_api import LLMDesignedMethod


class ParallelRefinePolicy(LLMDesignedMethod):
    NAME = "ParallelRefine"

    def _schedule(self, beta: float) -> Dict[str, Any]:
        sched = super()._schedule(beta)
        sched["roots_per_wave"] = max(1, int(round(1 + 3 * beta)))
        return sched

    def plan_grid(self, context: GridPlanningContext) -> GridPlan:
        w = int(self.config.get("branch_count", 0)) or context.fallback_branch_count
        r = int(self.config.get("refine_count", 0))
        if r == 0 and "refine_count" not in self.config:
            r = context.fallback_refine_count
        return GridPlan(w, r, "parallel-refine baseline: fixed width/depth floor")

    def select_batch(self, question: Any) -> List[str]:
        W = question.max_parallelism
        sched = self._schedule(self.beta)
        prefix = question.observed()
        opened = question.opened_branches()
        legal = set(question.legal_actions())
        batch: List[str] = []

        roots = [c for c in question.legal_roots() if c in legal]
        if not prefix:
            # first wave: launch several independent workspaces in parallel
            batch.extend(roots[:W])
        elif roots:
            batch.extend(roots[:min(W, sched["roots_per_wave"])])

        for b in opened:
            if len(batch) >= W:
                break
            frontier = [c for c in legal
                        if c.startswith(f"b{b}#") and c not in batch]
            if frontier:
                batch.append(sorted(frontier)[0])

        for r in roots:
            if len(batch) >= W:
                break
            if r not in batch:
                batch.append(r)
        return batch[:W]


NAME = ParallelRefinePolicy.NAME
