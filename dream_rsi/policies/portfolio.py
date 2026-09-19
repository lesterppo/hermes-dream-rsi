"""Adaptive portfolio policy (hand-written seed for the dreaming loop).

Implements the behaviour the improvement prompt asks for: trajectory-aware
ranking, a dynamic exploitation/exploration/recovery portfolio, portfolio-level
stopping, and one beta-routed threshold schedule.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Set

from dream_rsi.grid import GridPlan, GridPlanningContext
from dream_rsi.policy_api import (LLMDesignedMethod, branch_failed_hard,
                          probe_improved_vs_parent)

REPAIRABLE = {"invalid", "runtime", "compile", "timeout"}


class PortfolioPolicy(LLMDesignedMethod):
    NAME = "OptimalPolicy"

    def _schedule(self, beta: float) -> Dict[str, Any]:
        return {
            "beta": beta,
            # high beta -> wider, more patient, weaker pruning
            "explore_quota": 1 + int(round(2 * beta)),
            "recovery_quota": 1,
            "reserve_threshold": 0.40 - 0.30 * beta,
            "stagnation_patience": int(round(2 + 3 * beta)),
            "prune_ratio": 0.55 + 0.35 * beta,
            "underexplored_depth": int(round(1 + 2 * beta)),
        }

    def plan_grid(self, context: GridPlanningContext) -> GridPlan:
        hist = context.history or []
        if not hist:
            return GridPlan(context.fallback_branch_count,
                            context.fallback_refine_count,
                            "no live history yet: conservative bootstrap plan")
        gains = []
        for h in hist:
            best = h.get("final_best")
            base = h.get("baseline") or 0.0
            gains.append(0.0 if best is None else float(best) - float(base))
        if len(gains) >= 2 and gains[-1] <= gains[-2] * 1.01:
            w = min(context.hard_max_branch_count,
                    max(1, round(context.fallback_branch_count * 1.5)))
            r = context.fallback_refine_count
            reason = "live best plateaued: widen to uncover new directions"
        elif len(gains) >= 2 and gains[-1] > gains[-2]:
            w = context.fallback_branch_count
            r = min(context.hard_max_refine_count,
                    context.fallback_refine_count + 1)
            reason = "gains still arriving: hold width, deepen refinements"
        else:
            w, r = context.fallback_branch_count, context.fallback_refine_count
            reason = "insufficient history: bootstrap at fallback caps"
        return GridPlan(w, r, reason)

    # ------------------------------------------------------------- decisions
    def select_batch(self, question: Any) -> List[str]:
        sched = self._schedule(self.beta)
        W = question.max_parallelism
        prefix = question.observed()
        legal: Set[str] = set(question.legal_actions())
        if not legal:
            return []

        by_branch: Dict[int, List[Any]] = {}
        for obs in prefix.values():
            by_branch.setdefault(obs.branch, []).append(obs)
        for b in by_branch:
            by_branch[b].sort(key=lambda o: o.attempt)

        opened = question.opened_branches()
        closed: List[int] = []

        # 1. trajectory reconstruction + pruning (cumulative evidence only)
        trends: Dict[int, float] = {}
        for b, obs in by_branch.items():
            best = max((o.score for o in obs if o.evaluated), default=None)
            last = obs[-1]
            if best is None:
                trends[b] = -math.inf
                continue
            trend = (last.score - best) / max(1e-9, abs(best)) if last.evaluated else -1.0
            trends[b] = trend
            deep_enough = len(obs) >= sched["stagnation_patience"]
            if (deep_enough and trend <= -sched["prune_ratio"]
                    and branch_failed_hard(obs, b)):
                closed.append(b)

        for b in closed:
            question.close_branch(int(b))
        legal = set(question.legal_actions())
        if not legal:
            return []

        def frontier(b: int) -> List[str]:
            return sorted(c for c in legal if c.startswith(f"b{b}#"))

        # 2. rank frontiers: exploitation first, underexplored next, recovery last
        exploit, explore, recovery = [], [], []
        for b in opened:
            cells = frontier(b)
            if not cells:
                continue
            cell = cells[0]
            obs = by_branch.get(b, [])
            if not obs:
                explore.append((0.0, cell))
                continue
            last = obs[-1]
            anchor = max((o.score for o in obs if o.evaluated), default=None)
            delta = (last.score - anchor) if (anchor is not None and last.evaluated) else -1.0
            repairable = (not last.evaluated) or last.fail_class in REPAIRABLE or \
                (last.n_valid == 0)
            underexplored = len(obs) <= sched["underexplored_depth"]
            if repairable and probe_improved_vs_parent(last, eps=-1e9) is False and anchor:
                recovery.append((delta, cell))
            elif delta > -sched["reserve_threshold"]:
                exploit.append((delta, cell))
            elif underexplored:
                explore.append((delta, cell))

        batch: List[str] = []
        exploit.sort(key=lambda p: -p[0])
        explore.sort(key=lambda p: -p[0])
        recovery.sort(key=lambda p: -p[0])

        for _delta, cell in exploit:
            if len(batch) >= W:
                break
            batch.append(cell)
        recovery_slots = min(sched["recovery_quota"], len(recovery))
        for _delta, cell in recovery[:recovery_slots]:
            if len(batch) < W and cell not in batch:
                batch.append(cell)
        quota = sched["explore_quota"]
        for _delta, cell in explore:
            if len(batch) >= W or quota <= 0:
                break
            if cell not in batch:
                batch.append(cell)
                quota -= 1
        roots = sorted(c for c in legal if c.endswith("#a0") and c not in batch)
        for cell in roots:
            if len(batch) >= W or quota <= 0:
                break
            batch.append(cell)
            quota -= 1

        # 3. never leave workers idle while legal, unopened work remains
        for cell in sorted(legal):
            if len(batch) >= W:
                break
            if cell in batch:
                continue
            b = int(cell.split("#")[0][1:])
            if any(o.startswith(f"b{b}#") for o in batch):
                continue
            batch.append(cell)
        return batch[:W]


NAME = PortfolioPolicy.NAME
