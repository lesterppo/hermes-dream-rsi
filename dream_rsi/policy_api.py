"""Policy API surface (the paper's ``see.policy.api``).

Policies are LLM-designed, executable modules that implement::

    NAME = "OptimalPolicy"

    class OptimalPolicy(LLMDesignedMethod):
        def solve(self, question, budget=None): ...
        def plan_grid(self, context) -> GridPlan: ...

Both the live runner and the replay simulator drive policies through the same
``select_batch`` decision interface, so a policy that works online replays
offline unchanged.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from .grid import GridPlan, GridPlanningContext
from .simulator import (  # re-exported for policy modules
    CellMeta, IllegalBatch, Observation, SimResult, _budget_done, _record_curve,
    branch_failed_hard, branch_promising, finalize_result,
    probe_improved_vs_baseline, probe_improved_vs_parent, run_replay_episode,
)


class LLMDesignedMethod:
    """Base class for an executable exploration policy.

    Subclasses implement ``plan_grid`` (how much grid to create next cycle) and
    ``select_batch`` (the within-episode decision).  ``solve`` is the paper-level
    entry point for replay; the runner calls ``select_batch`` directly so the
    same code drives live and replayed episodes.
    """

    NAME = "LLMDesignedMethod"
    replay_round_limit = 64

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = dict(config or {})
        self.beta = float(self.config.get("beta", 0.6))
        self.max_parallelism = int(self.config.get("max_parallelism", 4))
        self._schedule_cache: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------ to override
    def _schedule(self, beta: float) -> Dict[str, Any]:
        """Route every behavioural threshold through one beta schedule."""
        return {
            "beta": beta,
            # high beta = more width, deeper patience, weaker pruning
            "reserve_threshold": 0.5 - 0.4 * beta,
            "recovery_min_gap": 0.35 - 0.25 * beta,
            "stagnation_patience": int(round(2 + 4 * beta)),
            "prune_ratio": 0.6 + 0.3 * beta,
            "max_roots_per_batch": max(1, int(round(1 + 2 * beta))),
        }

    def plan_grid(self, context: GridPlanningContext) -> GridPlan:
        raise NotImplementedError

    def select_batch(self, question: Any) -> List[str]:
        raise NotImplementedError

    # --------------------------------------------------- paper-level entry pt
    def solve(self, question: Any, budget: Optional[int] = None) -> SimResult:
        question.reset()
        res, closed = SimResult(), set()
        while not _budget_done(question, budget):
            prefix = question.observed()
            self.update_closed(closed, prefix, question)
            batch = self.select_batch(question)
            if not batch:
                break
            question.probe_batch(batch,
                                 on_reveal=lambda _o: _record_curve(res, question))
        return finalize_result(question, res)

    def update_closed(self, closed: set, prefix: Dict[str, Observation],
                      question: Any) -> None:
        """Optional prefix-only hook: retire hopeless branches."""

    # ---------------------------------------------------------------- helpers
    def reset(self) -> None:
        self._schedule_cache = None


def schedule_for(policy: LLMDesignedMethod) -> Dict[str, Any]:
    return policy._schedule(policy.beta)  # noqa: SLF001
