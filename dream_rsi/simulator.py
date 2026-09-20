"""Replay simulator: history as a world (Dream-RSI Sec. 3 "Offline evaluation").

A completed discovery tree is a frozen, irregular branch x attempt grid.  A
policy opens a root or refines the next cell of an already-open branch; each
revealed cell costs one probe.  The policy sees only the cells it has revealed
(prefix-only); unrevealed scores are unreachable through this API.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .grid import GridPlan, GridPlanningContext
from .tree import DiscoveryTree, Node


class IllegalBatch(ValueError):
    """Policy selected a batch that violates the decision interface."""


@dataclass(frozen=True)
class Observation:
    cell: str
    branch: int
    attempt: int
    parent_id: Optional[str] = None
    score: float = -math.inf
    evaluated: bool = True
    valid: bool = True
    fail_class: str = "ok"
    error: Optional[str] = None
    delta_vs_baseline: float = 0.0
    delta_vs_parent: float = 0.0
    n_valid: Optional[int] = None
    n_total: Optional[int] = None
    revealed_at_round: int = 0


@dataclass(frozen=True)
class CellMeta:
    cell: str
    branch: int
    attempt: int
    parent_id: Optional[str] = None
    seq: int = 0
    tags: Tuple[str, ...] = ()


@dataclass
class SimResult:
    """Replay outcome for one (policy, world, beta) episode."""

    curve: List[Tuple[int, float, int]] = field(default_factory=list)  # (probes,best,rounds)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    probes: int = 0
    decision_rounds: int = 0
    effective_sequential_rounds: float = 0.0
    best_score: float = -math.inf
    revealed: List[str] = field(default_factory=list)
    terminated_by: str = "unknown"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probes": self.probes,
            "decision_rounds": self.decision_rounds,
            "effective_sequential_rounds": round(self.effective_sequential_rounds, 4),
            "best_score": (None if self.best_score == -math.inf
                           else round(self.best_score, 6)),
            "revealed": len(self.revealed),
            "terminated_by": self.terminated_by,
            "curve": [[int(p), (None if b == -math.inf else round(b, 6)), int(k)]
                      for p, b, k in self.curve],
            "error": self.error,
        }


# --------------------------------------------------------------------------- #
# policy-facing helper functions (mirror see.policy.api helpers)               #
# --------------------------------------------------------------------------- #
def _record_curve(res: SimResult, question: "ReplayQuestion") -> None:
    res.curve.append((question.probes_spent, question.best_so_far,
                      question.decision_rounds))


def _budget_done(question: "ReplayQuestion", budget: Optional[int]) -> bool:
    if budget is None:
        return False
    return question.probes_spent >= int(budget)


def finalize_result(question: "ReplayQuestion", res: SimResult) -> SimResult:
    res.probes = question.probes_spent
    res.decision_rounds = question.decision_rounds
    res.effective_sequential_rounds = question.effective_sequential_rounds
    res.best_score = question.best_so_far
    res.revealed = list(question.revealed_order)
    if not res.curve:
        _record_curve(res, question)
    return res


def branch_promising(observations: Sequence[Observation], branch: int,
                     eps: float = 0.0) -> bool:
    obs = [o for o in observations if o.branch == branch and o.evaluated]
    if not obs:
        return False
    return any(o.delta_vs_baseline > eps or o.delta_vs_parent > eps for o in obs)


def branch_failed_hard(observations: Sequence[Observation], branch: int) -> bool:
    obs = [o for o in observations if o.branch == branch]
    if not obs:
        return False
    hard = {"compile", "runtime", "timeout", "env", "invalid"}
    return all((not o.evaluated) or o.fail_class in hard for o in obs) and \
        all(o.n_valid in (None, 0) for o in obs)


def probe_improved_vs_parent(obs: Observation, eps: float = 0.0) -> bool:
    return obs.evaluated and obs.delta_vs_parent > eps


def probe_improved_vs_baseline(obs: Observation, eps: float = 0.0) -> bool:
    return obs.evaluated and obs.delta_vs_baseline > eps


# --------------------------------------------------------------------------- #
# replay environment                                                          #
# --------------------------------------------------------------------------- #
class ReplayQuestion:
    """Prefix-only, replayable view of one frozen discovery tree.

    Legal action ids are grid cells ``b<branch>#a<attempt>``.  Opening a root
    cell reveals that branch's recorded first attempt; refining the frontier of
    an opened branch reveals its next recorded attempt.  Cells the recording
    does not contain are exhausted: probing them reveals nothing.
    """

    def __init__(self, trace: DiscoveryTree, max_parallelism: int = 4,
                 round_limit: int = 64) -> None:
        self._trace = trace                      # private: never exposed
        self.max_parallelism = max(1, int(max_parallelism))
        self.round_limit = int(round_limit)
        self.baseline_score = float(trace.baseline_score)
        self._seed = 0

        # episode state
        self._observed: Dict[str, Observation] = {}
        self._opened: List[int] = []             # branch creation order
        self._depth: Dict[int, int] = {}         # branch -> deepest revealed attempt
        self._exhausted: set = set()
        # bookkeeping only (policies must not use these to decide)
        self.probes_spent = 0
        self.decision_rounds = 0
        self.effective_sequential_rounds = 0.0
        self.revealed_order: List[str] = []
        self._closed_branches: set = set()
        self.reset()

    # ------------------------------------------------------------------ API
    def reset(self) -> None:
        self._observed = {}
        self._opened = []
        self._depth = {}
        self._exhausted = set()
        self.probes_spent = 0
        self.decision_rounds = 0
        self.effective_sequential_rounds = 0.0
        self.revealed_order = []

    def observed(self) -> Dict[str, Observation]:
        """Revealed prefix only."""
        return dict(self._observed)

    def legal_roots(self) -> List[str]:
        return [f"b{b}#a0" for b in self._all_branches()
                if b not in self._opened and b not in self._closed_branches]

    def opened_branches(self) -> List[int]:
        return list(self._opened)

    def legal_actions(self) -> List[str]:
        acts = list(self.legal_roots())
        for b in self._opened:
            if b in self._closed_branches:
                continue
            nxt = self._depth.get(b, -1) + 1
            cell = f"b{b}#a{nxt}"
            if cell in self._exhausted:
                continue
            # irregular grid: the recorded trace may lack the cell.  Probe it and
            # the reveal returns nothing (the branch is exhausted at that depth).
            acts.append(cell)
        return acts

    def meta(self, cell: str) -> CellMeta:
        node = self._trace.get(cell)
        b, a = _parse_cell(cell)
        if node is None:
            return CellMeta(cell=cell, branch=b, attempt=a)
        return CellMeta(cell=node.id, branch=node.branch, attempt=node.attempt,
                        parent_id=node.parent_id, seq=node.seq,
                        tags=tuple(node.tags))

    def close_branch(self, branch: int) -> None:
        """Explicitly retire a branch for the rest of this episode."""
        self._closed_branches.add(int(branch))

    def probe_batch(self, cells: Sequence[str],
                    on_reveal: Optional[Callable[[Observation], None]] = None
                    ) -> List[Observation]:
        cells = list(cells)
        self._validate(cells)
        out: List[Observation] = []
        for cell in cells:
            b, a = _parse_cell(cell)
            if a == 0 and b not in self._opened:
                self._opened.append(b)
            node = self._trace.get(cell)
            if node is None:
                self._exhausted.add(cell)
                continue
            obs = self._reveal(node)
            out.append(obs)
            if on_reveal is not None:
                on_reveal(obs)
        self.decision_rounds += 1
        self.effective_sequential_rounds += math.ceil(max(1, len(cells))
                                                      / self.max_parallelism)
        return out

    def remaining_actions(self) -> int:
        return len(self.legal_actions())

    def is_exhausted(self, cell: str) -> bool:
        return cell in self._exhausted

    def is_complete(self) -> bool:
        """All recorded nodes revealed."""
        return len(self.revealed_order) >= self._trace.size()

    # -------------------------------------------------------------- internals
    @property
    def best_so_far(self) -> float:
        if not self._observed:
            return -math.inf
        return max(o.score for o in self._observed.values() if o.evaluated)

    def _all_branches(self) -> List[int]:
        return sorted({n.branch for n in self._trace.nodes.values()})

    def _validate(self, cells: Sequence[str]) -> None:
        if len(cells) > self.max_parallelism:
            raise IllegalBatch(
                f"batch size {len(cells)} exceeds max_parallelism {self.max_parallelism}")
        if len(set(cells)) != len(cells):
            raise IllegalBatch("duplicate cell ids in batch")
        legal = set(self.legal_actions())
        for c in cells:
            if c not in legal:
                raise IllegalBatch(f"illegal cell {c}")
        for c in cells:
            b, a = _parse_cell(c)
            for other in cells:
                if other == c:
                    continue
                ob, oa = _parse_cell(other)
                if ob == b and abs(oa - a) == 1:
                    raise IllegalBatch(
                        f"batch contains parent and child together: {c}, {other}")

    def _reveal(self, node: Node) -> Observation:
        parent_score = math.inf
        if node.parent_id:
            p = self._observed.get(node.parent_id)
            if p is not None:
                parent_score = p.score
        delta_parent = (node.score - parent_score) if parent_score != math.inf else 0.0
        obs = Observation(
            cell=node.id, branch=node.branch, attempt=node.attempt,
            parent_id=node.parent_id, score=float(node.score),
            evaluated=bool(node.evaluated), valid=bool(node.valid),
            fail_class=node.fail_class, error=node.error,
            delta_vs_baseline=float(node.score - self.baseline_score),
            delta_vs_parent=float(delta_parent),
            n_valid=node.n_valid, n_total=node.n_total,
            revealed_at_round=self.decision_rounds,
        )
        self._observed[node.id] = obs
        self.revealed_order.append(node.id)
        self._depth[node.branch] = max(self._depth.get(node.branch, -1), node.attempt)
        self.probes_spent += 1
        return obs


def _parse_cell(cell: str) -> Tuple[int, int]:
    try:
        b, a = cell.split("#")
        return int(b[1:]), int(a[1:])
    except Exception as exc:  # noqa: BLE001
        raise IllegalBatch(f"malformed cell id {cell!r}") from exc


# --------------------------------------------------------------------------- #
# replay episode driver                                                       #
# --------------------------------------------------------------------------- #
def drives_stepwise(policy: Any) -> bool:
    """True when the policy implements ``select_batch`` itself."""
    from .policy_api import LLMDesignedMethod  # local import: avoid cycle
    for klass in type(policy).__mro__:
        if klass is LLMDesignedMethod:
            return False
        if "select_batch" in klass.__dict__:
            return True
    return True


def drive_question(policy: Any, question: Any, round_limit: int = 64,
                   budget: Optional[int] = None,
                   collect_timeline: bool = True) -> SimResult:
    """Drive any question-like environment (live or replay) with a policy.

    Supports both policy styles: stepwise (``select_batch``) and paper-style
    (``solve``), so LLM-designed policies run unchanged in either mode.
    """
    res = SimResult()
    if not hasattr(policy, "select_batch") and hasattr(policy, "solve"):
        inner = policy.solve(question, budget)
        if isinstance(inner, SimResult):
            res = inner
        if not res.curve:
            _record_curve(res, question)
        return finalize_result(question, res)
    if not drives_stepwise(policy) and hasattr(policy, "solve"):
        inner = policy.solve(question, budget)
        if isinstance(inner, SimResult):
            res = inner
        if not res.curve:
            _record_curve(res, question)
        return finalize_result(question, res)
    try:
        if hasattr(policy, "reset"):
            policy.reset()
        guard = 0
        while question.decision_rounds < round_limit:
            guard += 1
            if guard > round_limit + 8:
                res.terminated_by = "loop-guard"
                break
            if _budget_done(question, budget):
                res.terminated_by = "budget"
                break
            if question.remaining_actions() == 0:
                res.terminated_by = "exhausted"
                break
            if hasattr(policy, "update_closed"):
                policy.update_closed(set(), question.observed(), question)
            batch = list(policy.select_batch(question) or [])
            if not batch:
                res.terminated_by = "empty-batch"
                break
            revealed = question.probe_batch(
                batch, on_reveal=lambda _o: _record_curve(res, question))
            if collect_timeline:
                res.timeline.append({
                    "round": question.decision_rounds,
                    "batch": batch,
                    "revealed": [{"cell": o.cell, "score": round(o.score, 6),
                                  "evaluated": o.evaluated,
                                  "fail_class": o.fail_class} for o in revealed],
                    "best_so_far": (None if question.best_so_far == -math.inf
                                    else round(question.best_so_far, 6)),
                    "probes": question.probes_spent,
                })
        else:
            res.terminated_by = "round-limit"
        if res.terminated_by == "unknown":
            res.terminated_by = "round-limit"
    except IllegalBatch as exc:
        res.error = f"ILLEGAL_BATCH: {exc}"
        res.terminated_by = "illegal-batch"
    except Exception as exc:  # noqa: BLE001
        res.error = f"POLICY_ERROR: {type(exc).__name__}: {exc}"
        res.terminated_by = "policy-error"
    return finalize_result(question, res)


def run_replay_episode(policy: Any, trace: DiscoveryTree,
                       max_parallelism: int = 4,
                       round_limit: Optional[int] = None,
                       budget: Optional[int] = None,
                       collect_timeline: bool = True) -> SimResult:
    """Execute one replay episode.  Deterministic for a deterministic policy."""
    K2 = int(round_limit if round_limit is not None
             else getattr(policy, "replay_round_limit", 64))
    question = ReplayQuestion(trace, max_parallelism=max_parallelism,
                              round_limit=K2)
    return drive_question(policy, question, round_limit=K2, budget=budget,
                          collect_timeline=collect_timeline)


# --------------------------------------------------------------------------- #
# objectives                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class ObjectiveConfig:
    """Replay objective Eq. (1) plus the sweep-based pareto ranking."""
    beta1: float = 0.01        # execution-cost penalty per revealed node
    beta2: float = 0.05        # parallelism bonus
    lam: float = 1.0           # pareto.reward = auc - lam * parallel_penalty


def replay_score(res: SimResult, cfg: ObjectiveConfig = ObjectiveConfig()) -> float:
    """Eq. (1): max revealed score - beta1*N + beta2*N/max(1,k)."""
    if res.best_score == -math.inf:
        return -math.inf
    n = float(res.probes)
    k = max(1, res.decision_rounds)
    return res.best_score - cfg.beta1 * n + cfg.beta2 * (n / k)


def parallel_penalty(res: SimResult) -> float:
    if res.probes <= 0:
        return 1.0
    return res.effective_sequential_rounds / res.probes


def pareto_auc(res: SimResult, trace: DiscoveryTree,
               baseline: Optional[float] = None) -> float:
    """Area under the attainment-vs-budget curve, normalized to [0, 1].

    x-axis is ABSOLUTE work: cumulative probes / probes the world contains, so two
    policies are compared on the same axis (spending fewer probes is not rewarded
    by itself - spending slots on dead branches is, because it delays the rise).
    The curve is flat-extended to x = 1: whatever attainment a policy reached
    stands as attainable at any larger budget, which is the Pareto reading of
    "reaching high per-trace attainment".

    y-axis: (best-so-far - baseline) / (trace ceiling - baseline), clamped to
    [0, 1].  A policy that wastes early slots on unproductive branches rises late
    and scores lower; one that frees slots for the productive branch rises early
    and scores higher.
    """
    if not res.curve:
        return 0.0
    base = float(trace.baseline_score if baseline is None else baseline)
    ceiling = max(trace.max_score(), base + 1e-9)
    span = max(ceiling - base, 1e-9)
    total = max(1, trace.size())
    pts = res.curve
    xs = [0.0]
    ys = [0.0]
    for probes, best, _rounds in pts:
        xs.append(min(1.0, probes / total))
        y = 0.0 if best == -math.inf else (best - base) / span
        ys.append(max(0.0, min(1.0, y)))
    if xs[-1] < 1.0:                      # flat extension to the full budget axis
        xs.append(1.0)
        ys.append(ys[-1])
    area = 0.0
    for i in range(1, len(xs)):
        area += (xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1]) / 2.0
    return area


def first_reach_probe(res: SimResult, trace: DiscoveryTree) -> Optional[int]:
    """Probe index at which the world's ceiling was first reached (diagnostic)."""
    if not res.curve:
        return None
    ceiling = trace.max_score()
    for probes, best, _k in res.curve:
        if best != -math.inf and best >= ceiling - 1e-12:
            return probes
    return None


def pareto_reward(auc: float, penalty: float, cfg: ObjectiveConfig = ObjectiveConfig()
                  ) -> float:
    return auc - cfg.lam * penalty
