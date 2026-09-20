"""Online rollout: the live discovery tree (paper Sec. 3 "Online rollout").

The exploration policy chooses a batch of cells; each selected cell is one
generation-evaluation attempt: the discovery agent resumes that parent's saved
workspace and produces a new candidate, the evaluator scores it, and the child is
attached to the tree.  The policy code stays fixed throughout the rollout.
"""
from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .agents import Backend, extract_files
from .prompts import exploration_prompt
from .simulator import (CellMeta, IllegalBatch, Observation, SimResult,
                        drive_question)
from .tasks.base import EvalOutcome, Task
from .tree import DiscoveryTree, Node

# fail classes that a one-shot repair prompt can plausibly fix
REPAIRABLE_FAILURES = {"compile", "runtime", "timeout", "invalid", "agent", "env"}


class LiveSession:
    """Owns the live grid, agent, evaluator and per-round artifacts."""

    def __init__(self, round_dir: Path, task: Task, agent: Backend,
                 plan: Dict[str, int], workers: int = 4,
                 baseline_score: float = 0.0,
                 history_dir: Optional[Path] = None,
                 baseline_dir: Optional[Path] = None,
                 round_index: int = 1, seed_tree: Optional[DiscoveryTree] = None,
                 agent_timeout: int = 900, verbose: bool = False,
                 repairs: int = 1) -> None:
        self.round_dir = Path(round_dir)
        self.attempts_dir = self.round_dir / "trace" / "attempts"
        self.attempts_dir.mkdir(parents=True, exist_ok=True)
        self.task = task
        self.agent = agent
        self.branch_count = int(plan.get("branch_count", 4))
        self.refine_count = int(plan.get("refine_count", 3))
        self.workers = max(1, int(workers))
        self.history_dir = Path(history_dir) if history_dir else None
        self.baseline_dir = Path(baseline_dir) if baseline_dir else None
        self.agent_timeout = agent_timeout
        self.verbose = verbose
        self.repairs = max(0, int(repairs))

        self.tree = seed_tree or DiscoveryTree(task=task.name,
                                               baseline_score=baseline_score)
        self.tree.task = task.name
        self.tree.baseline_score = baseline_score
        self.tree.round_index = round_index
        self.tree.meta.update({"planned_grid": plan, "workers": self.workers})

    # ---------------------------------------------------------------- paths
    def attempt_dir(self, branch: int, attempt: int) -> Path:
        return self.attempts_dir / f"b{branch}" / f"a{attempt:04d}"

    # ------------------------------------------------------------ validation
    def legal_roots(self) -> List[str]:
        have = {n.branch for n in self.tree.nodes.values()}
        return [f"b{b}#a0" for b in range(self.branch_count) if b not in have]

    def legal_actions(self) -> List[str]:
        acts = list(self.legal_roots())
        for b in sorted({n.branch for n in self.tree.nodes.values()}):
            nxt = self._next_attempt(b)
            if nxt <= self.refine_count:
                acts.append(f"b{b}#a{nxt}")
        return acts

    def _next_attempt(self, branch: int) -> int:
        attempts = [n.attempt for n in self.tree.nodes.values() if n.branch == branch]
        return (max(attempts) + 1) if attempts else 0

    def _validate(self, cells: Sequence[str]) -> None:
        if len(cells) > self.workers:
            raise IllegalBatch(
                f"batch size {len(cells)} exceeds W={self.workers}")
        if len(set(cells)) != len(cells):
            raise IllegalBatch("duplicate cell ids in batch")
        legal = set(self.legal_actions())
        for c in cells:
            if c not in legal:
                raise IllegalBatch(f"illegal cell {c}")
        for c in cells:
            b, a = _parse(c)
            for other in cells:
                if other == c:
                    continue
                ob, oa = _parse(other)
                if ob == b and abs(oa - a) == 1:
                    raise IllegalBatch(f"parent and child in same batch: {c}, {other}")

    # --------------------------------------------------------------- attempt
    def probe_batch(self, cells: Sequence[str],
                    on_reveal: Optional[Callable[[Observation], None]] = None
                    ) -> List[Observation]:
        cells = list(cells)
        self._validate(cells)
        with ThreadPoolExecutor(max_workers=min(len(cells) or 1, self.workers)) as pool:
            futures = [pool.submit(self._run_attempt, *_parse(c)) for c in cells]
            nodes = [f.result() for f in futures]
        obs: List[Observation] = []
        for node in nodes:
            o = self._observation(node)
            obs.append(o)
            if on_reveal is not None:
                on_reveal(o)
        self._save_tree()
        return obs

    def _run_attempt(self, branch: int, attempt: int) -> Node:
        node_dir = self.attempt_dir(branch, attempt)
        node_dir.mkdir(parents=True, exist_ok=True)
        (node_dir / "eval").mkdir(exist_ok=True)
        parent_id = f"b{branch}#a{attempt - 1}" if attempt > 0 else None
        parent = self.tree.get(parent_id) if parent_id else None

        parent_summary = self._parent_summary(parent)
        prompt = exploration_prompt(
            task_desc=self.task.description,
            scoring_desc=self.task.scoring,
            eval_program=self.task.eval_program,
            node_dir=str(node_dir),
            history_dir=str(self.history_dir or ""),
            baseline_dir=str(self.baseline_dir or ""),
            problem_file=self.task.problem_file,
            direction_guidance=(self.task.direction_hint(branch)
                                if attempt == 0 else ""),
            branch_context=self.task.branch_context(branch, attempt, parent_summary),
            history_snapshot=self._history_snapshot())
        (node_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        t0 = time.time()
        result = self.agent.complete(prompt, cwd=str(node_dir))
        results = [result]
        outcome = self._grade(node_dir, result)
        repairs = 0
        while (repairs < self.repairs
               and outcome.fail_class in REPAIRABLE_FAILURES):
            repairs += 1
            repair_prompt = self._repair_prompt(prompt, outcome)
            (node_dir / f"prompt.repair{repairs}.txt").write_text(
                repair_prompt, encoding="utf-8")
            result = self.agent.complete(repair_prompt, cwd=str(node_dir))
            results.append(result)
            outcome = self._grade(node_dir, result)
        (node_dir / "agent.json").write_text(
            json.dumps({"results": [r.to_dict() for r in results],
                        "repairs": repairs}, indent=1), encoding="utf-8")

        node = self.tree.add_node(
            branch=branch, attempt=attempt, parent_id=parent_id,
            score=float(outcome.score), evaluated=True, valid=bool(outcome.valid),
            fail_class=outcome.fail_class, error=outcome.error,
            n_valid=outcome.n_valid, n_total=outcome.n_total,
            artifact=self.task.eval_program,
            workspace=str(node_dir), proposal="proposal.md",
            tags=[self.task.direction_hint(branch)] if attempt == 0 else [])
        if self.verbose:
            print(f"[online] b{branch}#a{attempt} score={outcome.score:.6g} "
                  f"valid={outcome.valid} {outcome.fail_class} "
                  f"({time.time() - t0:.1f}s)", flush=True)
        return node

    def _grade(self, node_dir: Path, result: Any) -> EvalOutcome:
        """Write the agent's files, then evaluate the attempt directory."""
        if result.text:
            (node_dir / "agent_output.txt").write_text(result.text, encoding="utf-8")
        wrote = False
        if getattr(result, "ok", False):
            files = extract_files(result.text,
                                  expected_name=self.task.eval_program)
            for rel, body in files.items():
                try:
                    target = node_dir / rel
                    if not str(target.resolve()).startswith(str(node_dir.resolve())):
                        continue  # ignore traversal attempts
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(body, encoding="utf-8")
                    wrote = True
                except OSError as exc:
                    # a bad path from the model must not kill the rollout
                    (node_dir / "eval" / "write_errors.txt").open("a").write(
                        f"{rel[:80]}: {exc}\n")
                    continue
            if getattr(result, "wrote_files", False):
                wrote = wrote or self.task.program_path(node_dir).exists()
        if not wrote:
            outcome = EvalOutcome(
                self.task.failure_score, valid=False,
                fail_class="env" if getattr(result, "ok", False) else "agent",
                error=getattr(result, "error", None)
                or "discovery agent produced no program file")
        else:
            outcome = self.task.evaluate(node_dir)
        (node_dir / "eval" / "score.json").write_text(
            json.dumps(outcome.to_dict(), indent=1), encoding="utf-8")
        if outcome.error:
            (node_dir / "eval" / "error.txt").write_text(outcome.error,
                                                         encoding="utf-8")
        return outcome

    def _repair_prompt(self, prompt: str, outcome: EvalOutcome) -> str:
        detail = (outcome.error or outcome.fail_class)[:1200]
        return (prompt
                + "\n\n## Your previous submission was rejected by the evaluator\n\n"
                + f"fail_class: {outcome.fail_class}\n\n```\n{detail}\n```\n\n"
                + f"Rewrite `{self.task.eval_program}` as a COMPLETE, runnable file "
                + "that fixes exactly this. Remove every placeholder, ellipsis, "
                + "question mark and pseudo-code comment. The file is executed "
                + "as-is, so it must be valid Python from the first to the last "
                + "line. Return the whole file again with the FILE protocol.")

    def _history_snapshot(self) -> str:
        """Measured history the discovery agent may read (the agent, not the policy)."""
        rows = self.tree.compact_rows()
        if not rows:
            return ""
        lines = ["| cell | branch | attempt | score | valid | fail_class |",
                 "|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['cell']} | {r['b']} | {r['a']} | {r['s']} | "
                         f"{r['valid']} | {r['fc']} |")
        best = self.tree.best()
        if best is not None and best.workspace:
            prog = self.task.program_path(Path(best.workspace))
            if prog.exists():
                code = prog.read_text(encoding="utf-8", errors="ignore")
                lines.append("")
                lines.append(f"### Best recorded program so far ({best.id}, "
                             f"score={best.score:.6g})")
                lines.append("```python")
                lines.append(code[:4000])
                lines.append("```")
        lines.append("")
        lines.append(f"Baseline floor to beat: {self.tree.baseline_score:.6g}")
        return "\n".join(lines)

    def _parent_summary(self, parent: Optional[Node]) -> str:
        if parent is None:
            return "This attempt opens a new branch at the root workspace."
        out = [f"Parent {parent.id}: score={parent.score:.6g} "
               f"valid={parent.valid} fail_class={parent.fail_class}"]
        if parent.proposal:
            p = Path(parent.workspace or "") / parent.proposal
            if p.exists():
                out.append("Parent proposal:\n" + p.read_text(encoding="utf-8")[:1500])
        ws = Path(parent.workspace or "")
        for f in sorted(ws.glob("*.*"))[:4]:
            if f.name == "prompt.txt" or f.suffix in (".json", ".txt"):
                continue
            out.append(f"Parent artifact file available to copy: {f.name}")
        return "\n".join(out)

    def _observation(self, node: Node) -> Observation:
        parent_score = math.inf
        if node.parent_id:
            p = self.tree.get(node.parent_id)
            if p is not None:
                parent_score = p.score
        return Observation(
            cell=node.id, branch=node.branch, attempt=node.attempt,
            parent_id=node.parent_id, score=node.score, evaluated=node.evaluated,
            valid=node.valid, fail_class=node.fail_class, error=node.error,
            delta_vs_baseline=node.score - self.tree.baseline_score,
            delta_vs_parent=(0.0 if parent_score == math.inf
                             else node.score - parent_score),
            n_valid=node.n_valid, n_total=node.n_total)

    def _save_tree(self) -> None:
        self.tree.save(self.round_dir / "trace" / "tree.json")


class LiveQuestion:
    """Policy-facing view of the live grid (same interface as ReplayQuestion)."""

    def __init__(self, session: LiveSession) -> None:
        self._session = session
        self.max_parallelism = session.workers
        self.baseline_score = session.tree.baseline_score
        self.probes_spent = 0
        self.decision_rounds = 0
        self.effective_sequential_rounds = 0.0
        self.revealed_order: List[str] = []

    def reset(self) -> None:
        self.probes_spent = 0
        self.decision_rounds = 0
        self.effective_sequential_rounds = 0.0
        self.revealed_order = []

    def observed(self) -> Dict[str, Observation]:
        return {n.id: self._session._observation(n)  # noqa: SLF001
                for n in self._session.tree.nodes.values()}

    def legal_actions(self) -> List[str]:
        return self._session.legal_actions()

    def legal_roots(self) -> List[str]:
        return self._session.legal_roots()

    def opened_branches(self) -> List[int]:
        return sorted({n.branch for n in self._session.tree.nodes.values()})

    def meta(self, cell: str) -> CellMeta:
        b, a = _parse(cell)
        node = self._session.tree.get(cell)
        return CellMeta(cell=cell, branch=b, attempt=a,
                        parent_id=(node.parent_id if node else None),
                        seq=(node.seq if node else 0),
                        tags=tuple(node.tags) if node else ())

    def close_branch(self, branch: int) -> None:
        """Live grids are bounded by the plan; closure is policy-local."""

    def remaining_actions(self) -> int:
        return len(self.legal_actions())

    @property
    def best_so_far(self) -> float:
        obs = self.observed()
        if not obs:
            return -math.inf
        return max((o.score for o in obs.values() if o.evaluated), default=-math.inf)

    def probe_batch(self, cells: Sequence[str],
                    on_reveal: Optional[Callable[[Observation], None]] = None
                    ) -> List[Observation]:
        cells = list(cells)
        obs = self._session.probe_batch(cells, on_reveal)
        self.probes_spent += len(cells)
        self.decision_rounds += 1
        self.effective_sequential_rounds += math.ceil(
            max(1, len(cells)) / self.max_parallelism)
        self.revealed_order.extend(o.cell for o in obs)
        return obs


def run_online_round(session: LiveSession, policy: Any,
                     round_limit: int = 32) -> Tuple[DiscoveryTree, SimResult]:
    """One online rollout: policy guides the session's discovery tree."""
    question = LiveQuestion(session)
    res = drive_question(policy, question, round_limit=round_limit,
                         collect_timeline=True)
    session._save_tree()  # noqa: SLF001
    return session.tree, res


def _parse(cell: str) -> Tuple[int, int]:
    b, a = cell.split("#")
    return int(b[1:]), int(a[1:])
