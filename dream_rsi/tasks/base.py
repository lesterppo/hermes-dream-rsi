"""Task interface: what a discovery cycle is discovering, and how it is scored.

A task owns:
  * the problem statement handed to the discovery agent
  * the evaluated program name (the artifact the agent writes)
  * an evaluator that turns an attempt directory into a score + diagnostics
  * a baseline program that establishes the floor score
  * a direction provider: the semantic hint attached to each new root
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PKG_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class EvalOutcome:
    score: float
    valid: bool = True
    fail_class: str = "ok"
    error: Optional[str] = None
    n_valid: Optional[int] = None
    n_total: Optional[int] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    stdout: str = ""
    runtime_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score, "valid": self.valid,
            "fail_class": self.fail_class, "error": self.error,
            "n_valid": self.n_valid, "n_total": self.n_total,
            "diagnostics": self.diagnostics,
            "runtime_s": round(self.runtime_s, 4),
        }


class Task:
    name = "task"
    eval_program = "solution.py"
    problem_file = "problem.md"
    description = ""
    scoring = ""
    directions: List[str] = []
    # grid defaults (paper: 10 workspaces x 11 refinements for Pro, 32 x 20 Flash)
    default_branches = 4
    default_refines = 3
    default_workers = 4
    failure_score = -1e9
    call_timeout = 900

    # ------------------------------------------------------------------ setup
    def prepare(self, run_dir: Path, baseline_dir: Path) -> EvalOutcome:
        base = baseline_dir
        base.mkdir(parents=True, exist_ok=True)
        (base / "eval").mkdir(exist_ok=True)
        (base / self.problem_file).write_text(self.description, encoding="utf-8")
        (base / "proposal.md").write_text(
            "Baseline reference implementation (paper: parallel-refine floor).\n",
            encoding="utf-8")
        src = self.baseline_source()
        if src is not None:
            (base / self.eval_program).write_text(src, encoding="utf-8")
        outcome = self.evaluate(base)
        (base / "eval" / "score.json").write_text(
            json.dumps(outcome.to_dict(), indent=1), encoding="utf-8")
        return outcome

    def baseline_source(self) -> Optional[str]:
        return None

    # -------------------------------------------------------------- evaluation
    def evaluate(self, attempt_dir: Path) -> EvalOutcome:
        raise NotImplementedError

    def direction_hint(self, index: int) -> str:
        if self.directions:
            return self.directions[index % len(self.directions)]
        return ""

    def program_path(self, attempt_dir: Path) -> Path:
        return Path(attempt_dir) / self.eval_program

    # ------------------------------------------------------- subprocess helper
    def run_harness(self, attempt_dir: Path, harness: str,
                    timeout: Optional[int] = None) -> EvalOutcome:
        """Run an evaluator script against the attempt's program in isolation."""
        attempt_dir = Path(attempt_dir)
        (attempt_dir / "eval").mkdir(parents=True, exist_ok=True)
        script = attempt_dir / "eval" / "_harness.py"
        script.write_text(harness, encoding="utf-8")
        params = self.harness_params(attempt_dir)
        env = dict(os.environ)
        env["DREAMRSI_ATTEMPT"] = str(attempt_dir)
        env["DREAMRSI_PARAMS"] = json.dumps(params)
        env["PYTHONPATH"] = f"{attempt_dir}{os.pathsep}{env.get('PYTHONPATH', '')}"
        t0 = time.time()
        try:
            proc = subprocess.run([sys.executable, str(script)], cwd=str(attempt_dir),
                                  capture_output=True, text=True,
                                  timeout=timeout or self.call_timeout, env=env)
        except subprocess.TimeoutExpired:
            return EvalOutcome(self.failure_score, valid=False, fail_class="timeout",
                               error=f"evaluator timeout after "
                                     f"{timeout or self.call_timeout}s",
                               runtime_s=time.time() - t0)
        out = proc.stdout or ""
        (attempt_dir / "eval" / "stdout.txt").write_text(
            out + ("\n[stderr]\n" + proc.stderr if proc.stderr else ""),
            encoding="utf-8")
        payload: Dict[str, Any] = {}
        for line in reversed(out.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    payload = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        if not payload:
            fail = "compile" if "Traceback" in (proc.stderr or "") and \
                "SyntaxError" in (proc.stderr or "") else "runtime"
            return EvalOutcome(self.failure_score, valid=False, fail_class=fail,
                               error=(proc.stderr or "evaluator produced no JSON")[-400:],
                               stdout=out, runtime_s=time.time() - t0)
        return EvalOutcome(
            score=float(payload.get("score", self.failure_score)),
            valid=bool(payload.get("valid", False)),
            fail_class=str(payload.get("fail_class", "ok" if payload.get("valid") else "invalid")),
            error=payload.get("error"),
            n_valid=payload.get("n_valid"), n_total=payload.get("n_total"),
            diagnostics=payload.get("diagnostics") or {},
            stdout=out, runtime_s=time.time() - t0,
        )

    def harness_params(self, attempt_dir: Path) -> Dict[str, Any]:
        return {}

    # ------------------------------------------------------------------ prompt
    def branch_context(self, branch: int, attempt: int, parent_summary: str = "") -> str:
        return (f"branch {branch} / attempt {attempt}\n"
                f"{parent_summary}".strip())
