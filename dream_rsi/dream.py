"""Dreaming-based policy improvement (paper Sec. 3 "Offline evaluation").

History is frozen; M candidate policy versions are constructed and each is
evaluated on *every* historical tree by replay, sweeping the single ``beta`` knob.
The version with the best sweep-based score is deployed online next cycle, which
guarantees the selected policy is no worse than the current one on the fixed
history.  Feedback from replay is handed to the policy-development agent between
revisions - never inside a policy's own decision loop.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .agents import Backend, extract_files
from .grid import GridPlanningContext
from .policy_api import LLMDesignedMethod
from .prompts import improvement_prompt
from .simulator import (ObjectiveConfig, SimResult, pareto_auc, pareto_reward,
                        parallel_penalty, run_replay_episode)
from .traces import RunLayout, append_jsonl, write_json
from .tree import DiscoveryTree

DEFAULT_BETAS = [0.2, 0.5, 0.8]


# --------------------------------------------------------------------------- #
# policy loading                                                              #
# --------------------------------------------------------------------------- #
def load_policy_module(path: str | Path):
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(f"dreamrsi_policy_{abs(hash(str(path)))}",
                                                  path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load policy from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def find_policy_class(module: Any) -> type:
    name = getattr(module, "NAME", None)
    if isinstance(name, str) and hasattr(module, name):
        cls = getattr(module, name)
        if isinstance(cls, type):
            return cls
    for attr in vars(module).values():
        if (isinstance(attr, type) and issubclass(attr, LLMDesignedMethod)
                and attr is not LLMDesignedMethod):
            return attr
    raise ImportError("no LLMDesignedMethod subclass found in policy module")


def make_policy(path: str | Path, beta: float, workers: int) -> Any:
    module = load_policy_module(path)
    cls = find_policy_class(module)
    return cls({"beta": float(beta), "max_parallelism": int(workers)})


# --------------------------------------------------------------------------- #
# evaluation                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class EpisodeEval:
    trace: str
    beta: float
    auc: float
    penalty: float
    reward: float
    probes: int
    decision_rounds: int
    best: Optional[float]
    terminated_by: str
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace": self.trace, "beta": self.beta,
            "auc": round(self.auc, 6), "parallel_penalty": round(self.penalty, 6),
            "reward": round(self.reward, 6), "probes": self.probes,
            "decision_rounds": self.decision_rounds, "best": self.best,
            "terminated_by": self.terminated_by, "error": self.error,
        }


@dataclass
class VersionReport:
    name: str
    path: str
    ok: bool = True
    error: Optional[str] = None
    betas: List[float] = field(default_factory=list)
    episodes: List[EpisodeEval] = field(default_factory=list)
    per_beta: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    best_beta: Optional[float] = None
    reward: float = float("-inf")
    auc: float = 0.0
    penalty: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.name, "path": self.path, "ok": self.ok,
            "error": self.error, "best_beta": self.best_beta,
            "reward": None if self.reward == float("-inf") else round(self.reward, 6),
            "auc": round(self.auc, 6), "parallel_penalty": round(self.penalty, 6),
            "per_beta": self.per_beta,
            "episodes": [e.to_dict() for e in self.episodes],
        }


def evaluate_version(policy_path: str | Path, traces: List[DiscoveryTree],
                     betas: List[float], workers: int = 4,
                     cfg: ObjectiveConfig = ObjectiveConfig(),
                     round_limit: int = 64,
                     collector: Optional[list] = None,
                     name: Optional[str] = None) -> VersionReport:
    path = Path(policy_path)
    report = VersionReport(name=name or path.stem, path=str(path), betas=list(betas))
    for beta in betas:
        try:
            policy = make_policy(path, beta, workers)
        except Exception as exc:  # noqa: BLE001
            report.ok = False
            report.error = f"LOAD: {type(exc).__name__}: {exc}"
            return report
        rewards = []
        for trace in traces:
            res = run_replay_episode(policy, trace, max_parallelism=workers,
                                     round_limit=round_limit)
            auc = pareto_auc(res, trace)
            pen = parallel_penalty(res)
            ep = EpisodeEval(
                trace=trace.task or f"trace{len(report.episodes)}", beta=beta,
                auc=auc, penalty=pen, reward=pareto_reward(auc, pen, cfg),
                probes=res.probes, decision_rounds=res.decision_rounds,
                best=(None if res.best_score == float("-inf")
                      else round(res.best_score, 6)),
                terminated_by=res.terminated_by, error=res.error)
            report.episodes.append(ep)
            rewards.append(ep.reward)
            if ep.error:
                report.ok = False
                report.error = ep.error
            if collector is not None:
                collector.append({
                    "version": report.name, "beta": beta,
                    "trace": ep.trace, "summary": res.to_dict(),
                    "timeline": res.timeline,
                })
        mean_reward = sum(rewards) / len(rewards) if rewards else float("-inf")
        mean_auc = sum(e.auc for e in report.episodes[-len(traces):]) / max(1, len(traces))
        mean_pen = sum(e.penalty for e in report.episodes[-len(traces):]) / max(1, len(traces))
        report.per_beta[f"{beta:.3f}"] = {
            "reward": round(mean_reward, 6), "auc": round(mean_auc, 6),
            "parallel_penalty": round(mean_pen, 6),
            "probes": sum(e.probes for e in report.episodes[-len(traces):]),
        }
        if mean_reward > report.reward:
            report.reward = mean_reward
            report.best_beta = beta
            report.auc = mean_auc
            report.penalty = mean_pen
    return report


def pareto_frontier(reports: List[VersionReport],
                    cfg: ObjectiveConfig = ObjectiveConfig()) -> List[Dict[str, Any]]:
    """Non-dominated (auc, penalty) points across versions x betas."""
    pts: List[Dict[str, Any]] = []
    for r in reports:
        for beta, row in r.per_beta.items():
            pts.append({"version": r.name, "beta": float(beta),
                        "auc": row["auc"], "parallel_penalty": row["parallel_penalty"],
                        "reward": row["reward"]})
    front = []
    for p in pts:
        dominated = any((q["auc"] >= p["auc"] and q["parallel_penalty"] <= p["parallel_penalty"])
                        and (q["auc"] > p["auc"] or q["parallel_penalty"] < p["parallel_penalty"])
                        for q in pts)
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda p: -p["reward"])


def select_best(reports: List[VersionReport]) -> Optional[VersionReport]:
    ok = [r for r in reports if r.ok and r.reward != float("-inf")]
    if not ok:
        return None
    return max(ok, key=lambda r: r.reward)


def feedback_text(reports: List[VersionReport], current: Optional[VersionReport],
                  traces: List[DiscoveryTree]) -> str:
    """Between-round feedback for the policy-development agent."""
    lines = ["### Replay sweep results (this cycle)", ""]
    lines.append(f"frozen worlds: {len(traces)}  ("
                 + ", ".join(f"{t.task}#{i}" for i, t in enumerate(traces)) + ")")
    lines.append("")
    lines.append("| version | beta | reward | auc | parallel_penalty | probes |")
    lines.append("|---|---|---|---|---|---|")
    for r in reports:
        for beta, row in sorted(r.per_beta.items(), key=lambda kv: float(kv[0])):
            lines.append(f"| {r.name} | {beta} | {row['reward']:.4f} | "
                         f"{row['auc']:.4f} | {row['parallel_penalty']:.4f} | "
                         f"{row['probes']} |")
    lines.append("")
    lines.append("Attainment/work facts per frozen world (best = highest score that "
                 "world contains):")
    lines.append("")
    lines.append("| version | beta | trace | probes spent | available | best | "
                 "terminated_by |")
    lines.append("|---|---|---|---|---|---|---|")
    avail = {f"{t.task}#{i}": t.size() for i, t in enumerate(traces)}
    for r in reports:
        for ep in r.episodes:
            lines.append(f"| {r.name} | {ep.beta:.2f} | {ep.trace} | {ep.probes} | "
                         f"{avail.get(ep.trace, '?')} | {ep.best} | "
                         f"{ep.terminated_by} |")
    lines.append("")
    lines.append("Reading guide: `auc` is attainment vs the probes *you* spent, so "
                 "reaching a high score EARLY in your own work is what pays: widen "
                 "early, and do not spend probes after attainment stalled. "
                 "`parallel_penalty` is mean(effective_sequential_rounds / probes); "
                 "with W workers a useful full batch approaches 1/W, a serial "
                 "policy approaches 1. Both matter: reward = auc - penalty.")
    if current is not None:
        lines.append("")
        lines.append(f"Current deployed policy scores reward={current.reward:.4f} "
                     f"at beta={current.best_beta} (auc={current.auc:.4f}, "
                     f"penalty={current.penalty:.4f}). Beat it: equal reward is "
                     f"rejected, the selection keeps the incumbent.")
    weak = [r for r in reports if not r.ok]
    if weak:
        lines.append("")
        lines.append("Failed versions (fix or avoid these mistakes):")
        for r in weak:
            lines.append(f"- {r.name}: {r.error}")
    lines.append("")
    lines.append("Diagnose concrete behaviour from the replay episodes: serial "
                 "batches, premature stops, over-pruning, wasted probes. Read "
                 "`proposal_results/policy_execution_traces.jsonl` (between-round "
                 "feedback only - never inside select_batch).")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# policy development                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class DevelopmentResult:
    path: str
    ok: bool
    error: Optional[str] = None
    agent: Optional[Dict[str, Any]] = None
    tries: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "ok": self.ok, "error": self.error,
                "agent": self.agent, "tries": self.tries}


def develop_policy(agent: Backend, round_dir: Path, version_tag: str,
                   history_dir: Path, baseline_dir: Path, feedback: str,
                   traces: List[DiscoveryTree], betas: List[float], workers: int,
                   cfg: ObjectiveConfig = ObjectiveConfig(),
                   round_limit: int = 64, attempts: int = 2) -> DevelopmentResult:
    """Ask the development agent for a new policy version, then validate it."""
    version_dir = round_dir / "policy_code" / version_tag
    version_dir.mkdir(parents=True, exist_ok=True)
    (round_dir / "policy_input").mkdir(parents=True, exist_ok=True)
    method_path = version_dir / "method.py"
    if not method_path.exists():
        method_path.write_text(_template_policy(), encoding="utf-8")

    prompt = improvement_prompt(str(method_path), str(history_dir),
                                str(baseline_dir), feedback)
    (round_dir / "policy_input" / f"{version_tag}.prompt.txt").write_text(
        prompt, encoding="utf-8")

    last_error = "no attempt"
    for try_index in range(1, attempts + 1):
        result = agent.complete(prompt, cwd=str(version_dir))
        (round_dir / "policy_input" / f"{version_tag}.agent{try_index}.json").write_text(
            json.dumps(result.to_dict(), indent=1), encoding="utf-8")
        if result.text:
            (round_dir / "policy_input" / f"{version_tag}.agent{try_index}.txt").write_text(
                result.text, encoding="utf-8")
        if not result.ok:
            last_error = result.error or "agent failure"
            continue
        if (result.usage or {}).get("finish_reason") == "length":
            last_error = ("agent output was truncated by the completion budget "
                          "(finish_reason=length) - return the complete file in "
                          "fewer tokens, no restatement of the prompt")
            prompt = (prompt + "\n\n## Previous attempt was truncated\n\n"
                      + "Your reply hit the output limit mid-file. Return the "
                      + "COMPLETE `method.py` again, much shorter: no prose, no "
                      + "restating the instructions, no duplicated helpers.")
            continue
        files = extract_files(result.text, expected_name=str(method_path.name))
        for rel, body in files.items():
            target = (version_dir / rel).resolve()
            if not str(target).startswith(str(version_dir.resolve())):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
        if method_path.read_text(encoding="utf-8") == _template_policy():
            last_error = ("the delivered file is byte-identical to the template - "
                          "the policy was never edited")
            prompt = (prompt + "\n\n## Previous attempt changed nothing\n\n"
                      + f"Write your policy to the relative path "
                      + f"`{method_path.name}` (no absolute paths). The file must "
                      + "differ from the template stub you were given.")
            continue
        if (result.usage or {}).get("finish_reason") == "length":
            last_error = "truncated output (finish_reason=length)"
            prompt = (prompt + "\n\n## Previous attempt was truncated\n\n"
                      + "Shorten the file: complete `method.py`, no prose.")
            continue
        ok, err = validate_policy(method_path, traces, betas, workers, cfg,
                                  round_limit)
        if ok:
            return DevelopmentResult(str(method_path), True,
                                     agent=result.to_dict(), tries=try_index)
        last_error = err or "validation failed"
        prompt = (prompt + "\n\n## Previous attempt was rejected\n\n"
                  + f"`{method_path}` failed validation: {last_error}\n"
                  + "Fix exactly that and return the complete file again.")
    return DevelopmentResult(str(method_path), False, error=last_error,
                             tries=attempts)


def validate_policy(path: Path, traces: List[DiscoveryTree], betas: List[float],
                    workers: int, cfg: ObjectiveConfig = ObjectiveConfig(),
                    round_limit: int = 64) -> Tuple[bool, Optional[str]]:
    """Import the policy and replay it once; illegal batches = invalid policy."""
    try:
        module = load_policy_module(path)
        cls = find_policy_class(module)
    except Exception as exc:  # noqa: BLE001
        return False, f"IMPORT: {type(exc).__name__}: {exc}"
    try:
        policy = cls({"beta": float(betas[0]) if betas else 0.6,
                      "max_parallelism": int(workers)})
    except Exception as exc:  # noqa: BLE001
        return False, f"CONSTRUCT: {type(exc).__name__}: {exc}"
    if traces:
        res = run_replay_episode(policy, traces[0], max_parallelism=workers,
                                 round_limit=round_limit)
        if res.error:
            return False, res.error
    ctx = GridPlanningContext(worker_cap=workers)
    try:
        plan = policy.plan_grid(ctx)
    except NotImplementedError:
        return False, "plan_grid not implemented"
    except Exception as exc:  # noqa: BLE001
        return False, f"PLAN_GRID: {type(exc).__name__}: {exc}"
    if plan is None:
        return False, "plan_grid returned None"
    return True, None


def _template_policy() -> str:
    baseline = Path(__file__).resolve().parent / "policies" / "baseline.py"
    return baseline.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# one dreaming phase                                                          #
# --------------------------------------------------------------------------- #
def dream_phase(layout: RunLayout, round_dir: Path, traces: List[DiscoveryTree],
                current_policy: Path, agent: Backend, versions: int = 3,
                betas: Optional[List[float]] = None, workers: int = 4,
                cfg: ObjectiveConfig = ObjectiveConfig(),
                round_limit: int = 64, seed: Optional[int] = None,
                stamp: str = "") -> Dict[str, Any]:
    """Evaluate the deployed policy plus ``versions`` new ones; pick the best."""
    betas = list(betas or DEFAULT_BETAS)
    results_dir = round_dir / "proposal_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    executed: List[Dict[str, Any]] = []

    reports: List[VersionReport] = []
    current_report = evaluate_version(current_policy, traces, betas, workers, cfg,
                                      round_limit, executed, name="m000_current")
    reports.append(current_report)

    history_dir = round_dir / "history"
    if not history_dir.exists():
        history_dir = round_dir / "trace"
    feedback = feedback_text(reports, None, traces)
    dev_results: List[DevelopmentResult] = []
    for m in range(1, versions + 1):
        tag = f"m{m:03d}{('_' + stamp) if stamp else ''}"
        dev = develop_policy(agent, round_dir, tag, history_dir,
                             layout.baseline_dir, feedback, traces, betas,
                             workers, cfg, round_limit)
        dev_results.append(dev)
        if not dev.ok:
            reports.append(VersionReport(name=tag, path=dev.path, ok=False,
                                         error=dev.error))
            feedback = feedback_text(reports, current_report, traces)
            continue
        report = evaluate_version(dev.path, traces, betas, workers, cfg,
                                  round_limit, executed, name=tag)
        reports.append(report)
        feedback = feedback_text(reports, current_report, traces)

    best = select_best(reports) or current_report
    write_json(results_dir / "beta_sweep.json", {
        "betas": betas,
        "objective": {"lambda": cfg.lam, "beta1": cfg.beta1, "beta2": cfg.beta2},
        "versions": [r.to_dict() for r in reports],
        "pareto_frontier": pareto_frontier(reports, cfg),
        "selected": {"version": best.name, "beta": best.best_beta,
                     "reward": None if best.reward == float("-inf")
                     else round(best.reward, 6)},
        "created_at": time.time(),
    })
    append_jsonl(results_dir / "policy_execution_traces.jsonl", executed)
    write_json(results_dir / "replay_summary.json", {
        "worlds": len(traces),
        "versions_evaluated": [r.name for r in reports],
        "developed": [d.to_dict() for d in dev_results],
        "selected": best.name,
        "selected_beta": best.best_beta,
        "selected_reward": (None if best.reward == float("-inf")
                            else round(best.reward, 6)),
        "current_reward": (None if current_report.reward == float("-inf")
                           else round(current_report.reward, 6)),
        "improved": bool(best.name != current_report.name),
    })
    return {
        "reports": reports,
        "selected": best,
        "dev_results": dev_results,
        "betas": betas,
        "summary_path": str(results_dir / "replay_summary.json"),
        "sweep_path": str(results_dir / "beta_sweep.json"),
    }
