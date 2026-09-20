"""Recursive self-improvement loop (paper Fig. 1).

per round t:
  1. plan_grid   -> how wide/deep the next live grid is (prefix-safe context)
  2. online explore with the current policy  -> discovery tree T_t
  3. build/refresh the replay simulator pool from history H_t
  4. dream: evaluate M candidate policies on every world, sweep beta, select
  5. redeploy the selected policy for round t+1 and expand the pool
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agents import Backend, make_agent
from .dream import (DEFAULT_BETAS, VersionReport, dream_phase, evaluate_version,
                    make_policy, select_best)
from .grid import GridPlan, GridPlanningContext
from .online import LiveSession, run_online_round
from .simulator import ObjectiveConfig
from .tasks import build_task
from .traces import RunLayout, live_manifest, write_json
from .tree import DiscoveryTree


def ensure_baseline(layout: RunLayout, task, state: Dict[str, Any]) -> Dict[str, Any]:
    if "baseline_score" in state and layout.baseline_dir.exists():
        return state
    outcome = task.prepare(layout.root, layout.baseline_dir)
    state["baseline_score"] = outcome.score
    state["baseline"] = outcome.to_dict()
    layout.save_state(state)
    return state


def ensure_policy(layout: RunLayout, state: Dict[str, Any]) -> Path:
    if layout.current_policy.exists():
        return layout.current_policy
    from .dream import _template_policy
    layout.current_policy.write_text(_template_policy(), encoding="utf-8")
    state["current_policy"] = str(layout.current_policy)
    state["current_beta"] = state.get("current_beta", 0.6)
    layout.save_state(state)
    return layout.current_policy


def planning_context(layout: RunLayout, task, traces: List[DiscoveryTree],
                     workers: int, hard_branch: int, hard_refine: int
                     ) -> GridPlanningContext:
    return GridPlanningContext(
        history=layout.history_manifests(),
        worker_cap=workers,
        fallback_branch_count=min(task.default_branches, hard_branch),
        fallback_refine_count=min(task.default_refines, hard_refine),
        hard_max_branch_count=hard_branch,
        hard_max_refine_count=hard_refine,
        trace_branch_count=(max((t.branch_count for t in traces), default=None)
                            if traces else None),
        trace_refine_count=(max((t.refine_count for t in traces), default=None)
                            if traces else None),
    )


def run_explore(layout: RunLayout, state: Dict[str, Any], task, agent: Backend,
                workers: int, k1: int, plan: Optional[Dict[str, int]] = None,
                hard_branch: int = 8, hard_refine: int = 6,
                agent_timeout: int = 900, verbose: bool = False,
                round_index: Optional[int] = None, repairs: int = 1
                ) -> Dict[str, Any]:
    """Stage 1: one online discovery rollout with the current policy."""
    traces = [DiscoveryTree.load(p) for p in layout.trace_paths()]
    policy_path = ensure_policy(layout, state)
    beta = float(state.get("current_beta", 0.6))
    policy = make_policy(policy_path, beta, workers)

    ctx = planning_context(layout, task, traces, workers, hard_branch, hard_refine)
    plan_obj: GridPlan = policy.plan_grid(ctx)
    if plan_obj is None:
        plan_obj = GridPlan(ctx.fallback_branch_count, ctx.fallback_refine_count,
                            "policy returned no plan; conservative bootstrap")
    plan_obj = plan_obj.clamp(ctx)
    if plan:
        plan_obj = GridPlan(int(plan.get("branch_count", plan_obj.branch_count)),
                            int(plan.get("refine_count", plan_obj.refine_count)),
                            plan_obj.reason + " (cli override)").clamp(ctx)

    index = int(round_index or (state.get("round", 0) + 1))
    round_dir = layout.new_round(index)
    prev = layout.round_dirs()[-2] if len(layout.round_dirs()) > 1 else None
    layout.link_history(round_dir, prev)
    shutil.copy2(policy_path, round_dir / "method.py")

    session = LiveSession(
        round_dir=round_dir, task=task, agent=agent,
        plan={"branch_count": plan_obj.branch_count,
              "refine_count": plan_obj.refine_count,
              "reason": plan_obj.reason},
        workers=workers, baseline_score=float(state.get("baseline_score", 0.0)),
        history_dir=round_dir / "history" if (round_dir / "history").exists() else None,
        baseline_dir=layout.baseline_dir, round_index=index,
        agent_timeout=agent_timeout, verbose=verbose, repairs=repairs)
    tree, res = run_online_round(session, policy, round_limit=k1)

    summary = tree.summary()
    manifest = live_manifest(
        task=task.name, round_index=index,
        plan={"branch_count": plan_obj.branch_count,
              "refine_count": plan_obj.refine_count, "reason": plan_obj.reason},
        tree_summary=summary, beta=beta, probes=res.probes,
        decision_rounds=res.decision_rounds, policy=policy_path.name,
        extra={"terminated_by": res.terminated_by, "policy_error": res.error,
               "plan_reason": plan_obj.reason})
    write_json(round_dir / "live_cycle_manifest.json", manifest)
    pool_dir = layout.trace_pool / f"iter{index:04d}"
    pool_dir.mkdir(parents=True, exist_ok=True)
    write_json(pool_dir / "live_cycle_manifest.json", manifest)

    state["round"] = index
    state["last_round_dir"] = str(round_dir)
    state["last_plan"] = {"branch_count": plan_obj.branch_count,
                          "refine_count": plan_obj.refine_count}
    if tree.nodes:
        best = tree.best()
        if best and (best.score > float(state.get("best_score", float("-inf")))):
            state["best_score"] = best.score
            state["best_cell"] = best.id
    layout.save_state(state)

    return {"round_dir": round_dir, "tree": tree, "result": res,
            "manifest": manifest, "plan": plan_obj, "policy": policy_path}


def run_dream(layout: RunLayout, state: Dict[str, Any], agent: Backend,
              versions: int = 3, betas: Optional[List[float]] = None,
              workers: int = 4, k2: int = 64,
              cfg: ObjectiveConfig = ObjectiveConfig(),
              round_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Stage 3: dreaming-based policy improvement over the frozen worlds."""
    traces = [DiscoveryTree.load(p) for p in layout.trace_paths()]
    if not traces:
        raise RuntimeError("no discovery history yet - run `explore` first")
    policy_path = ensure_policy(layout, state)
    round_dir = Path(round_dir or state.get("last_round_dir") or layout.round_dirs()[-1])
    out = dream_phase(layout, round_dir, traces, policy_path, agent,
                      versions=versions, betas=betas, workers=workers, cfg=cfg,
                      round_limit=k2, stamp=time.strftime("%Y%m%d-%H%M%S"))
    selected: VersionReport = out["selected"]
    if selected.ok and selected.path and Path(selected.path).exists():
        src = Path(selected.path).resolve()
        if src != layout.current_policy.resolve():
            shutil.copy2(src, layout.current_policy)
        if src != layout.current_policy.resolve():
            # keep an immutable copy of every deployed version
            dest = layout.versions_dir / f"{selected.name}.py"
            if not dest.exists():
                shutil.copy2(src, dest)
        state["current_policy"] = str(layout.current_policy)
        state["current_beta"] = selected.best_beta
        state["selected_version"] = selected.name
        state["selected_reward"] = (None if selected.reward == float("-inf")
                                    else selected.reward)
        state["worlds"] = len(traces)
    layout.save_state(state)
    return out


def run_loop(run_dir: str | Path, task_spec: str, agent_spec: str,
             rounds: int = 3, workers: int = 4, k1: int = 32, versions: int = 3,
             betas: Optional[List[float]] = None, k2: int = 64,
             hard_branch: int = 8, hard_refine: int = 6,
             agent_timeout: int = 900, verbose: bool = True,
             cfg: Optional[ObjectiveConfig] = None,
             task_kwargs: Optional[Dict[str, Any]] = None,
             repairs: int = 1, max_tokens: int = 32768,
             branches: Optional[int] = None,
             refines: Optional[int] = None) -> Dict[str, Any]:
    """Full RSI loop: explore -> simulate -> dream -> redeploy, T times."""
    cfg = cfg or ObjectiveConfig()
    betas = list(betas or DEFAULT_BETAS)
    layout = RunLayout(Path(run_dir)).ensure()
    state = layout.load_state()
    if state.get("task") != task_spec:
        state["task"] = task_spec
        state["round"] = 0
        state["current_beta"] = state.get("current_beta", 0.6)
    task = build_task(task_spec, **(task_kwargs or {}))
    # CLI grid overrides become the planning-context fallbacks (hard caps still
    # bound what any policy may create) and the round-1 grid.
    if branches:
        task.default_branches = int(branches)
    if refines is not None:
        task.default_refines = int(refines)
    state = ensure_baseline(layout, task, state)
    ensure_policy(layout, state)

    history: List[Dict[str, Any]] = []
    for _ in range(int(rounds)):
        agent = make_agent(agent_spec, timeout=agent_timeout,
                           program_name=task.eval_program, task=task.name,
                           max_tokens=max_tokens)
        exp = run_explore(layout, state, task, agent, workers=workers, k1=k1,
                          hard_branch=hard_branch, hard_refine=hard_refine,
                          agent_timeout=agent_timeout, verbose=verbose,
                          repairs=repairs)
        tree: DiscoveryTree = exp["tree"]
        dev_agent = make_agent(agent_spec, timeout=agent_timeout,
                               program_name=task.eval_program, task=task.name,
                               max_tokens=max_tokens)
        dream = run_dream(layout, state, dev_agent, versions=versions,
                          betas=betas, workers=workers, k2=k2, cfg=cfg,
                          round_dir=exp["round_dir"])
        history.append({
            "round": state.get("round"),
            "round_dir": str(exp["round_dir"]),
            "plan": exp["manifest"]["planned_grid"],
            "probes": exp["result"].probes,
            "best": tree.summary()["best"],
            "selected_version": dream["selected"].name,
            "selected_beta": dream["selected"].best_beta,
            "reward": (None if dream["selected"].reward == float("-inf")
                       else round(dream["selected"].reward, 6)),
            "improved": json.loads(
                Path(dream["summary_path"]).read_text(encoding="utf-8"))["improved"],
        })
        if verbose:
            last = history[-1]
            print(f"[loop] round {last['round']}: best={last['best']} "
                  f"probes={last['probes']} policy={last['selected_version']} "
                  f"beta={last['selected_beta']} improved={last['improved']}",
                  flush=True)
    state["history"] = history
    layout.save_state(state)
    report = layout.root / "loop_report.json"
    report.write_text(json.dumps(history, indent=1, default=str), encoding="utf-8")
    return {"state": state, "history": history, "layout": str(layout.root),
            "report": str(report)}
