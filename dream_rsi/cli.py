"""Agent-native CLI: compact pointer JSON on stdout, full detail on disk.

    dream-rsi tasks
    dream-rsi init     --run R --task circle_packing
    dream-rsi explore  --run R [--agent deepseek] [--workers 4] [--k1 32]
    dream-rsi dream    --run R [--agent deepseek] [--versions 3] [--betas .2,.5,.8]
    dream-rsi replay   --run R --policy P [--betas .2,.5,.8]
    dream-rsi loop     --run R --task circle_packing --rounds 3
    dream-rsi status   --run R
    dream-rsi show     --run R [--what sweep|summary|manifest|tree|policy|state]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import agents as agents_mod
from .agents import make_agent
from .dream import DEFAULT_BETAS, evaluate_version, make_policy, select_best
from .loop import ensure_baseline, ensure_policy, run_dream, run_explore, run_loop
from .simulator import ObjectiveConfig
from .tasks import build_task, task_names
from .traces import RunLayout
from .tree import DiscoveryTree


def _emit(payload: Dict[str, Any], verbose: bool = False) -> int:
    if verbose:
        print(json.dumps(payload, indent=1, default=str))
    else:
        print(json.dumps(payload, default=str))
    return 0 if payload.get("ok", True) else 1


def _betas(raw: Optional[str]) -> List[float]:
    if not raw:
        return list(DEFAULT_BETAS)
    return [float(x) for x in raw.split(",") if x.strip()]


def _layout(run: str) -> RunLayout:
    return RunLayout(Path(run).expanduser()).ensure()


# --------------------------------------------------------------------------- #
# commands                                                                    #
# --------------------------------------------------------------------------- #
def cmd_tasks(args: argparse.Namespace) -> int:
    return _emit({"ok": True, "tasks": task_names(),
                  "agents": agents_mod.agent_list()})


def cmd_init(args: argparse.Namespace) -> int:
    layout = _layout(args.run)
    state = layout.load_state()
    task = build_task(args.task)
    state["task"] = args.task
    state.setdefault("round", 0)
    state["current_beta"] = args.beta
    layout.save_state(state)
    state = ensure_baseline(layout, task, state)
    ensure_policy(layout, state)
    return _emit({"ok": True, "run": str(layout.root), "task": args.task,
                  "baseline": state.get("baseline", {}),
                  "baseline_score": state.get("baseline_score"),
                  "policy": str(layout.current_policy)}, args.verbose)


def cmd_explore(args: argparse.Namespace) -> int:
    layout = _layout(args.run)
    state = layout.load_state()
    spec = args.task or state.get("task")
    if not spec:
        return _emit({"ok": False, "err": "no task: pass --task or run init"})
    task = build_task(spec)
    agent = make_agent(args.agent, timeout=args.timeout,
                       program_name=task.eval_program, task=task.name,
                       max_tokens=args.max_tokens)
    exp = run_explore(layout, state, task, agent, workers=args.workers,
                      k1=args.k1, hard_branch=args.hard_branch,
                      hard_refine=args.hard_refine, agent_timeout=args.timeout,
                      verbose=args.verbose, repairs=args.repairs,
                      plan={"branch_count": args.branches,
                            "refine_count": args.refines}
                      if (args.branches or args.refines) else None)
    tree: DiscoveryTree = exp["tree"]
    summary = tree.summary()
    report = exp["round_dir"] / "explore_report.json"
    report.write_text(json.dumps({
        "summary": summary, "manifest": exp["manifest"],
        "result": exp["result"].to_dict(), "timeline": exp["result"].timeline,
    }, indent=1), encoding="utf-8")
    return _emit({"ok": True, "t": tree.size(), "best": summary["best"],
                  "base": summary["baseline"], "p": exp["result"].probes,
                  "k": exp["result"].decision_rounds,
                  "fail": summary["failures"], "f": str(report),
                  "round": summary["round"]}, args.verbose)


def cmd_dream(args: argparse.Namespace) -> int:
    layout = _layout(args.run)
    state = layout.load_state()
    agent = make_agent(args.agent, timeout=args.timeout,
                       max_tokens=args.max_tokens)
    out = run_dream(layout, state, agent, versions=args.versions,
                    betas=_betas(args.betas), workers=args.workers, k2=args.k2,
                    cfg=ObjectiveConfig(lam=args.lam))
    sel = out["selected"]
    return _emit({"ok": True, "sel": sel.name, "beta": sel.best_beta,
                  "reward": None if sel.reward == float("-inf")
                  else round(sel.reward, 6),
                  "versions": [r.name for r in out["reports"]],
                  "worlds": len(layout.trace_paths()),
                  "f": out["summary_path"]}, args.verbose)


def cmd_replay(args: argparse.Namespace) -> int:
    """Pure replay evaluation of one policy (no LLM, no online rollout)."""
    layout = _layout(args.run)
    traces = [DiscoveryTree.load(p) for p in layout.trace_paths()]
    if not traces:
        return _emit({"ok": False, "err": "no discovery history in run dir"})
    policy_path = Path(args.policy).expanduser() if args.policy else layout.current_policy
    if not policy_path.exists():
        return _emit({"ok": False, "err": f"policy not found: {policy_path}"})
    reps = [evaluate_version(policy_path, traces, _betas(args.betas), args.workers,
                             ObjectiveConfig(lam=args.lam), args.k2)]
    best = select_best(reps) or reps[0]
    out = layout.root / "policies" / "replay_report.json"
    out.write_text(json.dumps([r.to_dict() for r in reps], indent=1),
                   encoding="utf-8")
    return _emit({"ok": best.ok, "pol": policy_path.name, "beta": best.best_beta,
                  "reward": None if best.reward == float("-inf")
                  else round(best.reward, 6),
                  "auc": round(best.auc, 6), "pen": round(best.penalty, 6),
                  "worlds": len(traces), "f": str(out),
                  "err": best.error}, args.verbose)


def cmd_loop(args: argparse.Namespace) -> int:
    out = run_loop(args.run, args.task, args.agent, rounds=args.rounds,
                   workers=args.workers, k1=args.k1, versions=args.versions,
                   betas=_betas(args.betas), k2=args.k2,
                   hard_branch=args.hard_branch, hard_refine=args.hard_refine,
                   agent_timeout=args.timeout, verbose=args.verbose,
                   repairs=args.repairs, cfg=ObjectiveConfig(lam=args.lam),
                   max_tokens=args.max_tokens)
    layout = _layout(args.run)
    report = layout.root / "loop_report.json"
    report.write_text(json.dumps(out["history"], indent=1), encoding="utf-8")
    state = out["state"]
    return _emit({"ok": True, "rounds": len(out["history"]),
                  "best": state.get("best_score"),
                  "baseline": state.get("baseline_score"),
                  "policy": state.get("selected_version"),
                  "beta": state.get("current_beta"),
                  "f": str(report)}, args.verbose)


def cmd_status(args: argparse.Namespace) -> int:
    layout = _layout(args.run)
    state = layout.load_state()
    rounds = layout.round_dirs()
    sweeps = [d / "proposal_results" / "beta_sweep.json" for d in rounds]
    sweeps = [p for p in sweeps if p.exists()]
    last_sel = None
    if sweeps:
        try:
            last_sel = json.loads(sweeps[-1].read_text(encoding="utf-8")).get("selected")
        except Exception:  # noqa: BLE001
            last_sel = None
    return _emit({
        "ok": True, "run": str(layout.root), "task": state.get("task"),
        "round": state.get("round", 0), "rounds_on_disk": len(rounds),
        "worlds": len(layout.trace_paths()),
        "baseline": state.get("baseline_score"),
        "best": state.get("best_score"), "best_cell": state.get("best_cell"),
        "policy": state.get("current_policy"), "beta": state.get("current_beta"),
        "sweeps": len(sweeps), "last_selected": last_sel,
    }, args.verbose)


def cmd_show(args: argparse.Namespace) -> int:
    layout = _layout(args.run)
    what = args.what
    rounds = layout.round_dirs()
    target: Dict[str, Any] = {}
    if what == "state":
        target = layout.load_state()
    elif what == "tree":
        paths = layout.trace_paths()
        tree = DiscoveryTree.load(paths[-1]) if paths else DiscoveryTree()
        target = {"summary": tree.summary(), "nodes": tree.compact_rows()}
    elif what == "manifest":
        man = layout.history_manifests()
        target = man[-1] if man else {}
    elif what == "sweep":
        sweeps = [d / "proposal_results" / "beta_sweep.json" for d in rounds]
        sweeps = [p for p in sweeps if p.exists()]
        target = json.loads(sweeps[-1].read_text(encoding="utf-8")) if sweeps else {}
    elif what == "summary":
        reps = [d / "proposal_results" / "replay_summary.json" for d in rounds]
        reps = [p for p in reps if p.exists()]
        target = json.loads(reps[-1].read_text(encoding="utf-8")) if reps else {}
    elif what == "policy":
        p = layout.current_policy
        target = {"path": str(p), "code": p.read_text(encoding="utf-8")
                  if p.exists() else ""}
    out = layout.root / f"show_{what}.json"
    out.write_text(json.dumps(target, indent=1, default=str), encoding="utf-8")
    return _emit({"ok": True, "f": str(out), "what": what}, args.verbose)


# --------------------------------------------------------------------------- #
# parser                                                                      #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dream-rsi",
                                description="Dream-RSI: recursive self-improvement "
                                            "through evolving worlds (arXiv 2609.14858)")
    p.add_argument("--verbose", action="store_true",
                   help="print the full payload instead of pointer JSON")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--run", required=True, help="run directory")
        sp.add_argument("--workers", type=int, default=4, help="parallel workers W")
        sp.add_argument("--betas", default="", help="beta grid, e.g. 0.2,0.5,0.8")
        sp.add_argument("--lam", type=float, default=1.0,
                        help="lambda in pareto.reward = auc - lam*parallel_penalty")
        sp.add_argument("--k2", type=int, default=64, help="replay round limit K2")
        sp.add_argument("--timeout", type=int, default=900, help="agent timeout s")
        sp.add_argument("--max-tokens", type=int, default=65536,
                        help="agent completion budget (thinking shares it; policy "
                             "development needs headroom)")
        sp.add_argument("--verbose", action="store_true")

    sp = sub.add_parser("tasks", help="list tasks and agent back-ends")
    sp.set_defaults(func=cmd_tasks)

    sp = sub.add_parser("init", help="prepare run dir: baseline + seed policy")
    sp.add_argument("--run", required=True)
    sp.add_argument("--task", default="circle_packing")
    sp.add_argument("--beta", type=float, default=0.6)
    sp.add_argument("--verbose", action="store_true")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("explore", help="one online discovery rollout")
    add_common(sp)
    sp.add_argument("--task", default="")
    sp.add_argument("--agent", default="deepseek")
    sp.add_argument("--k1", type=int, default=32, help="online round limit K1")
    sp.add_argument("--branches", type=int, default=0)
    sp.add_argument("--refines", type=int, default=0)
    sp.add_argument("--hard-branch", type=int, default=8)
    sp.add_argument("--hard-refine", type=int, default=6)
    sp.add_argument("--repairs", type=int, default=1,
                    help="re-ask the discovery agent after a failed attempt")
    sp.set_defaults(func=cmd_explore)

    sp = sub.add_parser("dream", help="dreaming-based policy improvement")
    add_common(sp)
    sp.add_argument("--agent", default="deepseek")
    sp.add_argument("--versions", type=int, default=3, help="candidate versions M")
    sp.set_defaults(func=cmd_dream)

    sp = sub.add_parser("replay", help="replay-evaluate a policy file (no LLM)")
    add_common(sp)
    sp.add_argument("--policy", default="")
    sp.set_defaults(func=cmd_replay)

    sp = sub.add_parser("loop", help="full RSI loop: explore/dream/redeploy")
    add_common(sp)
    sp.add_argument("--task", required=True)
    sp.add_argument("--agent", default="deepseek")
    sp.add_argument("--rounds", type=int, default=3)
    sp.add_argument("--versions", type=int, default=3)
    sp.add_argument("--k1", type=int, default=32)
    sp.add_argument("--hard-branch", type=int, default=8)
    sp.add_argument("--hard-refine", type=int, default=6)
    sp.add_argument("--repairs", type=int, default=1,
                    help="re-ask the discovery agent after a failed attempt")
    sp.set_defaults(func=cmd_loop)

    sp = sub.add_parser("status", help="run status")
    sp.add_argument("--run", required=True)
    sp.add_argument("--verbose", action="store_true")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("show", help="dump one artifact")
    sp.add_argument("--run", required=True)
    sp.add_argument("--what", default="summary",
                    choices=["state", "tree", "manifest", "sweep", "summary", "policy"])
    sp.add_argument("--verbose", action="store_true")
    sp.set_defaults(func=cmd_show)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:  # noqa: BLE001
        return _emit({"ok": False, "err": f"{type(exc).__name__}: {exc}"},
                     getattr(args, "verbose", False))


if __name__ == "__main__":
    sys.exit(main())
