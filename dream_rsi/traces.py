"""On-disk artifact layout for a Dream-RSI run.

    <run>/
      state.json                          current round, deployed policy, task, grid
      baseline/                           baseline attempt used as floor
        proposal.md, <eval_program>, eval/score.json
      policies/
        current.py                        deployed exploration policy
        versions/v{m:03d}_beta{beta}.py   every evaluated version
      rounds/
        r0001_<ts>/
          method.py                       policy used for this live rollout
          live_cycle_manifest.json         one cycle's facts (prefix-safe)
          proposal_results/
            beta_sweep.json                per-beta AUC / penalty / reward
            policy_execution_traces.jsonl  one replay episode per (trace, beta)
            replay_summary.json            selected version + rationale
          policy_code/                     what the development agent wrote
          trace/
            tree.json
            attempts/b{branch}/a{attempt:04d}/
              proposal.md  <eval_program>  eval/{score.json,error.txt,stdout.txt}
          history -> ../../<prev round>/trace        (read-only prior trees)
      trace_pool/
        iter0001/live_cycle_manifest.json  sidecar of the live cycle
      proposals/                           policy-development transcripts
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RunLayout:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()

    # ------------------------------------------------------------- top level
    @property
    def state_file(self) -> Path:
        return self.root / "state.json"

    @property
    def baseline_dir(self) -> Path:
        return self.root / "baseline"

    @property
    def policies_dir(self) -> Path:
        return self.root / "policies"

    @property
    def versions_dir(self) -> Path:
        return self.policies_dir / "versions"

    @property
    def current_policy(self) -> Path:
        return self.policies_dir / "current.py"

    @property
    def rounds_dir(self) -> Path:
        return self.root / "rounds"

    @property
    def trace_pool(self) -> Path:
        return self.root / "trace_pool"

    @property
    def proposals_dir(self) -> Path:
        return self.root / "proposals"

    # ----------------------------------------------------------------- setup
    def ensure(self) -> "RunLayout":
        for p in (self.root, self.baseline_dir, self.versions_dir,
                  self.rounds_dir, self.trace_pool, self.proposals_dir):
            p.mkdir(parents=True, exist_ok=True)
        return self

    def round_dirs(self) -> List[Path]:
        if not self.rounds_dir.exists():
            return []
        return sorted(p for p in self.rounds_dir.iterdir() if p.is_dir())

    def new_round(self, index: int) -> Path:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        d = self.rounds_dir / f"r{index:04d}_{stamp}"
        for sub in ("proposal_results", "trace/attempts", "policy_code", "policy_input"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        return d

    def trace_paths(self) -> List[Path]:
        return [d / "trace" / "tree.json" for d in self.round_dirs()
                if (d / "trace" / "tree.json").exists()]

    def history_manifests(self) -> List[Dict[str, Any]]:
        out = []
        for d in self.round_dirs():
            m = d / "live_cycle_manifest.json"
            if m.exists():
                try:
                    out.append(json.loads(m.read_text(encoding="utf-8")))
                except Exception:  # noqa: BLE001
                    continue
        pool = self.trace_pool
        if pool.exists():
            for it in sorted(pool.iterdir()):
                m = it / "live_cycle_manifest.json"
                if m.exists():
                    try:
                        out.append(json.loads(m.read_text(encoding="utf-8")))
                    except Exception:  # noqa: BLE001
                        continue
        return out

    def link_history(self, round_dir: Path, prev_round: Optional[Path]) -> Optional[Path]:
        if prev_round is None:
            return None
        target = round_dir / "history"
        src = prev_round / "trace"
        if not src.exists():
            return None
        if target.exists() or target.is_symlink():
            try:
                target.unlink()
            except OSError:
                return target
        try:
            target.symlink_to(src, target_is_directory=True)
            return target
        except OSError:
            dst = round_dir / "history"
            shutil.copytree(src, dst, dirs_exist_ok=True)
            return dst

    # ----------------------------------------------------------------- state
    def load_state(self) -> Dict[str, Any]:
        if self.state_file.exists():
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        return {}

    def save_state(self, state: Dict[str, Any]) -> None:
        self.ensure()
        state = dict(state)
        state["updated_at"] = time.time()
        self.state_file.write_text(json.dumps(state, indent=1), encoding="utf-8")


def write_json(path: str | Path, payload: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    return p


def append_jsonl(path: str | Path, rows: List[Dict[str, Any]]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")
    return p


def live_manifest(task: str, round_index: int, plan: Dict[str, Any],
                  tree_summary: Dict[str, Any], beta: float,
                  probes: int, decision_rounds: int, policy: str,
                  extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Prefix-safe per-cycle facts (what plan_grid is allowed to read)."""
    man = {
        "task": task,
        "round": round_index,
        "policy": policy,
        "planned_grid": plan,
        "actual_within_support": {
            "branches": tree_summary.get("branches"),
            "max_attempts": tree_summary.get("max_attempts"),
        },
        "probes": probes,
        "decision_rounds": decision_rounds,
        "final_best": tree_summary.get("best"),
        "baseline": tree_summary.get("baseline"),
        "beta": beta,
        "created_at": time.time(),
    }
    if extra:
        man.update(extra)
    return man
