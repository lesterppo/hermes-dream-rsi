"""
dream_rsi — Dream-RSI recursive self-improvement (arXiv 2609.14858) for Hermes.

Turns a completed discovery history into a *replay simulator* ("world") in which
candidate exploration policies are evaluated offline by dreaming, then redeploys
the best policy online — a self-improvement loop at the exploration layer that
leaves the underlying coding agent unchanged.

Actions
  tasks    list discovery tasks + agent back-ends (deepseek | dsh | mock | cmd:…)
  init     prepare a run dir: baseline attempt (floor score) + seeded policy
  explore  one online discovery rollout with the deployed policy (live tree)
  dream    replay-evaluate the policy + M new versions over every frozen world,
           sweep beta, select the best by pareto.reward = auc - lambda*penalty
  replay   pure replay evaluation of one policy file (no LLM, no rollout)
  loop     the full RSI loop: explore -> simulate -> dream -> redeploy, T rounds
  status   run state: round, worlds, baseline/best score, deployed policy + beta
  show     dump one artifact (state|tree|manifest|sweep|summary|policy)
  policy   path + code of the currently deployed exploration policy

Agent-native: stdout is pointer JSON ({"ok":true,"f":"<report>",…}); full detail
lives on disk under the run dir (rounds/, trace_pool/, policies/, state.json).

Author: Peter/lesterppo
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

MAX_INLINE = 6000


def _hermes_home() -> Path:
    env = os.environ.get("HERMES_HOME")
    return Path(env) if env else Path.home() / ".hermes"


def _cli() -> Optional[Path]:
    candidates = [
        Path.home() / ".hermes" / "scripts" / "dream_rsi" / "bin" / "dream-rsi",
        Path.home() / ".local" / "bin" / "dream-rsi",
    ]
    for c in candidates:
        if c.exists():
            return c
    found = shutil.which("dream-rsi")
    return Path(found) if found else None


def _check() -> bool:
    cli = _cli()
    if cli is None:
        return False
    try:
        import numpy  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


_ACTIONS = ("tasks", "init", "explore", "dream", "replay", "loop", "status",
            "show", "policy")

SCHEMA: Dict[str, Any] = {
    "name": "dream_rsi",
    "description": (
        "Dream-RSI (arXiv 2609.14858) recursive self-improvement for discovery "
        "tasks: online rollout builds a discovery tree, history becomes a replay "
        "simulator, candidate exploration policies are improved by dreaming over "
        "that simulator, and the best policy is redeployed. Actions: tasks "
        "(list tasks/agents), init (run dir + baseline + seeded policy), explore "
        "(one live rollout), dream (replay + beta sweep + policy selection), "
        "replay (evaluate a policy file offline), loop (full explore->dream-"
        ">redeploy over T rounds), status, show, policy. Pointer JSON out; full "
        "reports on disk."),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(_ACTIONS)},
            "run": {"type": "string",
                    "description": "run directory (created by init)"},
            "task": {"type": "string",
                     "description": "circle_packing | lasso_path "
                                    "(or e.g. circle_packing:n=12)"},
            "agent": {"type": "string",
                      "description": "deepseek (default) | dsh | mock | "
                                     "openai:<base>|<model>|<keyfile> | cmd:<tpl>"},
            "rounds": {"type": "integer", "description": "loop: outer rounds T"},
            "workers": {"type": "integer", "description": "parallel workers W"},
            "k1": {"type": "integer", "description": "online decision rounds K1"},
            "k2": {"type": "integer", "description": "replay decision rounds K2"},
            "versions": {"type": "integer",
                         "description": "dream: candidate policy versions M"},
            "betas": {"type": "string",
                      "description": "beta sweep grid, e.g. 0.3,0.6,0.9"},
            "branches": {"type": "integer", "description": "explore: grid width"},
            "refines": {"type": "integer", "description": "explore: grid depth"},
            "policy": {"type": "string",
                       "description": "replay: policy file (default: deployed)"},
            "what": {"type": "string",
                     "enum": ["state", "tree", "manifest", "sweep", "summary",
                              "policy"]},
            "timeout": {"type": "integer", "description": "agent timeout (s)"},
        },
        "required": ["action"],
    },
}


def _cli_args(args: Dict[str, Any]) -> list:
    action = str(args.get("action", "")).strip()
    if action not in _ACTIONS:
        return []
    argv = [action]
    if action == "policy":
        argv = ["show", "--what", "policy"]
    if action in ("init", "explore", "dream", "replay", "loop", "status",
                  "show", "policy"):
        run = args.get("run")
        if not run:
            return []
        argv += ["--run", str(Path(str(run)).expanduser())]
    for key, flag in (("task", "--task"), ("agent", "--agent"),
                      ("workers", "--workers"), ("k1", "--k1"), ("k2", "--k2"),
                      ("versions", "--versions"), ("betas", "--betas"),
                      ("branches", "--branches"), ("refines", "--refines"),
                      ("policy", "--policy"), ("what", "--what"),
                      ("timeout", "--timeout"), ("rounds", "--rounds")):
        val = args.get(key)
        if val in (None, "", 0):
            continue
        argv += [flag, str(val)]
    return argv


def _read_pointer(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload)
    f = out.get("f")
    if f and Path(str(f)).exists():
        try:
            text = Path(str(f)).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            text = ""
        if text and len(text) <= MAX_INLINE and out.get("action") in ("show",):
            out["body"] = text
    return out


def handle(args: Dict[str, Any]) -> Dict[str, Any]:
    action = str(args.get("action", "")).strip()
    if action not in _ACTIONS:
        return {"error": f"unknown action {action!r}; expected one of "
                         f"{list(_ACTIONS)}"}
    cli = _cli()
    if cli is None:
        return {"error": "dream-rsi CLI not found "
                         "(expected ~/.hermes/scripts/dream_rsi/bin/dream-rsi)"}
    argv = _cli_args(args)
    if not argv:
        return {"error": "action needs a --run directory (see action='init')"}
    timeout = int(args.get("timeout") or 1800)
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(Path.home() / ".hermes" / "scripts" /
                                     "dream_rsi"))
    try:
        proc = subprocess.run([str(cli), *argv], capture_output=True, text=True,
                              timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"error": f"dream-rsi {action} timed out after {timeout}s"}
    line = ""
    for candidate in reversed((proc.stdout or "").splitlines()):
        if candidate.strip().startswith("{"):
            line = candidate.strip()
            break
    if not line:
        return {"error": f"dream-rsi {action} produced no JSON",
                "rc": proc.returncode, "stderr": (proc.stderr or "")[-500:]}
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        return {"error": f"bad JSON from dream-rsi: {exc}",
                "raw": line[:500]}
    payload["action"] = action
    if not payload.get("ok", True):
        payload.setdefault("error", payload.get("err"))
    return _read_pointer(payload)


def _register() -> None:
    try:
        from tools.registry import registry
    except Exception:  # noqa: BLE001 — tool imported outside Hermes (tests)
        return
    try:
        registry.register(
            name="dream_rsi", toolset="dream_rsi", schema=SCHEMA,
            handler=handle, check_fn=_check, emoji="\U0001F319",
            is_async=False,
        )
    except TypeError:
        registry.register(name="dream_rsi", toolset="dream_rsi", schema=SCHEMA,
                          handler=handle, check_fn=_check)


_register()
