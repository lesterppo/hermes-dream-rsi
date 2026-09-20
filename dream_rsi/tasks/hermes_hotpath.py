"""Optimise a hot path in a real repository, hermetically.

The task copies ONE function out of a repository module (the repository file is
never touched), freezes its output on saved fixtures as the golden reference, and
asks the discovery agent for a faster `parse(...)` implementation that produces
byte-identical results.  Score = median(reference time) / median(candidate time),
gated on exact output equality, so the search cannot trade correctness for speed.

Spec:  hermes_hotpath:name=<label>,module=<path>,func=<fn>,fixtures=<dir>
"""
from __future__ import annotations

import ast
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import EvalOutcome, Task

HARNESS = r'''
import importlib.util, json, os, statistics, sys, time
from pathlib import Path

P = json.loads(os.environ["DREAMRSI_PARAMS"])
ATT = Path(os.environ["DREAMRSI_ATTEMPT"])
REPS = int(P.get("reps", 5))
ENTRY = P.get("entry", "parse")


def emit(score, valid, fail_class="ok", error=None, diagnostics=None,
         n_valid=None, n_total=None):
    print(json.dumps({"score": score, "valid": valid, "fail_class": fail_class,
                      "error": error, "diagnostics": diagnostics or {},
                      "n_valid": n_valid, "n_total": n_total}))
    sys.exit(0)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fixtures = [Path(p) for p in P["fixtures"]]
goldens = [json.loads(Path(p).read_text(encoding="utf-8")) for p in P["goldens"]]
pages = [p.read_text(encoding="utf-8", errors="ignore") for p in fixtures]

ref = load(P["ref"], "hotpath_ref")
ref_fn = getattr(ref, P["ref_func"], None)
if not callable(ref_fn):
    emit(-1e9, False, "env", f"reference function {P['ref_func']} missing")

src = ATT / "solution.py"
if not src.exists():
    emit(-1e9, False, "runtime", "solution.py not written")
try:
    cand = load(src, "hotpath_candidate")
except Exception as exc:
    emit(-1e9, False, "compile", f"{type(exc).__name__}: {exc}")
cand_fn = getattr(cand, ENTRY, None)
if not callable(cand_fn):
    emit(-1e9, False, "runtime",
         f"{ENTRY}(html) not defined - the entry point must keep its name and signature")

# correctness gate: exact equality with the frozen reference output, with a
# field-level diff so a repair attempt can act on the exact discrepancy
def diff(gold, got, prefix="", out=None, limit=8):
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(gold, list) and isinstance(got, list):
        if len(gold) != len(got):
            out.append(f"{prefix}: length {len(got)} != {len(gold)}")
        for i, (g, c) in enumerate(zip(gold, got)):
            diff(g, c, f"{prefix}[{i}]", out, limit)
    elif isinstance(gold, dict) and isinstance(got, dict):
        for key in gold:
            if key not in got:
                out.append(f"{prefix}.{key}: missing (expected {gold[key]!r})")
            else:
                diff(gold[key], got[key], f"{prefix}.{key}", out, limit)
        for key in got:
            if key not in gold and len(out) < limit:
                out.append(f"{prefix}.{key}: unexpected key")
    elif gold != got:
        out.append(f"{prefix}: got {got!r} want {gold!r}")
    return out[:limit]


problems = []
for path, page, golden in zip(fixtures, pages, goldens):
    try:
        got = cand_fn(page)
    except Exception as exc:
        emit(-1e9, False, "runtime", f"{path.name}: {type(exc).__name__}: {exc}")
    if json.dumps(got, sort_keys=True, default=str) != json.dumps(
            golden, sort_keys=True, default=str):
        for line in diff(golden, got, path.stem):
            problems.append(line)
        if len(problems) >= 8:
            break
if problems:
    emit(-1e9, False, "invalid",
         "output differs from the reference - first discrepancies: "
         + " | ".join(problems[:8]))


def timeit(fn):
    samples = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        for page in pages:
            fn(page)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


ref_t = timeit(ref_fn)
cand_t = timeit(cand_fn)
speedup = ref_t / max(cand_t, 1e-9)
emit(float(speedup), True, "ok", None,
     {"speedup": speedup, "ref_time_s": ref_t, "cand_time_s": cand_t,
      "pages": [p.name for p in fixtures], "bytes": sum(len(p) for p in pages),
      "reps": REPS},
     len(fixtures), len(fixtures))
'''


class HermesHotPathTask(Task):
    """Function-level performance task over frozen real-world fixtures."""

    name = "hermes_hotpath"
    eval_program = "solution.py"
    problem_file = "problem.md"
    default_branches = 4
    default_refines = 2
    default_workers = 4
    call_timeout = 900

    def __init__(self, label: str, module: str, func: str, fixtures: str,
                 entry: str = "parse", note: str = "", reps: int = 5) -> None:
        self.label = label
        self.module_path = Path(module).expanduser()
        self.func = func
        self.fixtures_dir = Path(fixtures).expanduser()
        self.entry = entry
        self.note = note
        self.reps = int(reps)
        self.name = f"hermes_hotpath_{label}"
        if not self.module_path.exists():
            raise FileNotFoundError(f"module not found: {self.module_path}")
        if not self.fixtures_dir.exists():
            raise FileNotFoundError(f"fixtures dir not found: {self.fixtures_dir}")
        self.cache = (Path.home() / ".hermes" / "dream_rsi_tasks"
                      / f"hotpath_{label}")
        self.ref_path = self.cache / "ref.py"
        self.golden_dir = self.cache / "goldens"
        self._fixtures = sorted(self.fixtures_dir.glob("*.html"))
        if not self._fixtures:
            raise FileNotFoundError(f"no *.html fixtures in {self.fixtures_dir}")
        self.ref_source, self.module_sha = self._extract_function()
        self._freeze()

    # ------------------------------------------------------------- extraction
    def _extract_function(self) -> tuple:
        source = self.module_path.read_text(encoding="utf-8")
        digest = hashlib.sha256(source.encode()).hexdigest()[:12]
        tree = ast.parse(source)
        lines = source.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name == self.func:
                start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                body = "\n".join(lines[start - 1:node.end_lineno])
                if node.col_offset:
                    indent = " " * node.col_offset
                    body = "\n".join(ln[len(indent):] if ln.startswith(indent) else ln
                                     for ln in body.splitlines())
                return body, digest
        raise ValueError(f"{self.func} not found in {self.module_path}")

    def _freeze(self) -> None:
        """Write the frozen reference implementation + golden outputs once."""
        self.cache.mkdir(parents=True, exist_ok=True)
        self.golden_dir.mkdir(parents=True, exist_ok=True)
        header = (f"# frozen copy of {self.module_path}::{self.func} "
                  f"(module sha256:{self.module_sha})\n"
                  f"# repository file is never modified by this task\n"
                  f"import re, math, json\nfrom html import unescape\n")
        self.ref_path.write_text(header + "\n\n" + self.ref_source + "\n",
                                 encoding="utf-8")
        if not all((self.golden_dir / f"{p.stem}.json").exists()
                   for p in self._fixtures):
            ns: Dict[str, Any] = {}
            exec(compile(self.ref_path.read_text(encoding="utf-8"),
                         str(self.ref_path), "exec"), ns)  # noqa: S102
            fn = ns[self.func]
            for page in self._fixtures:
                out = fn(page.read_text(encoding="utf-8", errors="ignore"))
                (self.golden_dir / f"{page.stem}.json").write_text(
                    json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------- interface
    def goldens(self) -> List[Path]:
        return [self.golden_dir / f"{p.stem}.json" for p in self._fixtures]

    def baseline_source(self) -> Optional[str]:
        return ("# reference implementation: "
                f"{self.module_path.name}::{self.func}\n"
                f"# keep this entry point name and signature; output must match\n"
                f"# module sha256:{self.module_sha}\n"
                f"import re, math, json\nfrom html import unescape\n\n\n"
                + self.ref_source + f"\n\n\n{self.entry} = {self.func}\n")

    def harness_params(self, attempt_dir: Path) -> Dict[str, Any]:
        return {
            "fixtures": [str(p) for p in self._fixtures],
            "goldens": [str(p) for p in self.goldens()],
            "ref": str(self.ref_path),
            "ref_func": self.func,
            "entry": self.entry,
            "reps": self.reps,
        }

    def evaluate(self, attempt_dir: Path) -> EvalOutcome:
        return self.run_harness(attempt_dir, HARNESS, timeout=self.call_timeout)

    def direction_hint(self, index: int) -> str:
        hints = [
            "single pass: one regex scan collecting every field per card instead of "
            "re-searching the same segment for each field",
            "avoid repeated tail copies: never slice html[pos:] per card (that is "
            "O(n^2) copying); index with match.endpos/pos arguments instead",
            "compile every pattern once at module import, not per call",
            "split the page into card segments in one pass, then extract fields from "
            "each segment with precompiled patterns",
            "parse bytes/str with finditer over precompiled alternations to reduce "
            "scanner restarts",
        ]
        return hints[index % len(hints)]

    # ------------------------------------------------------------- prompting
    @property
    def description(self) -> str:  # type: ignore[override]
        rel = self.module_path
        return (
            f"Accelerate one hot path in an existing tool without changing its "
            f"behaviour.\n\n"
            f"Reference implementation (`{rel}::{self.func}`, frozen copy below at "
            f"module sha256:{self.module_sha}):\n\n```python\n{self.ref_source}\n```\n\n"
            f"Your program `solution.py` must define `{self.entry}(html)` where "
            f"`html` is one Trip.com search page (0.5-1 MB of HTML) and return "
            f"exactly what the reference returns: a list of dicts with the same "
            f"keys, the same order, the same values (ids, names, star counts, "
            f"scores, review counts, prices, position snippets, URLs).\n\n"
            f"The evaluator replays {len(self._fixtures)} saved real pages, compares "
            f"your output to the frozen reference output with exact JSON equality, "
            f"then times both implementations over the same pages.\n\n"
            f"{self.note}"
        )

    @property
    def scoring(self) -> str:  # type: ignore[override]
        return (
            f"score = median(reference seconds) / median(your seconds) over "
            f"{len(self._fixtures)} pages, best of {self.reps} timed repetitions. "
            f"Any output difference from the reference on any page makes the attempt "
            f"invalid (score 0). Import-time work is measured, so parse lazily and "
            f"compile patterns once. Constraints: standard library only (re, json, "
            f"math, html, functools, operator, itertools, array are already "
            f"available), no network, no file I/O, do not modify the fixtures."
        )

    def branch_context(self, branch: int, attempt: int,
                       parent_summary: str = "") -> str:
        return (f"branch {branch} / attempt {attempt}\n"
                f"task: hot-path optimisation of {self.module_path.name}::"
                f"{self.func} ({len(self._fixtures)} frozen pages)\n"
                f"{parent_summary}".strip())
