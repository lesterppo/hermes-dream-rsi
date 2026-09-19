"""Circle packing: pack n equal circles in the unit square, maximize the radius.

Objective (maximize): r = min(0.5 * min pairwise distance, min wall clearance).
Known ceiling for n=10 is r = 0.14820432 (Packomania); the baseline grid layout
reaches 0.125, so this task has real room for discovery and needs no compiler.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .base import EvalOutcome, Task

HARNESS = r'''
import json, math, os, sys
from pathlib import Path

ATT = Path(os.environ["DREAMRSI_ATTEMPT"])
PARAMS = json.loads(os.environ.get("DREAMRSI_PARAMS", "{}"))
N = int(PARAMS.get("n", 10))


def emit(score, valid, fail_class="ok", error=None, diagnostics=None,
         n_valid=None, n_total=None):
    print(json.dumps({"score": score, "valid": valid, "fail_class": fail_class,
                      "error": error, "diagnostics": diagnostics or {},
                      "n_valid": n_valid, "n_total": n_total}))
    sys.exit(0)


src = ATT / "solution.py"
if not src.exists():
    emit(-1e9, False, "runtime", "solution.py not written")

try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("candidate_solution", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
except Exception as exc:
    emit(-1e9, False, "compile", f"{type(exc).__name__}: {exc}")

fn = getattr(mod, "pack", None)
if not callable(fn):
    emit(-1e9, False, "runtime", "pack(n) not defined")

try:
    pts = fn(N)
except Exception as exc:
    emit(-1e9, False, "runtime", f"pack raised {type(exc).__name__}: {exc}")

try:
    pts = [(float(p[0]), float(p[1])) for p in pts]
except Exception as exc:
    emit(-1e9, False, "runtime", f"bad point format: {exc}")

if len(pts) != N:
    emit(-1e9, False, "invalid", f"expected {N} points, got {len(pts)}")

eps = 1e-9
for (x, y) in pts:
    if not (math.isfinite(x) and math.isfinite(y)):
        emit(-1e9, False, "invalid", "non-finite coordinate")
    if x < -eps or x > 1 + eps or y < -eps or y > 1 + eps:
        emit(-1e9, False, "invalid", f"point outside unit square: ({x}, {y})")

n_pairs = 0
n_ok = 0
min_pair = float("inf")
for i in range(N):
    for j in range(i + 1, N):
        dx = pts[i][0] - pts[j][0]
        dy = pts[i][1] - pts[j][1]
        d = math.hypot(dx, dy)
        min_pair = min(min_pair, d)
        n_pairs += 1
        if d >= 2 * 1e-9:
            n_ok += 1

wall = min(min(x, y, 1.0 - x, 1.0 - y) for (x, y) in pts)
r = min(min_pair / 2.0, wall)

emit(float(r), r > 0.0, "ok", None,
     {"r": r, "min_pair_dist": min_pair, "wall_clearance": wall, "n": N},
     n_ok, n_pairs)
'''

BASELINE = '''"""Baseline: regular cell grid, one circle per cell (r = 0.125 for n<=16)."""


def pack(n=10):
    import math
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    step = 1.0 / max(cols, rows)
    r = step / 2.0
    pts = []
    for i in range(n):
        c, rr = i % cols, i // cols
        pts.append((r + c * step, r + rr * step))
    return pts
'''


class CirclePackingTask(Task):
    name = "circle_packing"
    eval_program = "solution.py"
    description = (
        "Pack N (default 10) equal, non-overlapping circles inside the unit "
        "square [0,1]^2 maximising their common radius.\n\n"
        "Your program must define `pack(n)` returning a list of n (x, y) pairs. "
        "The evaluator computes r = min(0.5 * minimum pairwise distance, "
        "minimum wall clearance) and rewards larger r.\n\n"
        "Reference points: the naive grid gives 0.125; the best known packing for "
        "n=10 is 0.14820432. Numeric local search, hexagonal arrangements, "
        "billiard/perturbation relaxation, and boundary-aware layouts are all "
        "legitimate mechanisms."
    )
    scoring = (
        "score = the realised radius r (maximised). Points outside the unit square, "
        "the wrong point count, or non-finite coordinates score as failures."
    )
    directions = [
        "hexagonal/lattice initialisation followed by numerical relaxation",
        "billiard or perturbation-based local search from multiple random starts",
        "force-directed (spring/repulsion) relaxation with wall constraints",
        "boundary-aware concentric or radial arrangement as a different structural prior",
    ]
    default_branches = 4
    default_refines = 2
    default_workers = 4
    call_timeout = 300

    def __init__(self, n: int = 10) -> None:
        self.n = int(n)
        self.name = f"circle_packing_n{self.n}"

    def baseline_source(self) -> Optional[str]:
        return BASELINE

    def harness_params(self, attempt_dir: Path) -> Dict[str, Any]:
        return {"n": self.n}

    def evaluate(self, attempt_dir: Path) -> EvalOutcome:
        return self.run_harness(attempt_dir, HARNESS, timeout=self.call_timeout)
