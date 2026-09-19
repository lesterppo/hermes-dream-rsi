"""Lasso regularization path: the paper's algorithm-engineering task, in numpy.

Your program must define `lasso_path(X, y, lam_path) -> array (p, K)` giving the
Lasso solution path.  Correctness is gated on the objective value at every lambda
(the reference coordinate-descent solution is the bar); the score is the speedup
over the reference implementation on the same instances.  This mirrors the
paper's Lasso-path discovery task while staying compiler-free.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .base import EvalOutcome, Task

HARNESS = r'''
import json, os, sys, time
from pathlib import Path

import numpy as np

ATT = Path(os.environ["DREAMRSI_ATTEMPT"])
P = json.loads(os.environ.get("DREAMRSI_PARAMS", "{}"))
N = int(P.get("n", 900))
PDIM = int(P.get("p", 240))
K = int(P.get("k", 16))
ITERS = int(P.get("ref_iters", 220))
TOL = float(P.get("tol", 1e-3))
SEED = int(P.get("seed", 7))


def emit(score, valid, fail_class="ok", error=None, diagnostics=None,
         n_valid=None, n_total=None):
    print(json.dumps({"score": score, "valid": valid, "fail_class": fail_class,
                      "error": error, "diagnostics": diagnostics or {},
                      "n_valid": n_valid, "n_total": n_total}))
    sys.exit(0)


def make_data():
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((N, PDIM))
    X /= np.linalg.norm(X, axis=0, keepdims=True)
    beta_true = np.zeros(PDIM)
    idx = rng.choice(PDIM, size=max(5, PDIM // 20), replace=False)
    beta_true[idx] = rng.standard_normal(idx.size) * 2.0
    y = X @ beta_true + 0.1 * rng.standard_normal(N)
    y -= y.mean()
    lam_max = np.max(np.abs(X.T @ y)) / N
    lam_path = lam_max * np.logspace(0, -2.2, K)
    return X, y, lam_path


def soft(z, g):
    return np.sign(z) * np.maximum(np.abs(z) - g, 0.0)


def cd_path(X, y, lam_path, iters):
    """Reference coordinate descent along the path (warm starts)."""
    n, p = X.shape
    out = np.zeros((p, len(lam_path)))
    b = np.zeros(p)
    for k, lam in enumerate(lam_path):
        for _ in range(iters):
            b_old = b.copy()
            for j in range(p):
                r = y - X @ b + X[:, j] * b[j]
                rho = float(X[:, j] @ r) / n
                b[j] = soft(rho, lam) / 1.0
            if np.max(np.abs(b - b_old)) < 1e-6:
                break
        out[:, k] = b
    return out


def objectives(X, y, coefs, lam_path):
    n = X.shape[0]
    res = []
    for k, lam in enumerate(lam_path):
        b = coefs[:, k]
        res.append(float(np.sum((y - X @ b) ** 2) / (2 * n) + lam * np.abs(b).sum()))
    return res


X, y, lam_path = make_data()
t0 = time.perf_counter()
ref = cd_path(X, y, lam_path, ITERS)
ref_time = time.perf_counter() - t0
ref_obj = objectives(X, y, ref, lam_path)

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

fn = getattr(mod, "lasso_path", None)
if not callable(fn):
    emit(-1e9, False, "runtime", "lasso_path(X, y, lam_path) not defined")

t1 = time.perf_counter()
try:
    cand = np.asarray(fn(X.copy(), y.copy(), lam_path.copy()), dtype=float)
except Exception as exc:
    emit(-1e9, False, "runtime", f"candidate raised {type(exc).__name__}: {exc}")
cand_time = time.perf_counter() - t1

if cand.shape != (X.shape[1], len(lam_path)):
    emit(-1e9, False, "invalid",
         f"expected shape {(X.shape[1], len(lam_path))}, got {cand.shape}")

cand_obj = objectives(X, y, cand, lam_path)
ratios = [c / max(r, 1e-12) for c, r in zip(cand_obj, ref_obj)]
worst = max(ratios)
n_ok = sum(1 for r in ratios if r <= 1 + TOL)
if not n_ok and ratios:
    emit(-1e9, False, "invalid",
         f"objective worse than reference at every lambda (worst ratio {worst:.3f})",
         None, n_ok, len(ratios))
if worst > 1 + TOL:
    emit(-1e9, False, "invalid",
         f"objective ratio {worst:.4f} exceeds tolerance {1 + TOL:.4f}",
         None, n_ok, len(ratios))

speedup = ref_time / max(cand_time, 1e-9)
emit(float(speedup), True, "ok", None,
     {"speedup": speedup, "ref_time_s": ref_time, "cand_time_s": cand_time,
      "worst_obj_ratio": worst, "n": N, "p": PDIM, "k": K},
     n_ok, len(ratios))
'''

BASELINE = '''"""Baseline: naive warm-started coordinate descent (the reference speed = 1.0)."""

import numpy as np


def _soft(z, g):
    return np.sign(z) * np.maximum(np.abs(z) - g, 0.0)


def lasso_path(X, y, lam_path, iters=220):
    n, p = X.shape
    out = np.zeros((p, len(lam_path)))
    b = np.zeros(p)
    for k, lam in enumerate(lam_path):
        for _ in range(iters):
            b_old = b.copy()
            for j in range(p):
                r = y - X @ b + X[:, j] * b[j]
                rho = float(X[:, j] @ r) / n
                b[j] = _soft(rho, lam)
            if np.max(np.abs(b - b_old)) < 1e-6:
                break
        out[:, k] = b
    return out
'''


class LassoPathTask(Task):
    name = "lasso_path"
    eval_program = "solution.py"
    description = (
        "Implement `lasso_path(X, y, lam_path)` returning the Lasso regularization "
        "path: an array of shape (p, K) whose k-th column solves\n"
        "    minimise_b  ||y - X b||^2 / (2n) + lam_k * ||b||_1\n\n"
        "X is standardised (unit column norm), y is centred, and lam_path is "
        "decreasing from lam_max. The evaluator checks the objective value at every "
        "lambda against a reference coordinate-descent solve, then times both "
        "implementations on the same data. Faster is better, and numerical "
        "correctness is a hard gate."
    )
    scoring = (
        "score = reference_wall_clock / candidate_wall_clock (maximised). Any lambda "
        "whose objective exceeds the reference objective by more than 0.1% makes the "
        "attempt invalid. Exploit structure: active-set strategy, strong-rule "
        "screening, KKT/duality-gap pruning, lazy Gram updates, vectorised or "
        "blocked updates, warm starts along the path."
    )
    directions = [
        "active-set / working-set strategy with strong-rule screening",
        "vectorised or blocked coordinate updates instead of per-feature loops",
        "lazy Gram matrix construction with incremental residual updates",
        "Cauchy-Schwarz / KKT-based pruning of provably inactive features",
    ]
    default_branches = 4
    default_refines = 2
    default_workers = 4
    call_timeout = 900

    def __init__(self, n: int = 900, p: int = 240, k: int = 16) -> None:
        self.n, self.p, self.k = n, p, k
        self.name = f"lasso_path_n{n}_p{p}"

    def baseline_source(self) -> Optional[str]:
        return BASELINE

    def harness_params(self, attempt_dir: Path) -> Dict[str, Any]:
        return {"n": self.n, "p": self.p, "k": self.k}

    def evaluate(self, attempt_dir: Path) -> EvalOutcome:
        return self.run_harness(attempt_dir, HARNESS, timeout=self.call_timeout)
