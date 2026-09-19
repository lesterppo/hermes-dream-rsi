# dream_rsi — Dream-RSI for Hermes (arXiv 2609.14858)

Recursive self-improvement through evolving worlds, implemented against the paper's
own interface so it runs both as a Hermes tool and as a standalone CLI.

    discovery history  ->  replay simulator ("world")  ->  dreaming  ->  redeploy

* online rollout: the deployed exploration policy guides a coding agent over a
  branch x attempt grid and every attempt is recorded with its measured score
* simulator construction: the finished tree becomes a frozen, prefix-only world
* dreaming: M candidate policy versions are replayed over **every** world while
  the single `beta` knob is swept; the best version is redeployed next cycle

Only the exploration-policy code changes. The discovery agent, evaluator and
execution interfaces stay fixed — zero gradient steps on the coding agent.

## Paper mapping

| Paper element | Implementation |
|---|---|
| discovery tree, node = attempt with workspace + artifact + score + diagnostics | `tree.py` (`DiscoveryTree`, `Node`) |
| decision interface `A(T,W)`, legal batches of size ≤ W | `simulator.py` (`ReplayQuestion`, `LiveQuestion`) |
| replay reveals recorded children deterministically; irregular grid | `ReplayQuestion.probe_batch` (exhausted cells reveal nothing) |
| replay objective `max s_v − β₁N + β₂N/max(1,k)` | `simulator.replay_score` |
| sweep ranking `pareto.reward = pareto.auc − λ·parallel_penalty` | `simulator.pareto_auc` / `parallel_penalty` / `pareto_reward` |
| exploration prompt (App. B.1) | `prompts.EXPLORATION_PROMPT` |
| replay-based policy-improvement prompt (App. B.2) | `prompts.POLICY_IMPROVEMENT_PROMPT` |
| policy API `question.observed()/legal_actions()/probe_batch()/meta()` | `simulator` + `see/policy/api.py` shim |
| `plan_grid(context) -> GridPlan(W, R)` next-cycle planning | `grid.py`, `policy_api.plan_grid` |
| per-cycle artifacts (`live_cycle_manifest.json`, `beta_sweep.json`, `policy_execution_traces.jsonl`, `trace_pool/iter*/`) | `traces.py`, `dream.dream_phase` |

Prefix-only enforcement: `ReplayQuestion` never exposes unrevealed scores, and
`validate_policy` rejects any policy whose batch is illegal (duplicates,
parent+child together, > W, invented cells) before it can be deployed.

## CLI

    dream-rsi tasks
    dream-rsi init    --run R --task circle_packing
    dream-rsi explore --run R --agent deepseek --workers 3 --branches 3 --refines 2
    dream-rsi dream   --run R --agent deepseek --versions 2 --betas 0.3,0.6,0.9
    dream-rsi replay  --run R [--policy P] [--betas 0.2,0.5,0.8]
    dream-rsi loop    --run R --task circle_packing --rounds 3
    dream-rsi status  --run R
    dream-rsi show    --run R --what sweep|summary|manifest|tree|policy|state

Pointer JSON on stdout (`{"ok":true,"t":9,"best":0.1482,"f":"..."}`); full reports
on disk under the run dir. `--verbose` prints everything inline.

Agents: `deepseek[:model]` (default), `dsh` (agentic, edits files itself),
`openai:<base>|<model>|<keyfile>`, `cmd:<template>` (with `{prompt}`/`{file}`/`{dir}`),
`mock` (deterministic offline — the whole loop runs with no network).

## Tasks

* `circle_packing` — pack 10 equal circles in the unit square, maximise the radius.
  Baseline 0.125, best known 0.14820432. Scored by an isolated subprocess harness.
* `lasso_path` — implement `lasso_path(X, y, lam_path)`; objective-gated against a
  reference coordinate-descent solve, score = wall-clock speedup.

Both are real: the evaluator runs the candidate program in a separate interpreter
with a timeout, writes `eval/score.json`, and records `fail_class`
(`ok|compile|runtime|invalid|timeout|agent|env`).

## Python use

```python
from dream_rsi.loop import run_loop
run_loop("~/runs/demo", "circle_packing", "deepseek", rounds=3, workers=4)

from dream_rsi.dream import evaluate_version
from dream_rsi.tree import DiscoveryTree
evaluate_version("policies/current.py", [DiscoveryTree.load("rounds/r0001_x/trace/tree.json")],
                 betas=[0.2, 0.5, 0.8], workers=4)
```

Policies are ordinary Python modules — the paper's import path works unchanged:

```python
from see.policy.api import LLMDesignedMethod, GridPlan, GridPlanningContext

NAME = "OptimalPolicy"

class OptimalPolicy(LLMDesignedMethod):
    def plan_grid(self, context): return GridPlan(4, 3, "reason")
    def select_batch(self, question): return question.legal_actions()[:question.max_parallelism]
```

## Tests

    PYTHONPATH=. python3 -m pytest tests -q      # 33 tests

Covers tree round-trip, replay legality (duplicates / parent+child / oversize /
invented cells), prefix-only visibility, exhausted-cell semantics, determinism,
Eq. (1) arithmetic, AUC + parallel-penalty arithmetic, beta-schedule monotonicity,
policy-batch legality for both shipped policies, harness failure classes, task
baselines, policy validation rejection, sweep selection (selected reward ≥ current),
development retries, the full offline loop artifact set, and CLI pointer JSON.

## Author

Peter / [lesterppo](https://github.com/lesterppo). Paper: Zheng et al.,
*Dream-RSI: Recursive Self-Improvement through Evolving Worlds*, arXiv:2609.14858.
