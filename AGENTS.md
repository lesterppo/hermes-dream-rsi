# AGENTS.md — dream-rsi for AI agents

Token-efficient usage contract. Read this before touching the tool.

## What it is

Dream-RSI (arXiv 2609.14858): recursive self-improvement of *exploration*.
Discover (online) → freeze history as a replay simulator → dream candidate
policies over it → redeploy the best. The coding agent and evaluator never change.

## Cheapest useful calls

```
dream_rsi(action="tasks")                                  # tasks + agent back-ends
dream_rsi(action="status", run=RUN)                        # round, worlds, best vs baseline
dream_rsi(action="show", run=RUN, what="summary")          # last replay_summary.json
dream_rsi(action="show", run=RUN, what="sweep")            # per-beta auc/penalty/reward
dream_rsi(action="replay", run=RUN, betas="0.2,0.5,0.8")   # evaluate deployed policy, no LLM
dream_rsi(action="explore", run=RUN, agent="deepseek", workers=3)
dream_rsi(action="dream", run=RUN, agent="deepseek", versions=2, betas="0.3,0.6,0.9")
dream_rsi(action="loop", run=RUN, task="circle_packing", rounds=2)
```

Every call returns pointer JSON: `{"ok":true,"t":9,"best":0.148195,"f":"<report path>"}`.
Read `f` only when you need the detail — never re-derive numbers from prose.

## Rules

1. `init` before anything else (`--task circle_packing` is the cheap default).
2. `agent="mock"` runs the whole loop with zero network — use it to check a change
   to the machinery before spending real agent calls.
3. Cost control: online attempts = `branches x (refines+1) x rounds`; dreaming adds
   `versions` policy-development calls per round plus the replay sweep (free).
4. `--repairs N` re-asks the discovery agent after an evaluator failure; keep N≥1
   for code-writing agents (placeholder/truncated code is the usual failure).
5. Never edit files under a run dir by hand: `state.json`, `rounds/*/trace/tree.json`
   and the manifests are the simulator's ground truth.
6. Report honestly: `best <= baseline` or a negative `pareto.reward` means the
   cycle did not improve — say so instead of dressing up the numbers.

## Artifacts (per round)

```
rounds/rNNNN_<ts>/trace/tree.json             discovery tree (nodes = attempts)
rounds/rNNNN_<ts>/trace/attempts/b*/a*/       workspace: proposal.md, program, eval/score.json
rounds/rNNNN_<ts>/live_cycle_manifest.json    prefix-safe cycle facts (plan, probes, best, beta)
rounds/rNNNN_<ts>/proposal_results/beta_sweep.json
rounds/rNNNN_<ts>/proposal_results/policy_execution_traces.jsonl   one line per (version, trace, beta)
rounds/rNNNN_<ts>/proposal_results/replay_summary.json             selected version + improved flag
trace_pool/iterNNNN/live_cycle_manifest.json  sidecars plan_grid may read
policies/current.py                           deployed exploration policy
```
