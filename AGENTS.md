# AGENTS.md — dream-rsi for AI agents

Token-efficient usage contract. Read this before touching the tool.

## What it is

Dream-RSI (arXiv 2609.14858): recursive self-improvement of *exploration*.
Discover (online) → freeze history as a replay simulator → dream candidate
policies over it → redeploy the best. The coding agent and evaluator never change.
The deployed policy is selected including the incumbent, so a cycle never regresses
on the fixed history.

## Cheapest useful calls

```
dream_rsi(action="tasks")                                  # tasks + agent back-ends
dream_rsi(action="status", run=RUN)                        # round, worlds, best vs baseline
dream_rsi(action="show", run=RUN, what="summary")          # last replay_summary.json
dream_rsi(action="show", run=RUN, what="sweep")            # per-beta auc/penalty/reward
dream_rsi(action="replay", run=RUN, betas="0.2,0.5,0.8")   # evaluate deployed policy, no LLM
dream_rsi(action="explore", run=RUN, agent="gemini", workers=4)
dream_rsi(action="dream", run=RUN, agent="deepseek", versions=2, betas="0.3,0.6,0.9")
dream_rsi(action="loop", run=RUN, task="circle_packing", rounds=2)
```

Every call returns pointer JSON: `{"ok":true,"t":9,"best":0.148204,"f":"<report path>"}`.
Read `f` only when you need the detail — never re-derive numbers from prose.

## Agent back-ends (pick by cost)

| spec | cost | use |
|---|---|---|
| `gemini[:flash\|pro]` | $0 (browser cookies) | default for exploration runs |
| `deepseek[:model]` | paid API | when the artefact is complex code |
| `dsh` | paid API | agentic backend, edits files itself |
| `openai:<base>\|<model>\|<keyfile>` | varies | any OpenAI-compatible endpoint |
| `cmd:<template>` | varies | custom command, `{prompt}`/`{file}`/`{dir}` |
| `mock` | free | deterministic offline; use before spending real calls |

## Rules

1. `init` before anything else (`--task circle_packing` is the cheap default).
2. `agent="mock"` runs the whole loop with zero network — use it to check a change
   to the machinery before spending real agent calls.
3. Cost control: online attempts = `branches x (refines+1)` per round; dreaming adds
   `versions` policy-development calls per round plus the replay sweep (free).
   `--branches N` needs `--workers >= N` (the paper's W is both branch count and
   concurrency) and is bounded by `--hard-branch` / `--hard-refine`.
4. `--repairs N` re-asks the discovery agent after an evaluator failure; keep N>=1
   for code-writing agents (placeholder/truncated code is the usual failure).
5. `--max-tokens 65536` for policy development; reasoning tokens share the budget,
   so a truncated reply (`finish_reason=length`) must be treated as a failed attempt
   (otherwise the template file is silently redeployed as a "new" policy).
6. Never edit files under a run dir by hand: `state.json`, `rounds/*/trace/tree.json`
   and the manifests are the simulator's ground truth.
7. Report honestly: `best <= baseline`, `improved: false`, or a negative
   `pareto.reward` means the cycle did not improve — say so instead of dressing up
   the numbers. Equal reward keeps the incumbent by design.

## Optimising your own repository

`hermes_hotpath` lifts one function out of a repo file with `ast`, freezes its
output on saved fixtures as the golden reference, and scores
`median(reference) / median(candidate)` gated on **exact** output equality. The
repository file is never modified; goldens live under `~/.hermes/dream_rsi_tasks/`
so upstream changes cannot drift the score.

```
dream-rsi loop --run R --task \
  "hermes_hotpath:label=<name>,module=<path.py>,func=<fn>,fixtures=<dir with *.html>"
```

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
policies/versions/*.py                        immutable copy of every deployed version
```

## Verification before claiming success

```
PYTHONPATH=. python3 -m pytest tests -q                        # 38 tests
python3 -m dream_rsi.cli loop --run /tmp/x --task circle_packing --agent mock \
  --rounds 1 --workers 2 --k1 6 --versions 1 --betas 0.5 --hard-branch 2 --hard-refine 1
python3 privacy_sweep.py                                       # must exit 0
```
