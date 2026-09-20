---
name: dream-rsi
description: Use when running Dream-RSI recursive self-improvement (arXiv 2609.14858) — online discovery rollout, history as a replay simulator, dreaming-based exploration-policy improvement, beta sweep and pareto selection.
version: 1.0.0
author: Peter (lesterppo)
license: MIT
tags: [dream-rsi, rsi, self-improvement, exploration, replay-simulator, world-model, discovery, arxiv-2609.14858]
---

# Dream-RSI — recursive self-improvement through evolving worlds

Implementation of Zheng et al., *Dream-RSI: Recursive Self-Improvement through
Evolving Worlds* (arXiv 2609.14858), shipped as the `dream_rsi` Hermes tool plus a
standalone CLI. No reference code was released with the paper, so the loop was
implemented against the paper's method section and its two appendix prompts.

## Where things live

| Thing | Path |
|---|---|
| Package + CLI | `~/.hermes/scripts/dream_rsi/` (`bin/dream-rsi`, symlink `~/.local/bin/dream-rsi`) |
| Hermes tool | `~/.hermes/plugins/hermes_local_tools/dream_rsi_tool.py`, toolset `dream_rsi` |
| Runs (artifacts) | run dir passed to `--run`, e.g. `~/.hermes/dream_rsi_runs/<name>/` |
| Tests | `~/.hermes/scripts/dream_rsi/tests/` (33 tests, `PYTHONPATH=. pytest tests -q`) |

## The loop (what each piece is)

1. **Plan grid** — `plan_grid(context) -> GridPlan(W, R)`. Runs *before* a new live
   grid exists; may only read completed-cycle manifests (prefix-safe facts).
2. **Online explore** — the deployed policy picks batches of cells
   (`b<branch>#a<attempt>`), each selected cell is one generation-evaluation
   attempt by the discovery agent, scored by the task evaluator. Tree saved to
   `rounds/rNNNN_*/trace/tree.json`, manifest + `trace_pool/iterNNNN/` sidecar.
3. **Replay simulator** — the finished tree is a frozen, irregular branch×attempt
   grid. `ReplayQuestion` reveals recorded outcomes for selected cells only;
   probing beyond the recording reveals nothing (exhausted). The policy sees the
   revealed prefix only — unrevealed scores are unreachable through the API.
4. **Dreaming** — each candidate policy version is replayed over *every* world for
   every beta in the grid; ranked by
   `pareto.reward = pareto.auc - lambda * parallel_penalty`, where
   `parallel_penalty = mean(effective_sequential_rounds / total_probes)` and a
   batch of `k` with `W` workers costs `ceil(k/W)` effective sequential rounds.
5. **Redeploy** — best version (including the current one, so the selection is
   never worse on the fixed history) becomes `policies/current.py`; its best beta
   is baked in as the default for the next live cycle.

Eq. (1) objective, reported alongside: `max s_v - beta1*N + beta2*N/max(1,k)`.

## Doing a real repository hot path (`hermes_hotpath`)

Lifts one function out of a repo file with `ast`, freezes its output on read-only
fixtures as the golden reference, and scores `median(ref)/median(candidate)` gated on
**exact** output equality. The repository file is never modified. Verified live on
`ota.py::_parse_trip_cards`: 18/18 attempts byte-identical, best **3.36x** speedup
(5.54 ms → 1.65 ms per 5-page sweep), and the round-2 dream improved the policy
(0.196080 → 0.203183). Keep spec `note=` free of commas.

## Commands

```bash
dream-rsi init    --run R --task circle_packing
dream-rsi explore --run R --agent deepseek --workers 3 --branches 3 --refines 2 --repairs 2
dream-rsi dream   --run R --agent deepseek --versions 2 --betas 0.3,0.6,0.9
dream-rsi replay  --run R [--policy P] [--betas 0.2,0.5,0.8]     # no LLM, no rollout
dream-rsi loop    --run R --task circle_packing --rounds 2 --repairs 2
dream-rsi status  --run R
dream-rsi show    --run R --what sweep|summary|manifest|tree|policy|state
```

Hermes tool equivalents: `dream_rsi(action=..., run=..., task=..., agent=...)` with
actions `tasks|init|explore|dream|replay|loop|status|show|policy`. Pointer JSON out;
`show` inlines the artifact body.

## Agent back-ends

`deepseek[:model]` (default; key from `$DEEPSEEK_API_KEY` or `~/.dsh/.env`),
`gemini[:flash|pro|thinking|lite]` (**zero API cost** — drives the Gemini web CLI
on browser cookies; stdout is pointer JSON so the template writes the reply to a
file and cats it back for the FILE protocol). Live check: `explore --agent gemini
--branches 2 --refines 1` produced 4 attempts, 0 failures, best 0.148204 = the
published n=10 optimum, with no paid API involved.
`dsh` (DeepSeek Harness headless — the agent edits files itself),
`openai:<base>|<model>|<keyfile>`, `cmd:<template>` (`{prompt}`/`{file}`/`{dir}`),
`mock` (deterministic offline — the entire loop runs with no network, useful for
regression runs and for proving the machinery).

## Pitfalls learned the hard way

- **Reasoning models share the completion budget.** DeepSeek V4 Flash spends
  reasoning tokens from the same `max_tokens`; at 8192 the emitted program was
  truncated mid-file (`pack(n) not defined` on every attempt), and at 32768 the
  *policy-development* reply still hit `finish_reason: length`. Use 32768 for
  discovery attempts and 65536 for policy development, and treat
  `finish_reason == "length"` as a failed development attempt that is retried
  with a "return the complete file, shorter" instruction.
- **Long unary completions get cut (IncompleteRead).** The policy-improvement
  prompt is ~16 KB and the reply is tens of thousands of tokens; non-streamed
  requests died with `IncompleteRead(0 bytes read)`, which killed the whole
  dreaming phase. The DeepSeek back-end now streams (`stream=true`, SSE deltas
  accumulated, partial text preserved) and retries transient failures 3x with
  backoff. Streaming alone took a 115 s policy generation from "always fails" to
  "finish_reason: stop".
- **AUC must be measured against the policy's own committed work.**
  `x = cumulative probes / probes spent in that episode`, so a policy that reaches
  a high score early in its own trajectory scores high, and probing past a stalled
  frontier drags the score down. Normalizing x by `trace.size()` instead rewards
  *exhausting* the grid (the flat full-height tail fills the area), which is the
  opposite of the paper's "few total probes" intent — measured both ways on a
  15-node fixture: exhausting scored 0.69 vs 0.15, while the intended reading is
  that widening early and stopping on a plateau is what pays.
- **Model text can masquerade as a FILE header.** A Gemini reply echoed the prompt
  and the header regex captured a long prose/math string as the "path", producing
  `OSError: File name too long` that killed the whole rollout (`policy-error`, 0
  nodes). Paths are now sanitized (`_safe_rel_path`: charset + length + no `..`
  segments; absolute paths are tolerated and remapped by basename) and every
  per-file write is wrapped so one bad path cannot abort an episode.
- **FILE paths must stay relative.** Models emit the absolute workspace path in
  the FILE header; the naive `lstrip('/')` turned that into a nested directory and
  the real target stayed untouched, so the developed "policy" was byte-identical
  to the template while validation passed. `extract_files(..., expected_name=)`
  now remaps by basename, and `develop_policy` rejects a delivered file that still
  equals the template.
- **Stalled streams hang the cycle.** A provider that goes silent mid-stream holds
  the socket for the full timeout. The back-end arms a 240 s socket idle timeout
  and treats a stream that ends without `finish_reason` as a cut attempt that is
  retried.
- **A chat model with no tools will still emit tool calls.** DeepSeek replied with
  DSML `invoke name="shell"` markup when asked to write a file. Two fixes are both
  required: a no-tools system prompt (`agents.NO_TOOLS_SYSTEM`) and
  `strip_tool_markup()` before file extraction.
- **Placeholder code is the default failure mode.** Models write `...`, `?`, and
  "TODO" into the artifact. The exploration prompt now forbids placeholders
  explicitly and `--repairs N` re-asks the agent with the evaluator's error text
  (fail_class compile/runtime/timeout/invalid/agent/env are repairable).
- **Policies must be importable as files.** Anything deployed to
  `policies/current.py` is loaded by path in a separate process — use absolute
  imports (`from dream_rsi.policy_api import ...` or the paper's
  `from see.policy.api import ...`), never relative imports.
- **`plan_grid` is a hard requirement.** The paper's prompt insists the policy
  return an explicit `GridPlan` on every path; `validate_policy` rejects a version
  whose `plan_grid` is missing, returns None, or raises.
- **Illegal batches must be rejected before deployment.** Duplicates,
  parent+child in one batch, > W cells, invented cells: `validate_policy` replays
  the candidate once and refuses versions that violate the interface.
- **A version whose reward ties the current one is fine** — selection is
  `argmax` including the current policy, which is what guarantees monotone
  non-regression on the fixed history.

## Interpreting results

- `best` above `baseline` means the discovery agent actually found something;
  `pareto.reward` near or below zero means replay never beat the baseline floor
  (AUC clamps at 0 below baseline) — check the artifacts before blaming the policy.
- `proposal_results/replay_summary.json` holds `improved` (selected != current),
  the developed versions and the current reward — the fastest health check.
- `policy_execution_traces.jsonl` is one replay episode per (version, trace, beta)
  with the full round-by-round timeline: serial batches, premature stops and
  wasted probes are visible there.
- `live_cycle_manifest.json` per round is the prefix-safe fact sheet; the improved
  policy's `plan_grid` is expected to read it.

## Verification recipe

```bash
cd ~/.hermes/scripts/dream_rsi && PYTHONPATH=. python3 -m pytest tests -q   # 33 pass
python3 -m dream_rsi.cli loop --run /tmp/dr_mock --task circle_packing \
  --agent mock --rounds 2 --workers 2 --k1 6 --versions 1 --betas 0.5 --k2 12 \
  --hard-branch 2 --hard-refine 1          # full loop, zero network
```
