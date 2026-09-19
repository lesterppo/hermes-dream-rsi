# Dream-RSI integration — live verification, 2026-09-19

Agent: DeepSeek `deepseek-v4-flash` over the streaming chat back-end (key from
`~/.dsh/.env`). Task: `circle_packing` (n=10, unit square, maximise the common
radius; baseline grid = 0.125, known optimum = 0.14820432, Packomania).
Run dir: `~/.hermes/dream_rsi_runs/circle/`.

## What was verified live

| Check | Evidence |
|---|---|
| discovery works | round 1 (3 branches x 3 attempts, W=3): 0.14695 / **0.148157** / 0.147803, 0 failures |
| refinement works | round 1 refinements reached 0.148195; round 2 reached **0.148199** |
| hits the known ceiling | round 3, branch 2 attempt 0 emitted **0.148204** = the published optimum for n=10 |
| integrity of the grid | round 3 deep grid (W=3, R=4, 15 cells) completed in 5 decision rounds, `fail=0` |
| replay determinism | identical `SimResult` across repeated episodes (unit test) |
| beta discrimination | identical AUC across betas but distinct penalties per policy (m001 0.556 vs current 0.444) |
| no-regression selection | every cycle selected the incumbent when developed versions tied or lost (`improved: false`, `selected == current`) |
| artifacts | per round: `live_cycle_manifest.json`, `proposal_results/{beta_sweep.json, policy_execution_traces.jsonl, replay_summary.json}`, `trace_pool/iterNNNN/`, `policies/current.py` |
| CLI + tool | pointer JSON from every subcommand; `dream_rsi` tool registered in the `dream_rsi` toolset (`registry.get_entry` + handler dispatch verified) |
| tests | 33/33 (`PYTHONPATH=. python3 -m pytest tests -q`) |

Cost: ~40 discovery-agent calls (≈90-200 s each with thinking), 7 policy-development
calls (≈115-140 s each, ~27-30 k reasoning tokens), replay sweeps free.

## Honest negative result

The developed policies did **not** beat the deployed parallel-refine policy on this
history: three versions tied at reward 0.517321 (auc 0.94695) and one trailed
(0.517269). Reason, visible in the sweep: in worlds of 9-15 cells the replay-optimal
behaviour is to widen immediately and then exhaust the grid, and prefix-only
information does not allow ranking branches early enough to improve on that. The
selected incumbent is therefore correct, not a bug — the paper's guarantee
(`selected >= current` on the fixed history) is exactly what was observed. A
positive policy improvement needs many rounds with deeper, multi-world history
(the paper uses 5-10 recursive rounds), which this verification deliberately did
not pay for.

Metric sanity on a synthetic 15-node fixture (3 branches x 5 attempts, max of 1.20
reachable at branch 2 attempt 1):

- widen-first baseline: best 1.200 in 15 probes, auc 0.6900, penalty 0.4000, reward 0.2900
- serial widen-late policy: best 1.200 in 7 probes, auc 0.3143, penalty 0.7143, reward -0.4000

Both reach the world maximum; the one that gets there early in its own trajectory and
batches wins. That is the intended trade-off shape (attainment × work × parallelism).

## Failure modes met and fixed during the session

1. **Truncated program at `max_tokens=8192`** — reasoning and answer share the
   completion budget, so the emitted `solution.py` was cut mid-file and every attempt
   failed `pack(n) not defined`. Fix: 32768 for attempts, 65536 for policy development,
   plus `finish_reason == "length"` treated as a failed attempt with a shorten-and-retry
   instruction.
2. **DSML tool-call markup** — DeepSeek emitted `｜｜DSML｜｜ invoke name="shell"` because
   it believed it had tools. Fix: no-tools system prompt + `strip_tool_markup()`.
3. **Placeholder pseudo-code** — the first attempt wrote `...`, `?`, and "TODO" lines
   into the artifact and died with `SyntaxError`. Fix: explicit no-placeholder
   requirements in the prompt + `--repairs N` re-asking with the evaluator's error.
4. **`IncompleteRead(0 bytes read)`** — the 16 KB improvement prompt + tens of
   thousands of output tokens killed non-streamed requests and the whole dreaming
   phase. Fix: SSE streaming, 3x retry with backoff, 240 s socket idle timeout,
   and "stream ended without finish_reason = cut attempt".
5. **Absolute FILE paths** — the model wrote its own absolute workspace path into the
   FILE header, so the content landed in a nested directory and `method.py` stayed the
   template; validation passed anyway. Fix: basename remapping in `extract_files` +
   rejecting a delivered file that equals the template.
6. **`SameFileError` on deploy** — selecting `m000_current` tried to copy
   `policies/current.py` onto itself. Fix: same-file guard, plus an immutable copy per
   deployed version under `policies/versions/`.

## Reproduce

```bash
cd ~/.hermes/scripts/dream_rsi && PYTHONPATH=. python3 -m pytest tests -q
python3 -m dream_rsi.cli loop --run /tmp/dr --task circle_packing --agent mock \
  --rounds 2 --workers 2 --k1 6 --versions 1 --betas 0.5 --k2 12 \
  --hard-branch 2 --hard-refine 1           # offline, no network, full loop
python3 -m dream_rsi.cli explore --run R --agent deepseek --workers 3 \
  --branches 3 --refines 4 --repairs 1      # ~15 live attempts, ~10 min
```
