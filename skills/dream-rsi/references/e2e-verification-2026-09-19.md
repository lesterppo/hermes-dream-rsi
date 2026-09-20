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


## Update — 2026-09-20: first real policy improvement (lasso_path), and the AUC axis settled

Two changes mattered:

1. **AUC axis.** Attainment is integrated against ABSOLUTE work
   (`x = cumulative_probes / probes_in_world`, flat-extended to x = 1). Normalizing
   by the episode's own probes credits a policy for *spending more* after the
   plateau, so every policy tied at "probe everything"; normalizing by the world
   size and *not* extending rewards stopping early only because the tail is
   truncated. The absolute + flat-extended axis makes early attainment (which is
   decided by how fast slots are freed from dead branches) the scored quantity,
   and the new `first_reach_probe` diagnostic reports the concrete target
   ("ceiling reached at probe N").
   Fixture check (30-cell world, 2 productive branches of 6, W=3):
   widen-all-exhaust first_reach 22, auc 0.479, reward 0.079 vs
   best-first-and-wave first_reach 9, auc 0.797, reward 0.161.

2. **`loop --branches/--refines`.** Grid size was previously fixed by the task
   defaults, so a "wide" run silently produced a 4x2 grid. The CLI now sets the
   planning-context fallbacks (and round 1), still bounded by `--hard-branch` /
   `--hard-refine` and by the worker cap (the paper's `W` is both the branch count
   and the concurrency, so `--branches 6` needs `--workers 6`).

### Live result: `lasso_path` (n=600, p=120, k=10), 2 rounds, W=3, M=1

- reference coordinate descent = speedup 1.0 (baseline floor)
- round 1 (6 attempts): best 2.42x speedup, 0 failures; dreaming: m001 vs incumbent
  0.2715 → m001 lost, incumbent kept (`improved: false`)
- round 2 (6 attempts): best **4.27x speedup** (3.48x / 2.68x / 2.17x also > 2x)
- round 2 dreaming: incumbent (baseline policy) reward **0.168685** (auc 0.669,
  penalty 0.500) vs the LLM-developed 380-line adaptive portfolio policy
  **0.280949** (auc 0.781, penalty 0.500) — a +66.5% relative replay gain, so
  `m001_20260920-141505` was deployed to `policies/current.py` and archived under
  `policies/versions/`. `improved: true` — the RSI loop closed with a genuinely
  better exploration policy and the next online round found a better solver.

Honest reading: the win came from the algorithm-engineering task where branch
values are heterogeneous, which is where pruning and portfolio composition pay.
The circle-packing worlds (all branches near the ceiling) still tie, as expected.


## Update — 2026-09-20 evening: long-horizon run, 4 rounds x 6x5 grid (circle_packing)

Command: `loop --run ~/.hermes/dream_rsi_runs/long --task circle_packing --rounds 4
--workers 6 --k1 64 --versions 2 --betas 0.3,0.6,0.9 --k2 64 --hard-branch 6
--hard-refine 4 --branches 6 --refines 4 --repairs 1`. Cost: 102 discovery calls
(~2 h wall at W=6) + 8 policy-development calls. Zero failed attempts.

| round | plan | probes | best | selected version | reward | improved |
|---|---|---|---|---|---|---|
| 1 | 6x4 (30 cells) | 30 | 0.148204 | m002_20260920-144122 | 0.781071 | **true** |
| 2 | 6x3 (24) | 24 | 0.148204 | m000_current | 0.775931 | false |
| 3 | 6x3 (24) | 24 | 0.148204 | m000_current | 0.774204 | false |
| 4 | 6x3 (24) | 24 | 0.148204 | m000_current | 0.773146 | false |

- discovery: final best **0.14820432256522875** = the published optimum for n=10
  (cell `b3#a2`), reached in round 1 and held for three more rounds.
- round-1 dreaming: incumbent 0.780934 vs m002 0.781071 (auc 0.981071, penalty
  0.2) - a marginal but real gain, so `m002_20260920-144122` (413 lines) was
  deployed; rounds 2-4 then honestly kept the incumbent (candidates 0.77314 /
  0.764684 trailed).
- **adaptive cross-cycle planning observed**: rounds 2-4 planned `6x3` with the
  reason string `live best plateaued; widen one direction and trim depth`, emitted
  by the deployed LLM policy's `plan_grid` reading the prefix-safe cycle history -
  30 probes/round dropped to 24 with the same best score.
- the ceiling is `auc ~ 0.98 / penalty 0.2` here: all policies that widen first and
  exhaust the 30-cell grid reach the optimum, so replay gains are second-order.
  Heterogeneous worlds (lasso_path) are where the policy actually matters.

### Combined verdict

- machinery: verified live end-to-end (online rollout, replay, beta sweep,
  selection, redeploy, adaptive planning, artifacts, pointer JSON, tool dispatch)
- discovery: circle_packing hits the published optimum exactly; lasso_path reaches
  a 4.27x speedup over the reference solver
- policy self-improvement: lasso_path +66.5% replay reward and 4.27x online
  speedup after redeploy; circle_packing +0.02% then a stable plateau with
  policy-driven plan adaptation


## Update — 2026-09-20 night: zero-cost agent backend + path sanitizing

`--agent gemini[:flash|pro|thinking|lite]` now drives the Gemini web CLI
(`~/.local/bin/gemini.py`, browser-cookie auth) as the discovery and policy agent,
so a whole RSI run can cost $0 in API spend. Live check:

    explore --run /tmp/dr_gem2 --task circle_packing --agent gemini \
      --workers 2 --branches 2 --refines 1 --k1 8 --repairs 1
    -> {"ok":true,"t":4,"best":0.148204,"p":4,"k":2,"fail":0}

4 attempts, 0 failures, best = the published n=10 optimum, no paid API. The first
attempt at this backend failed for an unrelated reason worth remembering: the Gemini
reply echoed the prompt, the FILE-header regex captured prose/math as the path, and
`mkdir` raised `OSError: File name too long`, which surfaced as `policy-error` with
0 nodes. Fixed by `_safe_rel_path` (charset, length, no traversal segments; absolute
paths accepted then remapped by basename) plus per-file write guards in both the
online rollout and policy development. Regression test added (34 tests).


## Update — 2026-09-20 late: `hermes_hotpath` live on a real repository function

Task: `~/hermes/scripts/travel/ota.py::_parse_trip_cards` — parse Trip.com SSR hotel
cards out of a 0.5-1 MB page. 5 real pages captured once as read-only fixtures, golden
outputs frozen from an `ast`-extracted copy of the original function, repository file
never touched. Agent: `gemini:pro` (zero API cost), 2 rounds x 3 branches x 3 attempts,
`--repairs 2`.

- 18 attempts, **all valid**: exact JSON equality with the reference on every page
- speedups 1.05x - **3.36x** (best cell `b1#a2`: reference 5.54 ms per 5-page sweep →
  1.65 ms), i.e. the loop found a real 3x speedup in an existing tool of mine
- round-2 dreaming improved the exploration policy (0.196080 → **0.203183**, auc/penalty
  on the same two worlds) and redeployed it, `improved: true`

Four integration bugs surfaced and were fixed along the way:

1. **Helper signatures must match the paper's prompt.** Policies written to the
   Appendix B.2 API call `branch_failed_hard(obs)` with a single observation; our
   helpers required `(observations, branch)`, so every developed policy died with
   `TypeError` and the cycle silently kept the incumbent. Helpers now accept either
   form (single `Observation`, or a sequence plus branch id).
2. **Markdown autolinking corrupts generated code.** The Gemini CLI returns markdown,
   which rewrote `"https://…/hotelId=700948"` into
   `"[https://…/hotelId=](https://…/hotelId=)700948"` inside a candidate's string
   literal — a silent correctness failure. `delink()` now restores `[url](url)` →
   `url` and unescapes `\_`/`\*` in every extracted file, and the prompt forbids
   markdown link syntax inside code.
3. **The gate needs to say what is wrong.** Exact-equality rejection now reports the
   first field-level discrepancies (`trip_bangkok[0].p: got None want 437.0`) instead
   of just naming the page, which is what let the repair attempts converge.
4. **Spec strings travel through shells.** Commas inside a text kwarg split the spec;
   escaped commas were unreliable through two shell layers, so use a comma-free note
   (the parser still honours `\,`).

Fixture/golden caching lives under `~/.hermes/dream_rsi_tasks/hotpath_<label>/`, so a
change to the upstream function does not silently move the score.
