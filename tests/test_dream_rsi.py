"""Test suite for the Dream-RSI integration (paper arXiv 2609.14858)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dream_rsi.agents import MockBackend, extract_files, make_agent  # noqa: E402
from dream_rsi.dream import (VersionReport, develop_policy, evaluate_version,  # noqa: E402
                             find_policy_class, load_policy_module, make_policy,
                             select_best, validate_policy)
from dream_rsi.grid import GridPlan, GridPlanningContext  # noqa: E402
from dream_rsi.loop import run_loop  # noqa: E402
from dream_rsi.online import LiveQuestion, LiveSession, run_online_round  # noqa: E402
from dream_rsi.policies.baseline import ParallelRefinePolicy  # noqa: E402
from dream_rsi.policies.portfolio import PortfolioPolicy  # noqa: E402
from dream_rsi.policy_api import LLMDesignedMethod  # noqa: E402
from dream_rsi.simulator import (IllegalBatch, ObjectiveConfig, ReplayQuestion,  # noqa: E402
                                 pareto_auc, pareto_reward, parallel_penalty,
                                 replay_score, run_replay_episode)
from dream_rsi.tasks import build_task  # noqa: E402
from dream_rsi.traces import RunLayout  # noqa: E402
from dream_rsi.tree import DiscoveryTree  # noqa: E402
from dream_rsi import cli  # noqa: E402

PACKAGE_ROOT = ROOT / "dream_rsi"


def make_trace(scores=None, baseline=0.0, attempts=2, branches=3) -> DiscoveryTree:
    """Irregular grid: branch b has attempts 0..(attempts-1) with given scores."""
    scores = scores or [[1.0, 2.0], [0.5, 5.0], [3.0, 1.0]][:branches]
    tree = DiscoveryTree(task="unit", baseline_score=baseline)
    for b, branch_scores in enumerate(scores):
        parent = None
        for a, s in enumerate(branch_scores):
            parent = tree.add_node(b, a, parent, s).id
    return tree


# --------------------------------------------------------------------------- #
# tree                                                                        #
# --------------------------------------------------------------------------- #
def test_tree_add_leaves_best_roundtrip(tmp_path: Path) -> None:
    tree = make_trace()
    assert tree.size() == 6
    assert tree.max_score() == 5.0
    assert tree.best().id == "b1#a1"
    assert {n.id for n in tree.leaves()} == {"b0#a1", "b1#a1", "b2#a1"}
    p = tree.save(tmp_path / "t.json")
    again = DiscoveryTree.load(p)
    assert again.to_dict() == tree.to_dict()
    assert again.summary()["nodes"] == 6


def test_tree_rejects_duplicate_node() -> None:
    tree = make_trace()
    with pytest.raises(ValueError):
        tree.add_node(0, 0, None, 1.0)


# --------------------------------------------------------------------------- #
# replay simulator                                                            #
# --------------------------------------------------------------------------- #
def test_replay_prefix_only_hides_unrevealed_scores() -> None:
    tree = make_trace()
    q = ReplayQuestion(tree, max_parallelism=2)
    assert q.observed() == {}
    assert q.legal_roots() == ["b0#a0", "b1#a0", "b2#a0"]
    q.probe_batch(["b0#a0"])
    seen = q.observed()
    assert set(seen) == {"b0#a0"}
    assert seen["b0#a0"].score == 1.0
    # b1's recorded score (5.0) is still unreachable through the API
    assert all(o.branch != 1 for o in seen.values())


def test_replay_frontier_and_root_opening_semantics() -> None:
    tree = make_trace()
    q = ReplayQuestion(tree, max_parallelism=4)
    q.probe_batch(["b0#a0", "b1#a0"])
    assert q.opened_branches() == [0, 1]
    legal = q.legal_actions()
    assert "b0#a1" in legal and "b1#a1" in legal
    assert "b2#a0" in legal
    assert "b0#a2" not in legal  # depth beyond the recorded grid
    q.probe_batch(["b0#a1"])
    assert q.observed()["b0#a1"].score == 2.0
    assert q.observed()["b0#a1"].delta_vs_parent == 1.0


def test_replay_rejects_illegal_batches() -> None:
    tree = make_trace()
    q = ReplayQuestion(tree, max_parallelism=2)
    with pytest.raises(IllegalBatch):
        q.probe_batch(["b0#a0", "b1#a0", "b2#a0"])       # exceeds W
    with pytest.raises(IllegalBatch):
        q.probe_batch(["b0#a0", "b0#a0"])                # duplicates
    with pytest.raises(IllegalBatch):
        q.probe_batch(["b0#a9"])                         # unknown cell
    q.probe_batch(["b0#a0"])
    with pytest.raises(IllegalBatch):
        q.probe_batch(["b0#a1", "b0#a2"])                # child w/o parent revealed
    with pytest.raises(IllegalBatch):
        q.probe_batch(["nonsense"])                      # malformed id


def test_replay_parent_child_same_batch_rejected() -> None:
    tree = make_trace()
    q = ReplayQuestion(tree, max_parallelism=4)
    q.probe_batch(["b0#a0"])
    q.probe_batch(["b0#a1"])          # legal once the parent is revealed
    assert q.observed()["b0#a1"].attempt == 1
    q2 = ReplayQuestion(tree, max_parallelism=4)
    with pytest.raises(IllegalBatch):
        q2.probe_batch(["b0#a0", "b0#a1"])


def test_replay_exhausted_cell_costs_no_probe() -> None:
    tree = DiscoveryTree(task="unit", baseline_score=0.0)
    tree.add_node(0, 0, None, 1.0)
    q = ReplayQuestion(tree, max_parallelism=2)
    obs = q.probe_batch(["b0#a0"])
    assert q.probes_spent == 1
    assert len(obs) == 1
    # irregular grid: the trace has no b0#a1 -> probing it reveals nothing, costs
    # no probe, and is retired afterwards
    assert "b0#a1" in q.legal_actions()
    obs2 = q.probe_batch(["b0#a1"])
    assert obs2 == []
    assert q.probes_spent == 1
    assert q.decision_rounds == 2
    assert q.is_exhausted("b0#a1") and "b0#a1" not in q.legal_actions()


def test_replay_is_deterministic() -> None:
    tree = make_trace()
    for policy_cls in (ParallelRefinePolicy, PortfolioPolicy):
        r1 = run_replay_episode(policy_cls({"beta": 0.5, "max_parallelism": 3}),
                                tree, max_parallelism=3)
        r2 = run_replay_episode(policy_cls({"beta": 0.5, "max_parallelism": 3}),
                                tree, max_parallelism=3)
        assert r1.to_dict() == r2.to_dict()


def test_replay_reaches_trace_ceiling_with_parallel_refine() -> None:
    tree = make_trace()
    res = run_replay_episode(ParallelRefinePolicy({"beta": 0.6, "max_parallelism": 3}),
                             tree, max_parallelism=3)
    assert res.best_score == 5.0
    assert res.probes == tree.size()
    assert res.error is None
    assert res.terminated_by == "exhausted"


def test_replay_objective_matches_equation_1() -> None:
    tree = make_trace()
    res = run_replay_episode(ParallelRefinePolicy({"beta": 0.6, "max_parallelism": 3}),
                             tree, max_parallelism=3)
    cfg = ObjectiveConfig(beta1=0.01, beta2=0.05)
    n = float(res.probes)
    k = max(1, res.decision_rounds)
    expected = res.best_score - cfg.beta1 * n + cfg.beta2 * (n / k)
    assert replay_score(res, cfg) == pytest.approx(expected)


def test_pareto_metrics_hand_computed() -> None:
    tree = make_trace()

    class Serial(LLMDesignedMethod):
        """One cell per decision round: no batching, penalty near 1/W-scaled max."""

        NAME = "Serial"

        def select_batch(self, question):  # noqa: ANN001
            return sorted(set(question.legal_actions()))[:1]

        def plan_grid(self, context):  # noqa: ANN001
            return GridPlan(1, 1, "serial")

    serial = run_replay_episode(Serial({"beta": 0.5, "max_parallelism": 3}),
                                tree, max_parallelism=3)
    parallel = run_replay_episode(
        ParallelRefinePolicy({"beta": 0.5, "max_parallelism": 3}), tree,
        max_parallelism=3)
    # penalty definition: effective sequential rounds / probes
    assert parallel_penalty(serial) == pytest.approx(
        serial.effective_sequential_rounds / serial.probes)
    assert parallel_penalty(serial) > parallel_penalty(parallel)
    auc = pareto_auc(parallel, tree)
    assert 0.0 < auc <= 1.0
    assert pareto_reward(auc, 0.25, ObjectiveConfig(lam=1.0)) == pytest.approx(auc - 0.25)


def test_pareto_auc_zero_when_never_beats_baseline() -> None:
    tree = make_trace(baseline=100.0)
    res = run_replay_episode(ParallelRefinePolicy({"beta": 0.6, "max_parallelism": 3}),
                             tree, max_parallelism=3)
    assert pareto_auc(res, tree) == 0.0


# --------------------------------------------------------------------------- #
# policies                                                                    #
# --------------------------------------------------------------------------- #
def test_policies_emit_legal_batches_and_use_parallelism() -> None:
    tree = make_trace(attempts=3, branches=3)
    for cls in (ParallelRefinePolicy, PortfolioPolicy):
        q = ReplayQuestion(tree, max_parallelism=3)
        policy = cls({"beta": 0.8, "max_parallelism": 3})
        first = policy.select_batch(q)
        assert len(first) == 3, f"{cls.__name__} left workers idle: {first}"
        q.probe_batch(first)
        second = policy.select_batch(q)
        assert len(second) <= 3
        q.probe_batch(second)  # must not raise
        assert set(policy.select_batch(q)).issubset(set(q.legal_actions()))


def test_beta_schedule_is_monotonic_in_beta() -> None:
    low = PortfolioPolicy({"beta": 0.1})._schedule(0.1)   # noqa: SLF001
    high = PortfolioPolicy({"beta": 0.9})._schedule(0.9)  # noqa: SLF001
    assert high["stagnation_patience"] >= low["stagnation_patience"]
    assert high["prune_ratio"] >= low["prune_ratio"]
    assert high["reserve_threshold"] <= low["reserve_threshold"]


def test_grid_plan_clamps_and_support_check() -> None:
    ctx = GridPlanningContext(worker_cap=3, hard_max_branch_count=5,
                              hard_max_refine_count=2, trace_branch_count=4,
                              trace_refine_count=2)
    plan = GridPlan(99, 99, "too big").clamp(ctx)
    assert (plan.branch_count, plan.refine_count) == (3, 2)
    assert ctx.in_replay_support(plan)
    assert not ctx.in_replay_support(GridPlan(4, 3))


def test_plan_grid_is_prefix_safe_and_always_returns_a_plan() -> None:
    policy = PortfolioPolicy({"beta": 0.6})
    empty = policy.plan_grid(GridPlanningContext(worker_cap=4,
                                                 fallback_branch_count=4,
                                                 fallback_refine_count=3))
    assert isinstance(empty, GridPlan) and empty.reason
    hist = [{"final_best": 1.0, "baseline": 0.5},
            {"final_best": 1.02, "baseline": 0.5}]
    plateaud = policy.plan_grid(GridPlanningContext(history=hist, worker_cap=4,
                                                    fallback_branch_count=4,
                                                    fallback_refine_count=3,
                                                    hard_max_branch_count=8))
    assert plateaud.branch_count >= empty.branch_count
    assert plateaud.reason


# --------------------------------------------------------------------------- #
# agents                                                                      #
# --------------------------------------------------------------------------- #
def test_extract_files_protocol_and_fence_fallback() -> None:
    text = ("<<<FILE: a.py>>>\nprint(1)\n<<<END>>>\n"
            "<<<FILE: sub/b.md>>>\nhello\n<<<END>>>\n")
    files = extract_files(text)
    assert set(files) == {"a.py", "sub/b.md"}
    assert files["a.py"].strip() == "print(1)"
    assert extract_files("```python\nx = 1\n```", "s.py")["s.py"].strip() == "x = 1"
    assert extract_files("nothing useful", "s.py") == {}


def test_mock_agent_emits_policy_for_improvement_prompt() -> None:
    agent = MockBackend(task="circle_packing")
    res = agent.complete("You are improving one **prefix-only exploration policy**.")
    assert res.ok
    assert "method.py" in extract_files(res.text)


# --------------------------------------------------------------------------- #
# online rollout                                                              #
# --------------------------------------------------------------------------- #
def _tiny_session(tmp_path: Path, task, agent, **kw) -> LiveSession:
    round_dir = tmp_path / "rounds" / "r0001"
    (round_dir / "trace").mkdir(parents=True, exist_ok=True)
    return LiveSession(round_dir=round_dir, task=task, agent=agent,
                       plan={"branch_count": kw.pop("branches", 2),
                             "refine_count": kw.pop("refines", 1)},
                       workers=kw.pop("workers", 2),
                       baseline_score=kw.pop("baseline", 0.0), **kw)


def test_online_round_builds_tree_and_artifacts(tmp_path: Path) -> None:
    task = build_task("circle_packing")
    agent = MockBackend(program_name=task.eval_program, task=task.name)
    session = _tiny_session(tmp_path, task, agent, workers=2)
    tree, res = run_online_round(session, ParallelRefinePolicy({"beta": 0.6,
                                                                "max_parallelism": 2}),
                                 round_limit=8)
    assert tree.size() == 4        # 2 branches x (root + 1 refinement)
    assert res.error is None
    assert (session.round_dir / "trace" / "tree.json").exists()
    node_dir = session.attempt_dir(0, 0)
    assert (node_dir / "proposal.md").exists()
    assert (node_dir / task.eval_program).exists()
    assert (node_dir / "eval" / "score.json").exists()
    assert json.loads((node_dir / "eval" / "score.json").read_text())["fail_class"] == "ok"


def test_online_round_marks_agent_failure(tmp_path: Path) -> None:
    class Broken(MockBackend):
        def complete(self, prompt, cwd=None, system=None):  # noqa: ANN001
            from dream_rsi.agents import AgentResult
            return AgentResult(False, error="boom")

    task = build_task("circle_packing")
    session = _tiny_session(tmp_path, task, Broken(program_name=task.eval_program),
                            workers=2)
    tree, res = run_online_round(session, ParallelRefinePolicy({"beta": 0.6,
                                                                "max_parallelism": 2}),
                                 round_limit=4)
    assert tree.size() >= 1
    node = tree.nodes["b0#a0"]
    assert node.fail_class == "agent" and not node.valid


def test_live_question_rejects_illegal_batches(tmp_path: Path) -> None:
    task = build_task("circle_packing")
    session = _tiny_session(tmp_path, task,
                            MockBackend(program_name=task.eval_program), workers=2)
    q = LiveQuestion(session)
    with pytest.raises(IllegalBatch):
        q.probe_batch(["b0#a0", "b1#a0", "b2#a0"])
    with pytest.raises(IllegalBatch):
        q.probe_batch(["b5#a0"])


# --------------------------------------------------------------------------- #
# tasks                                                                       #
# --------------------------------------------------------------------------- #
def test_circle_packing_baseline_and_harness(tmp_path: Path) -> None:
    task = build_task("circle_packing")
    baseline_dir = tmp_path / "baseline"
    outcome = task.prepare(tmp_path, baseline_dir)
    assert outcome.score == pytest.approx(0.125)
    assert outcome.valid and outcome.n_valid == 45

    bad = tmp_path / "attempt"
    bad.mkdir()
    (bad / "solution.py").write_text("def pack(n):\n    return [(0.5, 0.5)]\n")
    out = task.evaluate(bad)
    assert not out.valid and out.fail_class == "invalid"
    assert out.score < 0

    broken = tmp_path / "attempt2"
    broken.mkdir()
    (broken / "solution.py").write_text("def pack(n:\n")
    out2 = task.evaluate(broken)
    assert not out2.valid and out2.fail_class == "compile"


def test_lasso_task_baseline_is_valid_and_scores_speedup(tmp_path: Path) -> None:
    task = build_task("lasso_path:n=240,p=60,k=6")
    outcome = task.prepare(tmp_path, tmp_path / "baseline")
    assert outcome.valid and outcome.score > 0.5   # same code as reference
    assert outcome.n_valid == outcome.n_total == 6


# --------------------------------------------------------------------------- #
# dreaming                                                                    #
# --------------------------------------------------------------------------- #
def test_policy_loading_and_class_discovery() -> None:
    path = PACKAGE_ROOT / "policies" / "portfolio.py"
    module = load_policy_module(path)
    assert getattr(module, "NAME") == "OptimalPolicy"
    assert find_policy_class(module).__name__ == PortfolioPolicy.__name__
    policy = make_policy(path, 0.4, 3)
    assert policy.beta == 0.4 and policy.max_parallelism == 3


def test_validate_policy_rejects_illegal_batch(tmp_path: Path) -> None:
    bad = tmp_path / "bad_policy.py"
    bad.write_text(
        "from dream_rsi.policy_api import LLMDesignedMethod\n"
        "from dream_rsi.grid import GridPlan\n"
        "NAME = 'Bad'\n"
        "class Bad(LLMDesignedMethod):\n"
        "    def plan_grid(self, context):\n"
        "        return GridPlan(2, 1, 'bad')\n"
        "    def select_batch(self, question):\n"
        "        return question.legal_actions()  # ignores max_parallelism\n")
    ok, err = validate_policy(bad, [make_trace()], [0.5], 2)
    assert not ok and "ILLEGAL_BATCH" in (err or "")


def test_validate_policy_rejects_missing_plan_grid(tmp_path: Path) -> None:
    bad = tmp_path / "no_plan.py"
    bad.write_text(
        "from dream_rsi.policy_api import LLMDesignedMethod\n"
        "NAME = 'NoPlan'\n"
        "class NoPlan(LLMDesignedMethod):\n"
        "    def select_batch(self, question):\n"
        "        return []\n")
    ok, err = validate_policy(bad, [], [0.5], 2)
    assert not ok and "plan_grid" in (err or "")


def test_evaluate_version_sweeps_beta_over_every_world() -> None:
    traces = [make_trace(), make_trace(scores=[[2.0, 4.0], [1.0, 3.0]])]
    collector: list = []
    report = evaluate_version(PACKAGE_ROOT / "policies" / "portfolio.py", traces,
                              [0.2, 0.8], workers=2, collector=collector)
    assert report.ok
    assert len(report.episodes) == 4          # 2 betas x 2 worlds
    assert set(report.per_beta) == {"0.200", "0.800"}
    assert report.best_beta in (0.2, 0.8)
    assert len(collector) == 4
    assert {"version", "beta", "trace", "timeline"} <= set(collector[0])


def test_selection_is_no_worse_than_current_policy() -> None:
    traces = [make_trace()]
    current = evaluate_version(PACKAGE_ROOT / "policies" / "baseline.py", traces,
                               [0.5], workers=3, name="m000_current")
    candidate = evaluate_version(PACKAGE_ROOT / "policies" / "portfolio.py", traces,
                                 [0.5], workers=3, name="m001")
    best = select_best([current, candidate])
    assert best.reward >= current.reward


def test_develop_policy_retries_then_reports_failure(tmp_path: Path) -> None:
    class Garbage:
        def complete(self, prompt, cwd=None, system=None):  # noqa: ANN001
            from dream_rsi.agents import AgentResult
            return AgentResult(True, text="<<<FILE: method.py>>>\nnot python(\n<<<END>>>")

    round_dir = tmp_path / "rounds" / "r0001"
    round_dir.mkdir(parents=True)
    dev = develop_policy(Garbage(), round_dir, "m001", tmp_path, tmp_path,
                         "feedback", [make_trace()], [0.5], 2, attempts=2)
    assert not dev.ok and dev.tries == 2
    assert (round_dir / "policy_input" / "m001.prompt.txt").exists()


# --------------------------------------------------------------------------- #
# full loop                                                                   #
# --------------------------------------------------------------------------- #
def test_offline_loop_produces_full_artifact_set(tmp_path: Path) -> None:
    out = run_loop(tmp_path / "run", "circle_packing:n=10", "mock", rounds=2,
                   workers=2, k1=8, versions=1, betas=[0.5], k2=16,
                   hard_branch=2, hard_refine=1, verbose=False)
    layout = RunLayout(tmp_path / "run")
    state = layout.load_state()
    assert state["baseline_score"] == pytest.approx(0.125)
    assert state["round"] == 2
    assert len(layout.trace_paths()) == 2
    assert (layout.trace_pool / "iter0001" / "live_cycle_manifest.json").exists()
    assert (layout.trace_pool / "iter0002" / "live_cycle_manifest.json").exists()
    for round_dir in layout.round_dirs():
        assert (round_dir / "proposal_results" / "beta_sweep.json").exists()
        assert (round_dir / "proposal_results" / "replay_summary.json").exists()
        assert (round_dir / "proposal_results"
                / "policy_execution_traces.jsonl").exists()
        assert (round_dir / "live_cycle_manifest.json").exists()
        assert (round_dir / "method.py").exists()
    sweep = json.loads((layout.round_dirs()[-1] / "proposal_results"
                        / "beta_sweep.json").read_text())
    assert sweep["selected"]["version"]
    assert len(sweep["versions"]) >= 2         # current + developed
    hist = json.loads((tmp_path / "run" / "loop_report.json").read_text())
    assert [h["round"] for h in hist] == [1, 2]
    assert out["state"]["current_beta"] is not None


def test_cli_pointer_json_and_status(tmp_path: Path, capsys) -> None:
    run = tmp_path / "cli_run"
    assert cli.main(["init", "--run", str(run), "--task", "circle_packing"]) == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["ok"] and payload["baseline_score"] == pytest.approx(0.125)
    assert cli.main(["status", "--run", str(run)]) == 0
    status = json.loads(capsys.readouterr().out.strip())
    assert status["task"] == "circle_packing" and status["round"] == 0
    assert cli.main(["explore", "--run", str(run), "--agent", "mock",
                     "--workers", "2", "--branches", "2", "--refines", "1",
                     "--k1", "6"]) == 0
    explore = json.loads(capsys.readouterr().out.strip())
    assert explore["ok"] and explore["t"] >= 2
    assert cli.main(["replay", "--run", str(run), "--betas", "0.5"]) == 0
    replay = json.loads(capsys.readouterr().out.strip())
    assert replay["worlds"] == 1
    assert cli.main(["show", "--run", str(run), "--what", "tree"]) == 0
    assert json.loads(capsys.readouterr().out.strip())["ok"]
    assert cli.main(["tasks"]) == 0
    assert "circle_packing" in json.loads(capsys.readouterr().out.strip())["tasks"]


def test_cli_reports_errors_as_pointer_json(tmp_path: Path, capsys) -> None:
    code = cli.main(["explore", "--run", str(tmp_path / "nope"), "--agent", "mock"])
    payload = json.loads(capsys.readouterr().out.strip())
    assert code == 1 and payload["ok"] is False and payload["err"]


# --------------------------------------------------------------------------- #
# shim                                                                        #
# --------------------------------------------------------------------------- #
def test_see_policy_api_shim_exports_surface() -> None:
    from see.policy.api import (GridPlan, GridPlanningContext,  # noqa: F401
                                LLMDesignedMethod, Observation, SimResult,
                                _budget_done, _record_curve, branch_failed_hard,
                                branch_promising, finalize_result)
    assert LLMDesignedMethod.__name__ == "LLMDesignedMethod"
    assert callable(_record_curve) and callable(finalize_result)
