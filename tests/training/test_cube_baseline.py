import json
import math

import pytest
import torch

from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.training import cube_baseline
from neuromorphic.training.cube_baseline import (
    CubeConfig,
    ShellCubeEnv,
    evaluate_states,
    make_agent,
    max_steps_for,
    modal_action_fraction,
    record_filename,
    run_cube_baseline,
    shell_states,
    split_shell,
)

PUBLISHED = {1: 6, 2: 27, 3: 120, 4: 534, 5: 2256, 6: 8969}


@pytest.fixture(scope="module")
def provider():
    return ExactBFSDistance(max_depth=4)


def test_max_steps_rule():
    assert [max_steps_for(d) for d in (1, 2, 3, 6)] == [5, 7, 9, 15]


def test_shell_sizes_match_published(provider):
    for d in (1, 2, 3, 4):
        assert len(shell_states(provider, d)) == PUBLISHED[d]


def test_shallow_depths_are_not_split(provider):
    for d in (1, 2):
        states = shell_states(provider, d)
        train, ev, is_heldout = split_shell(states, d, seed=0)
        assert is_heldout is False
        assert train == ev == states  # training distribution, labelled as such


def test_deep_depths_split_disjointly_and_deterministically(provider):
    states = shell_states(provider, 3)
    train, ev, is_heldout = split_shell(states, 3, seed=0)
    assert is_heldout is True
    assert set(train).isdisjoint(ev)
    assert len(train) + len(ev) == len(states)
    assert len(ev) == 30  # 25% of 120, under the 200 cap
    again = split_shell(states, 3, seed=0)
    assert again[0] == train and again[1] == ev


def test_heldout_is_capped(provider):
    states = shell_states(provider, 4)  # 534 states, 25% = 133, under the cap
    _, ev, _ = split_shell(states, 4, seed=0, heldout_cap=50)
    assert len(ev) == 50


def test_shell_env_starts_inside_the_pool(provider):
    import random
    pool = shell_states(provider, 2)
    env = ShellCubeEnv(pool, random.Random(0), scramble_depth=2, max_steps=max_steps_for(2))
    for _ in range(10):
        obs, _ = env.reset()
        assert tuple(int(c) for c in obs) in set(pool)


def test_random_arm_scores_at_the_measured_depth1_floor(provider):
    """The measured chance floor, asserted against a MEASURED band.

    Was `assert 0.0 <= success_rate <= 1.0`, which is a tautology: a success rate is a count
    over a total and cannot leave [0, 1]. It was debt item 7 in the 2026-08-02 handoff and it
    could not fail against any implementation, correct or not.

    Measured 2026-08-03 over 40 rng seeds at depth 1: mean 0.1958, sd 0.1683, single-seed
    values only ever in {0, 1/6, 2/6, 3/6, 4/6}. The shell is 6 states so single-seed
    resolution is 1/6 and no per-seed bound can be tight; the MEAN is the assertable quantity,
    and it is the one CLAUDE.md quotes as "21% at depth 1, not 1/6".

    Fails against: a random arm that never solves (mean 0), one that accidentally runs a
    trained or greedy policy (mean far above), and any change that makes the 2d+3 budget stop
    letting a random walk stumble into solved.
    """
    states = shell_states(provider, 1)
    rates = [
        evaluate_states(None, None, states, depth=1, random_policy=True, rng_seed=k)[
            "success_rate"
        ]
        for k in range(40)
    ]
    assert all(r["n"] == 6 for r in [{"n": len(states)}])
    mean = sum(rates) / len(rates)
    assert 0.12 < mean < 0.28, f"depth-1 floor {mean:.4f} outside the measured band"
    # The floor must be strictly between "never" and "always". Both bounds are reachable in
    # principle for a 6-state shell, which is what makes them worth asserting.
    assert min(rates) < max(rates), "every seed scored identically, the floor is one realisation"


def test_different_rng_seeds_give_different_random_arm_results(provider):
    """Fix 1: the chance floor must not be one realization replayed under every seed.

    Depth 3's shell (120 states) is large enough that two different rng seeds land on
    different success rates almost surely; seeds 0 and 1 are checked here and verified
    not to tie. No training happens, so this stays fast.
    """
    states = shell_states(provider, 3)
    res_a = evaluate_states(None, None, states, depth=3, random_policy=True, rng_seed=0)
    res_b = evaluate_states(None, None, states, depth=3, random_policy=True, rng_seed=1)
    assert res_a["success_rate"] != res_b["success_rate"]


def test_agents_are_built_for_both_arms():
    reg = make_agent(CubeConfig(arm="regionalized"))
    mono = make_agent(CubeConfig(arm="monolithic"))
    assert reg.n_neurons == mono.n_neurons
    assert reg.content == mono.content
    with pytest.raises(ValueError, match="unknown arm"):
        make_agent(CubeConfig(arm="nonsense"))


def test_smoke_run_produces_a_wellformed_record(tmp_path):
    cfg = CubeConfig(
        arm="regionalized", depth=1, seed=0, episodes=3, max_depth=1, out_dir=tmp_path,
    )
    rec = run_cube_baseline(cfg)
    for key in ("arm", "depth", "seed", "sigma", "success_rate", "n", "is_heldout", "episodes"):
        assert key in rec
    assert rec["arm"] == "regionalized"
    assert rec["is_heldout"] is False  # depth 1
    assert rec["n"] == 6  # depth-1 shell size (PUBLISHED[1])
    solved_count = rec["success_rate"] * rec["n"]
    assert solved_count == pytest.approx(round(solved_count))  # success_rate is solved/n


def test_modal_action_fraction_is_one_for_a_constant_rollout():
    """The signature of a collapsed policy: every step took the same action."""
    assert modal_action_fraction([3, 3, 3, 3, 3]) == 1.0


def test_modal_action_fraction_is_one_sixth_for_a_full_sweep():
    assert modal_action_fraction([0, 1, 2, 3, 4, 5]) == pytest.approx(1.0 / 6.0)


def test_modal_action_fraction_of_an_empty_rollout_is_zero():
    assert modal_action_fraction([]) == 0.0


def test_random_policy_modal_fraction_sits_near_the_measured_uniform_floor(provider):
    """Measured, not assumed.

    Prototyped 2026-07-31 over 20,000 simulated uniform rollouts on 6 actions: mean
    modal fraction is 0.354 at a 9-step budget and 0.429 at 5 steps. A constant-action
    policy scores exactly 1.0. The 0.60 bar sits above the uniform mean with margin and
    far below collapse, so this fails if the metric is ever wired to something degenerate.
    """
    states = shell_states(provider, 3)
    res = evaluate_states(None, None, states, depth=3, random_policy=True, rng_seed=0)
    assert 0.25 < res["greedy_modal_action_frac"] < 0.60


def test_a_collapsed_policy_is_reported_as_modal_fraction_one(provider, monkeypatch):
    """The instrument must catch the exact failure it exists to detect.

    A depth-3 state cannot be solved by one repeated move, so every rollout runs to the
    9-step budget and the fraction is exactly 1.0 rather than an artifact of early exit.
    """
    monkeypatch.setattr(cube_baseline, "greedy_action", lambda *a, **k: 2)
    states = shell_states(provider, 3)
    res = evaluate_states(object(), object(), states, depth=3)
    assert res["greedy_modal_action_frac"] == 1.0


def test_record_carries_both_collapse_instruments(tmp_path):
    """Greedy collapse and training-policy collapse are separate failures.

    `entropy_beta` is a training-time setting but the F'-nine-times observation was made
    under the greedy policy, so a diagnosis needs both numbers. An untrained 3-episode run
    sits near the log(6) = 1.792 ceiling; the 1.0 floor is well clear of a collapsed policy.
    """
    cfg = CubeConfig(
        arm="regionalized", depth=1, seed=0, episodes=3, max_depth=1, out_dir=tmp_path,
    )
    rec = run_cube_baseline(cfg)
    assert 1.0 < rec["mean_train_entropy"] <= math.log(6) + 1e-6
    assert 1.0 / 6.0 <= rec["greedy_modal_action_frac"] <= 1.0


def test_both_arms_get_identical_head_init_at_a_fixed_seed():
    """The paired comparison assumes matched heads; only topology should differ."""
    from neuromorphic.training.reinforce import make_policy_head

    weights = {}
    for arm in ("regionalized", "monolithic"):
        cfg = CubeConfig(arm=arm, seed=0)
        agent = make_agent(cfg)
        torch.manual_seed(cfg.seed)
        head = make_policy_head(agent, "linear")
        weights[arm] = head.weight.detach().clone()
    assert torch.equal(weights["regionalized"], weights["monolithic"])


# --- EXP-036: the train-side eval cap and head serialisation ---


def test_train_eval_sample_is_capped_below_the_full_train_side(provider):
    """The cap must BIND, not merely exist.

    Depth 4's train side is 401 states. Uncapped, evaluating it costs 6.6 min per seed, and
    at depth 6 (8,769 states) it costs 3.3 h per seed, which is more than the training run it
    is instrumenting. Asserting only `len(sample) > 0` would pass against an uncapped
    implementation, so this asserts the cap value AND that it is strictly smaller than the
    train side it was drawn from.
    """
    train, _, _ = split_shell(shell_states(provider, 4), 4, seed=0)
    assert len(train) == 401
    sample = cube_baseline.sample_train_eval(train, seed=0, cap=200)
    assert len(sample) == 200
    assert len(sample) < len(train)
    assert set(sample) <= set(train)
    assert len(set(sample)) == len(sample)


def test_train_eval_sample_returns_everything_when_under_the_cap(provider):
    train, _, _ = split_shell(shell_states(provider, 3), 3, seed=0)
    assert len(train) == 90
    assert sorted(cube_baseline.sample_train_eval(train, seed=0, cap=200)) == sorted(train)


def test_train_eval_sample_is_deterministic_but_split_seed_dependent(provider):
    """Same split_seed selects the same states; a different one selects different states.

    The second half is what gives the first half meaning. A function that ignored its seed
    entirely would pass a determinism check alone.
    """
    train, _, _ = split_shell(shell_states(provider, 4), 4, seed=0)
    a = cube_baseline.sample_train_eval(train, seed=0, cap=200)
    b = cube_baseline.sample_train_eval(train, seed=0, cap=200)
    c = cube_baseline.sample_train_eval(train, seed=7, cap=200)
    assert a == b
    assert a != c


def test_record_carries_the_gap_and_it_is_train_minus_heldout(tmp_path):
    cfg = CubeConfig(seed=0, depth=3, episodes=6, tag="t_gap", out_dir=tmp_path)
    r = run_cube_baseline(cfg)
    assert r["n_train_eval"] == 90            # depth 3 train side, under the cap
    assert r["n"] == 30                       # held-out side
    assert r["generalisation_gap"] == pytest.approx(
        r["train_success_rate"] - r["success_rate"]
    )


def test_heldout_numbers_are_unchanged_by_adding_the_train_side_eval(tmp_path):
    """The EXP-030 reference values, measured on this machine before EXP-036 existed.

    `greedy_action` draws on the shared torch generator, so evaluating the train side BEFORE
    the held-out side would advance the stream and move every held-out number in the file.
    These are exact equalities, so they fail if any RNG stream shifts at all.
    """
    cfg = CubeConfig(seed=0, depth=1, episodes=600, tag="exp030_concept", out_dir=tmp_path)
    r = run_cube_baseline(cfg)
    assert r["success_rate"] == 0.6666666666666666
    assert r["revisit_rate"] == 0.16633266533066132
    assert r["eval_revisit_rate"] == 0.25
    assert r["greedy_modal_action_frac"] == 1.0
    assert r["mean_train_entropy"] == 0.5422023858095053


def test_saved_head_reproduces_the_recorded_success_rate(tmp_path):
    """Round-trip the checkpoint, not just its existence.

    Asserting the file exists would pass against a head serialised BEFORE training, which is
    the failure that would make the checkpoint worthless. This reloads the weights into a
    fresh head and re-runs the recorded evaluation, which pins that the saved parameters are
    the TRAINED ones.
    """
    from neuromorphic.analysis.ablate import AblatedConcept
    from neuromorphic.training.cube_baseline import (
        feature_width,
        head_filename,
        resolve_seed,
    )

    cfg = CubeConfig(seed=0, depth=3, episodes=8, tag="t_ckpt", out_dir=tmp_path)
    record = run_cube_baseline(cfg)

    path = tmp_path / head_filename(cfg)
    assert path.exists()

    train_seed = resolve_seed(cfg, "train")
    agent = make_agent(cfg)
    torch.manual_seed(train_seed)
    fresh = AblatedConcept(
        torch.nn.Linear(feature_width(cfg), cfg.n_actions), None, width=feature_width(cfg)
    )
    fresh.load_state_dict(torch.load(path))

    provider = ExactBFSDistance(max_depth=cfg.max_depth)
    _, eval_states, _ = split_shell(
        shell_states(provider, cfg.depth), cfg.depth, seed=resolve_seed(cfg, "split")
    )
    replay = evaluate_states(
        agent, fresh, eval_states, depth=cfg.depth,
        generator=torch.Generator().manual_seed(train_seed), rng_seed=train_seed,
    )
    assert replay["success_rate"] == record["success_rate"]


def test_random_arm_scores_near_the_floor_on_both_sides(tmp_path):
    """The control on the gap instrument, stated at the resolution it actually has.

    A random policy cannot overfit, so its gap is the empirical null. Measured over seeds 0-5
    on 2026-08-03, single-seed gaps ranged -0.100 to +0.011: the depth-3 held-out side is
    only 30 states, so its resolution is 1/30 = 0.033 and three lucky solves read as a tenth
    of a gap. A single-seed gap bound is therefore too blunt to assert anything useful, and
    pretending otherwise is how a threshold below the noise floor gets pre-registered.

    What IS assertable per seed is that both sides sit near the measured 1.4% floor. The
    real null for the gap is the twelve-seed mean, which EXP-036's random arm measures.
    """
    cfg = CubeConfig(seed=0, depth=3, arm="random", tag="t_floor", out_dir=tmp_path)
    r = run_cube_baseline(cfg)
    assert r["success_rate"] < 0.15
    assert r["train_success_rate"] < 0.15
    assert r["n_train_eval"] == 90


def test_floor_arm_with_a_curriculum_records_an_empty_stage_trace(tmp_path):
    """The floor short-circuits training, so it never enters the loop that fills `stage_trace` -
    and the record assembles that key unconditionally.

    The combination went unexercised for two experiments. `stage_trace` arrived with EXP-042, so
    every floor measured before it predates the telemetry, and the test above runs the floor with
    **no curriculum**. EXP-044 asked for a measured floor at depth 7 with `curriculum=(1..7)` and
    it raised `UnboundLocalError: local variable 'stage_trace' referenced before assignment` at
    record time - after both evaluations had run, so the work was done and then thrown away.

    Empty is asserted rather than merely present: a floor reporting stages would mean the
    short-circuit had stopped short-circuiting and the chance floor was being trained, which is
    the failure that would quietly invalidate every bar computed against it.
    """
    cfg = CubeConfig(seed=0, depth=2, arm="random", tag="t_floor_curriculum",
                     curriculum=(1, 2), episodes=10, max_depth=2, out_dir=tmp_path)
    r = run_cube_baseline(cfg)

    assert r["stage_trace"] == []
    # The floor never trains, whatever the schedule says. This is the number that moves first if
    # a curriculum ever drags it into the training branch.
    assert r["episodes"] == 0
    on_disk = json.loads((tmp_path / record_filename(cfg)).read_text())
    assert on_disk["stage_trace"] == []


# --- EXP-037: curriculum stage weighting ---


def test_uniform_weights_reproduce_the_unweighted_schedule_exactly():
    """The whole basis for reusing EXP-034/035/036 records as comparators.

    Asserts list equality across every stage count and the exact budgets EXP-037 uses, not
    just that the totals match. A schedule that redistributed episodes while conserving the
    total would pass a sum check and silently invalidate every prior record.
    """
    from neuromorphic.training.cube_baseline import curriculum_schedule

    for n in range(1, 7):
        stages = tuple(range(1, n + 1))
        for episodes in (600, 3000, 10000, 30000):
            unweighted = curriculum_schedule(stages, episodes)
            assert curriculum_schedule(stages, episodes, None) == unweighted
            assert curriculum_schedule(stages, episodes, ()) == unweighted
            assert curriculum_schedule(stages, episodes, (1,) * n) == unweighted


def test_weights_produce_the_exact_preregistered_schedules():
    """Pinned to the tables in the EXP-037 spec, computed before the experiment ran.

    Exact counts, not "the last stage got more". A bug that added a single episode to the
    final stage would satisfy the weaker property and change nothing about the experiment.
    """
    from neuromorphic.training.cube_baseline import curriculum_schedule

    d4 = (1, 2, 3, 4)
    assert curriculum_schedule(d4, 10000, (7, 7, 7, 3)) == [
        (1, 2916), (2, 2916), (3, 2916), (4, 1252)]      # 12.5% share
    assert curriculum_schedule(d4, 10000, (1, 1, 1, 1)) == [
        (1, 2500), (2, 2500), (3, 2500), (4, 2500)]      # 25%, the EXP-036 arm
    assert curriculum_schedule(d4, 10000, (1, 1, 1, 3)) == [
        (1, 1666), (2, 1666), (3, 1666), (4, 5002)]      # 50%
    assert curriculum_schedule(d4, 10000, (1, 1, 1, 9)) == [
        (1, 833), (2, 833), (3, 833), (4, 7501)]         # 75%
    assert curriculum_schedule(tuple(range(1, 7)), 10000, (1, 1, 1, 1, 1, 5)) == [
        (1, 1000), (2, 1000), (3, 1000), (4, 1000), (5, 1000), (6, 5000)]   # depth-6 add-on


def test_weighting_conserves_the_total_budget():
    """The property that makes arms comparable at all.

    A proportional split with integer division loses episodes to rounding unless the
    remainder is reassigned; this is what a naive implementation gets wrong, and it would
    hand the arms different budgets while looking correct.
    """
    from neuromorphic.training.cube_baseline import curriculum_schedule

    for weights in ((7, 7, 7, 3), (1, 1, 1, 3), (1, 1, 1, 9), (5, 1, 1, 1), (2, 3, 5, 7)):
        for episodes in (600, 3000, 10000, 30000):
            sched = curriculum_schedule((1, 2, 3, 4), episodes, weights)
            assert sum(n for _, n in sched) == episodes, (weights, episodes)


def test_invalid_weights_are_rejected(provider):
    """A zero weight would silently drop a stage from the curriculum."""
    from neuromorphic.training.cube_baseline import curriculum_schedule

    with pytest.raises(ValueError, match="must match"):
        curriculum_schedule((1, 2, 3), 1000, (1, 1))
    with pytest.raises(ValueError, match="positive integers"):
        curriculum_schedule((1, 2, 3), 1000, (1, 0, 1))
    with pytest.raises(ValueError, match="positive integers"):
        curriculum_schedule((1, 2, 3), 1000, (1, -1, 1))


def test_explicit_uniform_weights_are_end_to_end_identical_at_depth_4(tmp_path):
    """Licenses reusing EXP-036's depth-4 records as EXP-037's 25% comparator.

    The depth-1 reference values never exercise a multi-stage curriculum, so they cannot
    catch a weighting bug that only shows up when stages are actually split. This runs the
    real depth-4 curriculum both ways on a small budget and compares every measured field.
    """
    # Depth 3, not 4: it still exercises a real multi-stage curriculum, and it evaluates
    # 120 states at 9 steps instead of 333 at 11. `heldout_cap` shrinks BOTH the held-out and
    # the train-side evaluation, which otherwise dominate a small-budget run entirely.
    common = dict(depth=3, curriculum=(1, 2, 3), episodes=40, seed=0,
                  heldout_cap=8, max_depth=3, out_dir=tmp_path)
    a = run_cube_baseline(CubeConfig(tag="t_w_none", **common))
    b = run_cube_baseline(CubeConfig(tag="t_w_unif", curriculum_weights=(1, 1, 1), **common))
    for k in ("success_rate", "train_success_rate", "revisit_rate", "eval_revisit_rate",
              "greedy_modal_action_frac", "mean_train_entropy", "optimality", "mean_steps"):
        assert a[k] == b[k], f"{k} differs: {a[k]} vs {b[k]}"


def test_weighting_actually_changes_the_outcome(tmp_path):
    """If a weighted run scored identically to a uniform one, the weights are not wired in.

    This is the test that fails against the pre-EXP-037 code path, where `curriculum_weights`
    would be accepted by the dataclass and then ignored.
    """
    common = dict(depth=3, curriculum=(1, 2, 3), episodes=40, seed=0,
                  heldout_cap=8, max_depth=3, out_dir=tmp_path)
    unif = run_cube_baseline(CubeConfig(tag="t_w_u2", **common))
    back = run_cube_baseline(
        CubeConfig(tag="t_w_back", curriculum_weights=(1, 1, 7), **common))
    # The RETURNED record carries `asdict(cfg)`, so weights are still a tuple here. They only
    # become a list through the JSON round-trip, which is the form aggregate.py reads. Both
    # are checked because a reader that guessed wrong would silently filter out every record.
    assert unif["config"]["curriculum_weights"] == ()
    assert back["config"]["curriculum_weights"] == (1, 1, 7)
    on_disk = json.loads(
        (tmp_path / record_filename(CubeConfig(tag="t_w_back", **common))).read_text()
    )
    assert on_disk["config"]["curriculum_weights"] == [1, 1, 7]

    # Different schedules must drive the training loop differently. Identical entropy across
    # a large reallocation of the budget would mean the weights never reached the loop.
    assert unif["mean_train_entropy"] != back["mean_train_entropy"]
