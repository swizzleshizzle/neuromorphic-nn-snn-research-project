"""EXP-029: the v1 fail-first baseline on the 2x2 cube, plus its unregionalized control.

``run_generalization`` cannot be reused: it is built on ``split_goals``, ``manhattan``
optimality and ``GridWorldEnv``. This is the cube analogue. ``reinforce.py`` IS reused
unchanged, because it is already environment-agnostic.

Difficulty is exact distance-to-solved, not move count, so the collapse curve is read off
a true axis. The distance table is an instrument only: the agent observes raw facelets and
never sees a distance. See docs/superpowers/specs/2026-07-25-cube-baseline-design.md.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from neuromorphic.analysis.ablate import AblatedConcept, AblationSpec
from neuromorphic.brain import Brain
from neuromorphic.encoders import cube_encoder
from neuromorphic.envs.cube import CubeEnv
from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.monolithic import MonolithicBrain
from neuromorphic.training.reinforce import (
    concept_rate,
    ema,
    greedy_action,
    policy_parameters,
    train_episode,
)

CUBE_N_OBS = 144      # 24 facelets x 6 colors
CUBE_OBS_WIDTH = 24   # raw facelets


@dataclass
class CubeConfig:
    """One run: one arm, one depth, one seed, one sigma."""

    seed: int = 0
    depth: int = 1
    arm: str = "regionalized"   # "regionalized" | "monolithic" | "random"
    sigma: float = 0.0          # Gaussian concept-noise dose (the EXP-028 operator)
    episodes: int = 600
    lr: float = 1e-2
    gamma: float = 0.99
    baseline_beta: float = 0.1
    entropy_beta: float = 0.0
    normalize_advantages: bool = False
    content: int = 64
    n_actions: int = 6
    readout: str = "concept"    # "concept" | "memory" | "memory_shuffled" | "memory_amnesic"
    # Depth curriculum for training (EXP-034). Empty means the shipped behaviour: train
    # only at `depth`. When set, `episodes` is SPLIT across the listed depths rather than
    # multiplied, so a curriculum arm never buys extra compute over a fixed-budget arm.
    # Evaluation always happens at `depth`, whatever the schedule.
    curriculum: tuple[int, ...] = ()
    # EXP-037. Proportional shares for the curriculum stages; empty means uniform, which is
    # what every experiment before EXP-037 ran. Empty is a safe sentinel here, unlike the
    # `encoder_seed=0` case that forced `None` in 12bbbf8: an empty weight tuple has no
    # meaningful non-default reading. Must be the same length as `curriculum`.
    curriculum_weights: tuple[int, ...] = ()
    # `seed` alone used to drive FIVE independent things: the encoder init, the head init,
    # the action-sampling stream, the environment's scramble stream and the train/held-out
    # split. EXP-034 Finding 4 could therefore only bound the seed variance, never attribute
    # it. These split it into three. Each defaults to None meaning "fall back to `seed`", so
    # every existing config is byte-identical.
    #   encoder_seed  the frozen brain's random initialisation
    #   train_seed    head init, action sampling, env scramble stream, readout rng
    #   split_seed    the train/held-out state split
    encoder_seed: int | None = None
    train_seed: int | None = None
    split_seed: int | None = None
    # EXP-040 (vault Stage 2). Path to a serialised `SensoryCortex` state dict to load into the
    # brain's sensory region after construction, replacing its random initialisation. The
    # encoder remains FROZEN during RL either way, so this changes WHICH weights the frozen
    # encoder holds and nothing else: the trainable parameter count stays at 390.
    #
    # `None` is the shipped behaviour and must reproduce every prior cube record byte-for-byte.
    # Verified against a baseline captured before this field existed; see
    # tests/training/test_encoder_seam.py.
    encoder_state_path: str | None = None
    # EXP-042. Per-depth TRAINING step-budget overrides as ((depth, steps), ...).
    #
    # EXP-041 found the trap this exists to close: `max_steps_for(d) = 2d+3` gives depth 1 a
    # budget of 5 where optimal is 1, and a face move has ORDER 4 - so from a one-move scramble
    # any repeated move either inverts it (1 step) or cycles back to solved (3 steps). Both fit.
    # A constant-action policy therefore scores 0.3333 at depth 1 against a random policy's
    # 0.2208, and curriculum stage 1 positively selects for the worst possible policy.
    # Capping depth 1 at 2 steps admits the inverse but not the cycle: 0.1667, below random.
    #
    # TRAINING ONLY. Evaluation always uses `max_steps_for(cfg.depth)`, so an override cannot
    # change how any arm is scored - otherwise the comparison would measure the yardstick.
    # Empty is the shipped behaviour and must reproduce every prior cube record byte-for-byte.
    max_steps_by_depth: tuple[tuple[int, int], ...] = ()
    max_depth: int = 6          # BFS table bound
    heldout_cap: int = 200
    heldout_frac: float = 0.25
    tag: str = "exp029"
    out_dir: Path = field(default_factory=lambda: Path("outputs"))


def max_steps_for(depth: int) -> int:
    """Step budget at a given exact distance. Optimal is ``depth``; this is generous."""
    return 2 * depth + 3


def shell_states(provider: ExactBFSDistance, depth: int) -> list[tuple[int, ...]]:
    """Every state at exact distance ``depth``, sorted so the order is deterministic."""
    return provider.states_at_distance(depth)


def split_shell(
    states: list[tuple[int, ...]],
    depth: int,
    *,
    seed: int,
    heldout_cap: int = 200,
    heldout_frac: float = 0.25,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]], bool]:
    """Partition a shell into (train, eval, is_heldout).

    Depths 1 and 2 have only 6 and 27 states. Holding out 1 of 6 is not a generalization
    test, so those depths are NOT split: train and eval are both the whole shell and the
    caller must label the number training-distribution. Deeper shells are split, with the
    held-out side capped so a single evaluation stays affordable (brain.step is 90 ms).
    """
    if depth <= 2:
        return list(states), list(states), False
    shuffled = list(states)
    random.Random(seed).shuffle(shuffled)
    n_eval = min(heldout_cap, int(len(shuffled) * heldout_frac))
    return shuffled[n_eval:], shuffled[:n_eval], True


def sample_train_eval(
    train_states: list[tuple[int, ...]],
    *,
    seed: int,
    cap: int = 200,
) -> list[tuple[int, ...]]:
    """A capped, deterministic subset of the train side, for measuring the train/held-out gap.

    EXP-036 needs train-side success next to held-out success. Evaluating the WHOLE train
    side is not affordable and the cap is load-bearing, not an optimisation. At 90 ms per
    ``brain.step`` an uncapped train-side evaluation costs, per seed:

        depth 3     90 states x  9 steps ->   1.2 min
        depth 4    401 states x 11 steps ->   6.6 min
        depth 5   2056 states x 13 steps ->  40.1 min
        depth 6   8769 states x 15 steps -> 197.3 min

    Depth 6 alone would be 3.3 h per seed, roughly 40 core-hours over twelve seeds, against
    about 2.6 h of actual training per run. The instrument would cost more than the thing it
    instruments.

    The cap also makes the gap a FAIR comparison. ``split_shell`` already caps the held-out
    side at ``heldout_cap``, so an uncapped train side would put a 200-state estimate against
    an 8,769-state one and the difference in sampling noise would show up as gap. Matching
    the cap matches the noise.

    Drawn from ``split_seed`` rather than ``train_seed``: which states are evaluated is a
    property of the split, not of the training run, so two arms sharing a split compare on
    identical states.
    """
    if len(train_states) <= cap:
        return list(train_states)
    shuffled = list(train_states)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:cap]


class ShellCubeEnv(CubeEnv):
    """A ``CubeEnv`` whose ``reset()`` draws its start state from a fixed pool.

    ``train_episode`` calls ``env.reset()`` with no arguments, so this is how the training
    loop is confined to the train side of the split without touching ``reinforce.py``.
    """

    def __init__(self, states, rng: random.Random, **kwargs):
        super().__init__(**kwargs)
        self._pool = list(states)
        self._pool_rng = rng
        self.visited: list[tuple[int, ...]] = []

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if options is None:
            options = {"state": self._pool_rng.choice(self._pool)}
        result = super().reset(seed=seed, options=options)
        self.visited = [self._state]
        return result

    def step(self, action):
        result = super().step(action)
        self.visited.append(self._state)
        return result


FAMILIARITY_WIDTH = 1


def feature_width(cfg: CubeConfig) -> int:
    """Width of the policy head's input for a config's readout mode."""
    if cfg.readout == "concept":
        return cfg.content
    if cfg.readout in ("memory", "memory_shuffled", "memory_amnesic"):
        return cfg.content * 2 + FAMILIARITY_WIDTH
    raise ValueError(f"unknown readout {cfg.readout!r}")


class MemoryReadout:
    """Builds the policy head's input from a ``brain.step`` output.

    ``concept``: the sensory concept alone, exactly v1.
    ``memory``: concept, the hippocampal recall code, and a familiarity scalar.
    ``memory_shuffled``: the same three, but recall and familiarity are computed from a
    DIFFERENT state visited earlier this episode. Both memory readouts stay real,
    in-distribution and correctly scaled; only the correspondence between the agent's
    current state and the memory it receives is destroyed. That isolates memory content
    from head width, which is the confound the control exists for.
    ``memory_amnesic``: the same feed-forward expansion of the CURRENT concept as
    ``memory``, but queried against an emptied attractor (``W_rec`` zeroed for the read).
    This isolates memory CONTENT from "more features of the state the agent is already
    looking at": measured over 79 real policy steps, 65% of the recall block's energy is
    a memory-free nonlinear transform of the current concept (cosine 0.802 against a
    ``W_rec``-zeroed version), so ``memory`` vs ``memory_shuffled`` alone cannot rule out
    that a win is just "wider head reading the current state" rather than memory helping.
    """

    def __init__(self, mode: str, rng: random.Random, brain):
        self.mode = mode
        self.rng = rng
        self.brain = brain
        self._cache: list[torch.Tensor] = []
        self.unshuffled_steps = 0

    def reset(self) -> None:
        """Drop the episode's cached concepts. Call at every episode start."""
        if self.mode not in ("concept", "memory", "memory_shuffled", "memory_amnesic"):
            raise ValueError(f"unknown readout {self.mode!r}")
        self._cache = []
        self.unshuffled_steps = 0

    def __call__(self, out: dict) -> torch.Tensor:
        with torch.no_grad():
            concept = concept_rate(out)                     # [content]
            if self.mode == "concept":
                return concept

            snapshot = out["concept"].mean(dim=0)           # [B, content]

            if self.mode == "memory_amnesic":
                # Same feed-forward expansion of the CURRENT concept, zero stored content.
                # This is the arm that isolates memory content from "more features of the
                # state the agent is already looking at".
                saved = self.brain.hippo.W_rec
                self.brain.hippo.W_rec = torch.zeros_like(saved)
                try:
                    recall = self.brain.hippo(
                        snapshot.unsqueeze(0).expand(self.brain.T, *snapshot.shape)
                    ).mean(dim=0)[0]
                    fam = self.brain.hippo.familiarity(snapshot)
                finally:
                    self.brain.hippo.W_rec = saved
                self._cache.append(snapshot)
                return torch.cat([concept, recall, fam[:1]])

            if self.mode == "memory_shuffled":
                if len(self._cache) >= 1:
                    query = self.rng.choice(self._cache)
                else:
                    self.unshuffled_steps += 1
                    query = snapshot
            else:
                query = snapshot
            self._cache.append(snapshot)

            recall = self.brain.hippo(
                query.unsqueeze(0).expand(self.brain.T, *query.shape)
            ).mean(dim=0)[0]                                # [content]
            fam = self.brain.hippo.familiarity(query)       # [B]
            return torch.cat([concept, recall, fam[:1]])


def make_agent(cfg: CubeConfig):
    """Build the arm's feature extractor. Both arms are frozen at random init.

    Uses `encoder_seed`, which falls back to `seed`, so holding the encoder fixed while
    varying training is possible without also changing the brain.
    """
    encoder_seed = resolve_seed(cfg, "encoder")
    if cfg.arm == "regionalized":
        brain = Brain(
            encoder=cube_encoder(), n_obs=CUBE_N_OBS, obs_width=CUBE_OBS_WIDTH,
            n_actions=cfg.n_actions, content=cfg.content, seed=encoder_seed,
        )
        if cfg.encoder_state_path:
            # EXP-040. `strict=True` deliberately: a silently partial load would leave half a
            # random encoder in place, and every downstream number would then describe an
            # architecture nobody chose. The region stays frozen; only its weights differ.
            brain.sensory.load_state_dict(
                torch.load(cfg.encoder_state_path, map_location="cpu"), strict=True
            )
        return brain
    if cfg.arm == "monolithic":
        if cfg.encoder_state_path:
            # The monolithic arm has no `sensory` region to load into. Refusing beats loading
            # nothing and reporting a "pretrained" number that is a random encoder.
            raise ValueError("encoder_state_path is only supported for arm='regionalized'")
        reference = Brain(
            encoder=cube_encoder(), n_obs=CUBE_N_OBS, obs_width=CUBE_OBS_WIDTH,
            n_actions=cfg.n_actions, content=cfg.content, seed=encoder_seed,
        )
        return MonolithicBrain(
            n_obs=CUBE_N_OBS, n_actions=cfg.n_actions, total_neurons=reference.n_neurons,
            content=cfg.content, obs_width=CUBE_OBS_WIDTH, encoder=cube_encoder(),
            seed=encoder_seed,
        )
    # "random" never reaches here: run_cube_baseline short-circuits it, since the chance
    # floor needs no feature extractor at all.
    raise ValueError(f"unknown arm {cfg.arm!r} (expected regionalized or monolithic)")


def resolve_seed(cfg, which: str) -> int:
    """The effective seed for one role, falling back to `cfg.seed` when unset.

    `None` is the only sentinel, deliberately. A truthiness check would treat
    `encoder_seed=0` as unset, and 0 is the most-used seed in this repo.
    """
    value = getattr(cfg, f"{which}_seed")
    return cfg.seed if value is None else value


def curriculum_schedule(
    stages: tuple[int, ...],
    episodes: int,
    weights: tuple[int, ...] | None = None,
) -> list[tuple[int, int]]:
    """Split `episodes` across `stages` in order, as [(depth, n_episodes), ...].

    The total is CONSERVED. A curriculum that ran `episodes` at every stage would train N
    times longer than the arm it is compared against, and any win would be attributable to
    compute rather than to the schedule.

    The remainder goes to the final stage, which is the evaluated depth.

    `weights` (EXP-037) gives the stages unequal shares, proportionally. `None` or empty
    means uniform and produces **the identical list** the unweighted version produced: with
    every weight 1, `episodes * 1 // n` is exactly `episodes // n`, so the default path is
    byte-identical and every EXP-034/035/036 record remains comparable.

    Why this needed a code change at all, rather than the obvious hack: weights can be
    faked by REPEATING a depth in `stages`, since `(1, 2, 3, 4, 4, 4)` does give depth 4
    half the budget. **That is a trap.** `run_cube_baseline` builds a fresh `ShellCubeEnv`
    per stage with `random.Random(train_seed)`, so consecutive identical stages replay the
    SAME start-state sequence rather than drawing new ones. It is not 5,000 fresh episodes
    at depth 4, it is one 1,666-episode sequence looped three times with head and optimizer
    state carrying over, and a win under that scheme would be unattributable. Real weights
    give one continuous stage with one continuous stream.
    """
    if len(stages) > episodes:
        raise ValueError(
            f"episode budget {episodes} too small for {len(stages)} stages"
        )
    if not weights:
        weights = (1,) * len(stages)
    if len(weights) != len(stages):
        raise ValueError(
            f"{len(weights)} weights for {len(stages)} stages; they must match"
        )
    # A zero weight would silently drop a stage from the curriculum, and a negative one
    # would steal episodes from its neighbours. Both are far easier to debug here.
    if any(not isinstance(w, int) or w < 1 for w in weights):
        raise ValueError(f"weights must be positive integers, got {weights}")
    total = sum(weights)
    counts = [episodes * w // total for w in weights]
    counts[-1] += episodes - sum(counts)
    return list(zip(stages, counts))


def record_filename(cfg) -> str:
    """Name of the JSON record a run writes. One file per run, never a shared append.

    NOTE: this encodes tag, arm, depth, seed and sigma, and NOT `entropy_beta`,
    `normalize_advantages`, `episodes`, `curriculum`, `encoder_seed`, `train_seed` or
    `split_seed`. Any sweep over those must make `tag` unique per cell or the records
    overwrite each other silently. This matters most for a seed-decomposition sweep, where
    `seed` is held constant while `encoder_seed` and `train_seed` are crossed: every cell
    would land in one file. Exposed as a function so a sweep driver's collision guard tests
    the real naming rather than a copy of it.
    """
    return f"{cfg.tag}_{cfg.arm}_d{cfg.depth}_s{cfg.seed}_sig{cfg.sigma}.json"


def head_filename(cfg) -> str:
    """Name of the head checkpoint a run writes, beside its JSON record.

    Shares ``record_filename``'s stem so a record and its weights are trivially paired, and
    inherits exactly the same collision warning: a sweep over anything the stem does not
    encode must make ``tag`` unique per cell, or the checkpoints overwrite each other as
    silently as the records do.
    """
    return record_filename(cfg).replace(".json", "_head.pt")


def modal_action_fraction(actions) -> float:
    """Fraction of a rollout spent on its single most-common action.

    A policy that has collapsed to a constant action scores 1.0. A uniform policy over
    the 6 cube moves averages 0.354 over a 9-step budget and 0.429 over 5 steps (measured
    over 20,000 simulated rollouts, 2026-07-31), so collapse is well separated from
    chance even on the short budgets used here.
    """
    if not actions:
        return 0.0
    counts: dict[int, int] = {}
    for a in actions:
        counts[a] = counts.get(a, 0) + 1
    return max(counts.values()) / len(actions)


def evaluate_states(
    agent,
    head,
    states,
    *,
    depth: int,
    generator: torch.Generator | None = None,
    random_policy: bool = False,
    rng_seed: int = 0,
    feature_fn=None,
    store: bool = False,
    recall: bool = False,
) -> dict:
    """Greedy rollouts from each state. ``random_policy`` measures the chance floor.

    ``feature_fn``/``store``/``recall`` mirror ``train_episode``'s memory kwargs so a
    memory-readout head can be evaluated with the same readout it trained with (default
    ``None``/``False``/``False`` reproduces the old concept-only eval exactly). Each
    state's rollout is treated as its own episode: the readout's cache and the
    hippocampus are cleared before it starts, so held-out states don't leak memory.

    Also tracks ``eval_revisit_rate``: the revisit rate under the deterministic GREEDY
    policy actually used here, as opposed to ``run_cube_baseline``'s ``revisit_rate``
    which is measured over the stochastic sampled TRAINING policy (including the
    untrained phase). Under near-uniform sampling a revisit happens whenever a move is
    undone, about 1 in 6 by construction, so the training-time number reads misleadingly
    high and always clears the pre-registered gate. A repeat under the greedy policy is a
    deterministic cycle, which is what memory could actually break, so it is the
    pre-registered quantity. ``CubeEnv`` (used here, unlike ``ShellCubeEnv``) has no
    built-in visit tracking, so it is tracked locally per rollout via ``env._state``.
    """
    limit = max_steps_for(depth)
    env = CubeEnv(scramble_depth=depth, max_steps=limit, scramble_seed=rng_seed)
    rng = random.Random(rng_seed)
    solved = 0
    steps_solved: list[int] = []
    eval_revisits = 0
    eval_steps = 0
    modal_fracs: list[float] = []
    for state in states:
        obs, _ = env.reset(options={"state": state})
        if feature_fn is not None:
            feature_fn.reset()
        if store:
            agent.hippo.clear()
        visited = [env._state]
        actions: list[int] = []
        for t in range(1, limit + 1):
            if random_policy:
                action = rng.randrange(env.action_space.n)
            else:
                with torch.no_grad():
                    action = greedy_action(
                        agent, head, obs, generator=generator,
                        store=store, recall=recall, feature_fn=feature_fn,
                    )
            actions.append(int(action))
            obs, _, terminated, truncated, _ = env.step(action)
            visited.append(env._state)
            eval_steps += 1
            if terminated:
                solved += 1
                steps_solved.append(t)
                break
            if truncated:
                break
        eval_revisits += len(visited) - len(set(visited))
        modal_fracs.append(modal_action_fraction(actions))
    n = len(states)
    total_steps = sum(steps_solved)
    return {
        "success_rate": solved / n if n else 0.0,
        "mean_steps": total_steps / len(steps_solved) if steps_solved else 0.0,
        "optimality": (depth * len(steps_solved) / total_steps) if total_steps else 0.0,
        "n": n,
        "eval_revisit_rate": (eval_revisits / eval_steps) if eval_steps else 0.0,
        "greedy_modal_action_frac": (sum(modal_fracs) / len(modal_fracs)) if modal_fracs else 0.0,
    }


def run_cube_baseline(cfg: CubeConfig) -> dict:
    """One (arm, depth, seed, sigma) run. Returns a JSON-safe record."""
    torch.set_num_threads(1)
    train_seed = resolve_seed(cfg, "train")
    split_seed = resolve_seed(cfg, "split")
    torch.manual_seed(train_seed)
    generator = torch.Generator().manual_seed(train_seed)

    provider = ExactBFSDistance(max_depth=max(cfg.max_depth, cfg.depth, *cfg.curriculum or (0,)))
    states = shell_states(provider, cfg.depth)
    train_states, eval_states, is_heldout = split_shell(
        states, cfg.depth, seed=split_seed,
        heldout_cap=cfg.heldout_cap, heldout_frac=cfg.heldout_frac,
    )

    # Which train-side states the gap is measured on. Fixed by the split, so every arm
    # sharing a `split_seed` is scored on identical states.
    train_eval_states = sample_train_eval(
        train_states, seed=split_seed, cap=cfg.heldout_cap
    )

    if cfg.arm == "random":
        result = evaluate_states(
            None, None, eval_states, depth=cfg.depth, random_policy=True, rng_seed=train_seed
        )
        # The chance floor cannot overfit, so its gap must come out near zero. That is the
        # control on the gap instrument itself: if the random arm shows a gap, the number is
        # measuring the sampling difference between the two sides rather than generalisation.
        train_result = evaluate_states(
            None, None, train_eval_states, depth=cfg.depth,
            random_policy=True, rng_seed=train_seed,
        )
        episodes_run = 0
        revisits, steps_total, stored_counts = 0, 0, []
        unshuffled_steps = 0
        entropies: list[float] = []  # the chance floor never trains, so there is no policy
    else:
        agent = make_agent(cfg)
        torch.manual_seed(train_seed)  # head init and sampling stream matched across arms
        readout = MemoryReadout(cfg.readout, random.Random(train_seed), agent)
        use_memory = cfg.readout != "concept"
        spec = AblationSpec(kind="gaussian", dose=cfg.sigma, seed=train_seed) if cfg.sigma else None
        head = AblatedConcept(
            nn.Linear(feature_width(cfg), cfg.n_actions), spec, width=feature_width(cfg)
        )
        optimizer = torch.optim.Adam(policy_parameters(head), lr=cfg.lr)
        # One stage when no curriculum is set, so the environment is constructed exactly
        # once from random.Random(train_seed) as before and the default path is unchanged.
        stages = curriculum_schedule(
            cfg.curriculum or (cfg.depth,), cfg.episodes, cfg.curriculum_weights or None
        )
        baseline = 0.0
        revisits, steps_total, stored_counts = 0, 0, []
        unshuffled_steps = 0
        entropies = []
        # Per-stage telemetry (week 19 diagnosis). `mean_train_entropy` is one number for the
        # whole run, which cannot say WHEN a policy collapsed - and EXP-040 produced two seeds
        # that ended at entropy 0.04 with modal 1.000 while their encoders measured completely
        # normal. This records the trajectory so the stage at which entropy dies is visible.
        # ADDITIVE ONLY: a new record key, no existing field or code path changes.
        stage_trace = []
        for stage_depth, stage_episodes in stages:
            if stage_depth == cfg.depth:
                stage_train = train_states
            else:
                # Shells at different distances are disjoint, so an earlier stage cannot
                # leak the evaluated depth's held-out states.
                stage_train, _, _ = split_shell(
                    shell_states(provider, stage_depth), stage_depth, seed=split_seed,
                    heldout_cap=cfg.heldout_cap, heldout_frac=cfg.heldout_frac,
                )
            # EXP-042 seam. TRAINING budget only; evaluation below always uses
            # max_steps_for(cfg.depth), so an override cannot change how an arm is scored.
            stage_limit = dict(cfg.max_steps_by_depth).get(
                stage_depth, max_steps_for(stage_depth)
            )
            env = ShellCubeEnv(
                stage_train, random.Random(train_seed),
                scramble_depth=stage_depth, max_steps=stage_limit,
            )
            stage_ents: list[float] = []
            stage_solved = 0
            for _ in range(stage_episodes):
                readout.reset()
                if use_memory:
                    agent.hippo.clear()
                stats = train_episode(
                    agent, head, env, optimizer,
                    gamma=cfg.gamma, baseline=baseline, generator=generator,
                    max_steps=stage_limit,
                    entropy_beta=cfg.entropy_beta,
                    normalize_advantages=cfg.normalize_advantages,
                    store=use_memory, recall=use_memory, feature_fn=readout,
                )
                baseline = ema(baseline, stats["mean_return"], cfg.baseline_beta)
                entropies.append(stats["mean_entropy"])
                stage_ents.append(stats["mean_entropy"])
                stage_solved += int(stats["reached_goal"])
                steps_total += stats["steps"]
                revisits += len(env.visited) - len(set(env.visited))
                stored_counts.append(agent.hippo.n_stored if use_memory else 0)
                unshuffled_steps += readout.unshuffled_steps
            if stage_ents:
                tenth = max(1, len(stage_ents) // 10)
                stage_trace.append({
                    "depth": stage_depth,
                    "episodes": stage_episodes,
                    "entropy_first_10pct": sum(stage_ents[:tenth]) / tenth,
                    "entropy_last_10pct": sum(stage_ents[-tenth:]) / tenth,
                    "entropy_min": min(stage_ents),
                    "train_solved_frac": stage_solved / len(stage_ents),
                })
        result = evaluate_states(
            agent, head, eval_states, depth=cfg.depth, generator=generator, rng_seed=train_seed,
            feature_fn=readout if use_memory else None, store=use_memory, recall=use_memory,
        )
        # STRICTLY AFTER the held-out evaluation. `greedy_action` draws on `generator`, so
        # evaluating the train side first would advance the stream and move every held-out
        # number in this file. The byte-identity test against the EXP-030 reference values is
        # what catches that if this order is ever swapped.
        train_result = evaluate_states(
            agent, head, train_eval_states, depth=cfg.depth, generator=generator,
            rng_seed=train_seed,
            feature_fn=readout if use_memory else None, store=use_memory, recall=use_memory,
        )
        episodes_run = cfg.episodes

    record = {
        "arm": cfg.arm,
        "depth": cfg.depth,
        "seed": cfg.seed,
        "sigma": cfg.sigma,
        "episodes": episodes_run,
        "is_heldout": is_heldout,
        "n_train": len(train_states),
        # EXP-036. `train_success_rate` is scored on `n_train_eval` states drawn from the
        # train side, capped to match the held-out cap. `generalisation_gap` is the
        # pre-registered quantity: train minus held-out, positive meaning the policy does
        # better on states it trained on.
        "train_success_rate": train_result["success_rate"],
        "n_train_eval": train_result["n"],
        "generalisation_gap": train_result["success_rate"] - result["success_rate"],
        "tag": cfg.tag,
        "readout": cfg.readout,
        "revisit_rate": (revisits / steps_total) if steps_total else 0.0,
        "mean_train_entropy": (sum(entropies) / len(entropies)) if entropies else 0.0,
        "stage_trace": stage_trace,
        "mean_n_stored": (sum(stored_counts) / len(stored_counts)) if stored_counts else 0.0,
        "unshuffled_steps": unshuffled_steps,
        "unshuffled_frac": (unshuffled_steps / steps_total) if steps_total else 0.0,
        "config": {**asdict(cfg), "out_dir": str(cfg.out_dir)},
        **result,
    }

    # One file per run, never a shared append. The driver fans out over processes, and
    # concurrent appends to a single file interleave and corrupt lines on Windows.
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / record_filename(cfg)).write_text(json.dumps(record), encoding="utf-8")

    # Serialise the trained head. Until EXP-036 no weights were saved anywhere, so every
    # question of the form "evaluate that trained policy differently" cost a full retrain:
    # the train/held-out gap, the EXP-030 memory re-ask, any depth-transfer probe. The head
    # is a Linear(64 -> 6), 390 parameters, so this is close to free and removes that cost
    # permanently. Written after the record, and after both evaluations, so a failure here
    # cannot corrupt or withhold the run's actual result.
    if cfg.arm != "random":
        torch.save(head.state_dict(), out_dir / head_filename(cfg))
    return record
