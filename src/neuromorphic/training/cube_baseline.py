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
import statistics
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
    make_critic,
    make_constant_critic,
    policy_parameters,
    train_episode,
)

CUBE_N_OBS = 144      # 24 facelets x 6 colors
CUBE_OBS_WIDTH = 24   # raw facelets

GATE_WARMUP = 100
"""Episodes at the start of a run during which the gate is forced open (EXP-053).

The threshold is a running median and is undefined before there is history. 100 of 10,000
episodes is 1% of the run. Warmup episodes DO enter the median history.
"""

GATE_RNG_OFFSET = 90_000
"""Offset applied to `train_seed` for arm R's gate stream (EXP-053).

A DEDICATED `random.Random`, never the torch `generator` and never the env's stream. If the
gate drew on either, arm R's action sampling and scrambles would diverge from arm G's and the
two arms would differ in more than which episodes updated.
"""


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
    # EXP-047. Learning rate for the SENSORY ENCODER during RL. `None` keeps the encoder
    # FROZEN, which is what every cube result from EXP-029 onward was produced with, and is a
    # strict no-op: `test_encoder_finetune_seam.py` asserts the pre-change baseline still
    # reproduces byte-for-byte, and that `encoder_lr=0.0` (grad ON, zero step) does too.
    #
    # THIS CHANGES THE ARCHITECTURE, not a hyperparameter. Setting it takes the trainable
    # surface from 390 parameters (the `Linear(64 -> 6)` head) to 27,206 (head + `fc1`'s
    # 18,560 + `fc2`'s 8,256), a factor of 70. "The same 390 trainable parameters" is
    # load-bearing in every RESULTS.md and in the whole visual story, so a fine-tuned run is
    # a DIFFERENT ARM and must never be reported as another cell of the depth series.
    #
    # It also costs 1.33x per step (56.17 -> 74.87 ms, measured 2026-08-20 on the VPS), which
    # EXP-046's budget curve prices at +0.027 success at depth 6. See the spec.
    #
    # Only meaningful for `arm="regionalized"` with `readout="concept"`: the monolithic arm has
    # no `sensory` region, and `MemoryReadout` re-wraps the concept in `no_grad`, which would
    # silently detach the encoder and train nothing. Both are refused rather than ignored.
    encoder_lr: float | None = None
    # EXP-053. Learning rate for the CRITIC, a `Linear(64 -> 1)` on the same concept the
    # policy head reads. `None` keeps the scalar EMA baseline, which is what every cube
    # record from EXP-029 onward was produced with, and is a strict no-op.
    #
    # Setting it takes the trainable surface from 390 to 455 and changes the ADVANTAGE from
    # `G_t - baseline` to `G_t - V(s_t)`. That is a different learning rule, not a
    # hyperparameter, so a critic run is a different arm and must never be tabulated as
    # another cell of the depth series.
    #
    # Monte-Carlo, not TD: `train_episode` already computes returns-to-go, so `G_t - V(s_t)`
    # is the smaller change and carries no bootstrapping bias. See the spec section 2.1.
    critic_lr: float | None = None
    # EXP-056. Replaces `V(s_t)` with its episode mean when forming advantages, removing the
    # critic's WITHIN-EPISODE state-dependence while leaving calibration and fitting untouched.
    # Default False is a strict no-op: every cube run before EXP-056 forms advantages against
    # the per-timestep prediction. Requires `critic_lr`; it is meaningless without a critic.
    flatten_critic: bool = False
    # EXP-057. Replaces the state-dependent critic with a single learned scalar, fitted by the
    # same MSE loss at the same rate. Requires `critic_lr`; meaningless without a critic.
    # Default False is a strict no-op: every critic run before EXP-057 reads the state.
    constant_critic: bool = False
    # EXP-053. Gates ENCODER plasticity on the neuromodulatory bus.
    #
    #   None         the encoder steps every episode (EXP-047's behaviour, and the control)
    #   "dopamine"   it steps only when `Brain.learn()` reports `bus.learning_enabled`
    #   "random"     it steps on a coin flip at a rate supplied per seed (the CONTROL for
    #                "dopamine", because a gated arm also does FEWER updates)
    #   "always"     test-only. Forces the gate open, so the split-optimizer machinery can be
    #                proved mathematically identical to the single two-group Adam it replaces.
    #
    # This is the first code in the project to READ `NeuromodBus.learning_enabled`, and the
    # first caller of `Brain.learn()` in a training loop. Requires `encoder_lr`: gating
    # plasticity that does not exist would silently do nothing.
    plasticity_gate: str | None = None
    # EXP-053 arm R. Per-seed update rates as ((seed, rate), ...), taken from arm G's
    # realized `gate_rate` so the two arms perform the same NUMBER of encoder updates and
    # differ only in WHICH episodes got them. Required when `plasticity_gate="random"`;
    # a missing seed is an error rather than a default, because defaulting to 1.0 would
    # silently turn arm R back into arm G's always-on control.
    gate_rate_by_seed: tuple[tuple[int, float], ...] = ()
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
        # EXP-047. The concept path returns BEFORE the `no_grad` below, deliberately.
        #
        # `run_cube_baseline` passes `feature_fn=readout` on the TRAINING call unconditionally,
        # for every readout including "concept" (only the EVALUATION calls pass `None`). So this
        # identity branch is on the policy path of every cube run ever recorded. While it sat
        # inside `no_grad` it detached the concept, and fine-tuning silently trained nothing:
        # the run looked completely normal and `fc1.weight` moved by exactly 0.0.
        #
        # Moving it out is numerically inert for every frozen run - `brain.step` already ran
        # under `no_grad` there, so the tensor carries no graph either way and the returned
        # VALUES are bit-identical. `test_encoder_finetune_seam.py` asserts exactly that against
        # the pre-change baseline.
        #
        # The `no_grad` stays for the memory modes. Those touch `hippo.W_rec` in place and
        # swap it out under `memory_amnesic`, which is not something to build a graph through.
        if self.mode == "concept":
            return concept_rate(out)                        # [content]

        with torch.no_grad():
            concept = concept_rate(out)                     # [content]
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


def encoder_filename(cfg) -> str:
    """Name of the FINE-TUNED encoder a run writes (EXP-047), beside its JSON record.

    Only written when ``encoder_lr`` is set. Shares ``record_filename``'s stem for the same
    pairing reason as ``head_filename``, and inherits the same collision warning.
    """
    return record_filename(cfg).replace(".json", "_encoder.pt")


def within_rms(fit: dict, key: str) -> float:
    """RMS of a within-episode centred sum, over a whole curriculum stage (EXP-056).

    `critic_fit_terms` centres each episode against its OWN mean before summing, so this is a
    pooled within-episode spread and never picks up between-episode variation. That distinction
    is the whole point: flattening removes within-episode variation only.
    """
    n = fit.get("critic_n", 0)
    return (fit[key] / n) ** 0.5 if n else 0.0


def critic_filename(cfg) -> str:
    """Name of the trained CRITIC a run writes (EXP-053), beside its JSON record.

    Only written when ``critic_lr`` is set. Shares ``record_filename``'s stem for the same
    pairing reason as ``head_filename``, and inherits the same collision warning.
    """
    return record_filename(cfg).replace(".json", "_critic.pt")


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


def explained_variance(fit: dict) -> float:
    """`1 - SSE/SST` from the streamed sums (EXP-053). 1.0 is a perfect critic, 0.0 is no
    better than predicting the stage's mean return, and negative is worse than that.

    Returns 0.0 for an empty or constant-return stage, where SST is 0 and the ratio is
    undefined - reported as "no better than the mean", which is what it means.
    """
    n = fit["critic_n"]
    if n == 0:
        return 0.0
    sst = fit["return_sq_sum"] - (fit["return_sum"] ** 2) / n
    if sst <= 0.0:
        return 0.0
    return 1.0 - fit["critic_sse"] / sst


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


class DopamineGate:
    """Opens encoder plasticity when the reward-prediction error clears its running median.

    EXP-053. One call per episode, after the episode has been rolled out:

        bus.learning_threshold = median(dopamine seen so far)
        gate_open              = Brain.learn(mean_return, baseline)

    `Brain.learn` writes `mean_return - baseline` into `bus.dopamine` and returns
    `bus.learning_enabled`, both exactly as they were written in L11. Nothing about the bus
    changes here; it simply acquires its first reader.

    **Why a running median and not `NeuromodBus`'s default 0.5.** That default is meaningless
    against a return scale set by `solve_reward=10.0` and `step_penalty=-1.0`. A quantile
    self-calibrates, needs no tuned constant, puts the realized update rate near 50% by
    construction, and tracks the distribution as the EMA `baseline` moves rather than drifting
    shut.

    **Why signed and not absolute.** `learning_enabled` is `dopamine >= threshold`, which is
    one-sided. Gating on better-than-expected episodes uses it as written and is the literal
    phasic-dopamine story. See the amendment note in the spec, section 2.2.
    """

    def __init__(self, brain, warmup: int = GATE_WARMUP):
        self.brain = brain
        self.warmup = warmup
        self.history: list[float] = []

    def __call__(self, mean_return: float, baseline: float) -> bool:
        dopamine = mean_return - baseline
        self.history.append(dopamine)
        if len(self.history) <= self.warmup:
            # Force it open, but still write the bus so the record is honest about what the
            # signal was during warmup.
            self.brain.learn(mean_return, baseline)
            return True
        self.brain.bus.learning_threshold = statistics.median(self.history)
        return self.brain.learn(mean_return, baseline)


class RandomGate:
    """Opens on a coin flip at a fixed rate, blind to the dopamine value (EXP-053 arm R).

    The control for `DopamineGate`. It holds the update RATE fixed and varies only WHICH
    episodes are chosen, which is the single thing the neuromodulatory claim rests on.
    """

    def __init__(self, rate: float, rng: random.Random):
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"gate rate must be in [0, 1], got {rate}")
        self.rate = rate
        self.rng = rng

    def __call__(self, _mean_return: float, _baseline: float) -> bool:
        return self.rng.random() < self.rate


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
        # ...and for the same reason it never enters the stage loop that fills this. Empty is the
        # honest value, not a missing key: EXP-042 added `stage_trace` and every floor arm before
        # it predates the telemetry, so the combination went unexercised until EXP-044 asked for a
        # measured floor at depth 7 WITH a curriculum. It raised UnboundLocalError at record time.
        stage_trace: list[dict] = []
        trainable_params = 0   # the chance floor has no policy to train
        critic = None   # the chance floor has no critic either
        gate_opens, gate_calls = 0, 0   # the chance floor has no encoder to gate either
    else:
        agent = make_agent(cfg)
        torch.manual_seed(train_seed)  # head init and sampling stream matched across arms
        readout = MemoryReadout(cfg.readout, random.Random(train_seed), agent)
        use_memory = cfg.readout != "concept"
        spec = AblationSpec(kind="gaussian", dose=cfg.sigma, seed=train_seed) if cfg.sigma else None
        head = AblatedConcept(
            nn.Linear(feature_width(cfg), cfg.n_actions), spec, width=feature_width(cfg)
        )
        # EXP-047. The frozen path builds the optimizer exactly as every prior experiment did,
        # from `policy_parameters(head)` alone, so its Adam state and update sequence are
        # unchanged. Fine-tuning ADDS a second parameter group rather than merging the two,
        # which keeps the head's lr and the encoder's separate quantities.
        finetune = cfg.encoder_lr is not None
        gated = cfg.plasticity_gate is not None
        if gated and not finetune:
            raise ValueError(
                f"plasticity_gate requires encoder_lr (got {cfg.plasticity_gate!r} with "
                "encoder_lr=None): gating plasticity that does not exist does nothing."
            )
        if gated and cfg.plasticity_gate not in ("dopamine", "random", "always"):
            raise ValueError(f"unknown plasticity_gate {cfg.plasticity_gate!r}")

        encoder_optimizer = None
        if finetune:
            if cfg.arm != "regionalized":
                raise ValueError(
                    f"encoder_lr requires arm='regionalized' (got {cfg.arm!r}): only that arm "
                    "has a `sensory` region to fine-tune."
                )
            if cfg.readout != "concept":
                # `MemoryReadout.__call__` wraps its body in `torch.no_grad()`, so a memory
                # readout would detach the concept and the encoder would silently receive no
                # gradient at all. The run would look fine and train nothing. Refuse instead.
                raise ValueError(
                    f"encoder_lr requires readout='concept' (got {cfg.readout!r}): MemoryReadout "
                    "detaches the concept, so the encoder would train on nothing."
                )
            if gated:
                # SEPARATE optimizers (EXP-053). Adam applies `exp_avg` on every `step()`, so
                # zeroing the encoder group's gradients inside a shared optimizer would still
                # move the encoder. Skipping a separate optimizer's `step()` is the only way
                # to actually withhold an update. Adam state is per-parameter, so this is
                # mathematically identical to the single two-group Adam while the gate is
                # open, which `test_always_open_gate_reproduces_the_single_optimizer_run`
                # asserts to 1e-6.
                optimizer = torch.optim.Adam(list(policy_parameters(head)), lr=cfg.lr)
                encoder_optimizer = torch.optim.Adam(
                    list(agent.sensory.parameters()), lr=cfg.encoder_lr)
            else:
                optimizer = torch.optim.Adam([
                    {"params": list(policy_parameters(head)), "lr": cfg.lr},
                    {"params": list(agent.sensory.parameters()), "lr": cfg.encoder_lr},
                ])
        else:
            optimizer = torch.optim.Adam(policy_parameters(head), lr=cfg.lr)
        # EXP-053. A SEPARATE optimizer, not a third parameter group, so the head's Adam
        # state and update sequence stay exactly what every prior record was produced with.
        critic = None
        critic_optimizer = None
        if cfg.critic_lr is not None:
            if cfg.arm != "regionalized":
                raise ValueError(
                    f"critic_lr requires arm='regionalized' (got {cfg.arm!r}): the critic "
                    "reads the sensory concept, which the monolithic arm does not have."
                )
            if cfg.readout != "concept":
                raise ValueError(
                    f"critic_lr requires readout='concept' (got {cfg.readout!r}): "
                    "MemoryReadout detaches the concept, so the critic would train on nothing."
                )
            critic = make_constant_critic(agent) if cfg.constant_critic else make_critic(agent)
            critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
        # Counted from the optimizer rather than from the config, so it reports what is
        # ACTUALLY being trained. 390 on every frozen run since EXP-029; 27,206 fine-tuned.
        # Recorded because the depth series is only comparable at a fixed trainable surface,
        # and a number in the record is harder to lose than a sentence in a write-up.
        trainable_params = sum(
            p.numel() for group in optimizer.param_groups for p in group["params"]
        )
        # Task 2 added a critic-only version of this. Both optimizers are folded into one
        # loop so the count stays "what is ACTUALLY being trained" with no double-counting.
        for opt in (critic_optimizer, encoder_optimizer):
            if opt is not None:
                trainable_params += sum(
                    p.numel() for group in opt.param_groups for p in group["params"]
                )
        # One stage when no curriculum is set, so the environment is constructed exactly
        # once from random.Random(train_seed) as before and the default path is unchanged.
        stages = curriculum_schedule(
            cfg.curriculum or (cfg.depth,), cfg.episodes, cfg.curriculum_weights or None
        )
        baseline = 0.0
        revisits, steps_total, stored_counts = 0, 0, []
        unshuffled_steps = 0
        entropies = []
        gate_fn = None
        if cfg.plasticity_gate == "dopamine":
            gate_fn = DopamineGate(agent)
        elif cfg.plasticity_gate == "always":
            gate_fn = lambda _mean_return, _baseline: True   # noqa: E731 - test seam only
        elif cfg.plasticity_gate == "random":
            rates = dict(cfg.gate_rate_by_seed)
            if cfg.seed not in rates:
                raise ValueError(
                    f"plasticity_gate='random' but no gate rate for seed {cfg.seed}; arm R "
                    "is rate-matched to arm G per seed and must not silently default."
                )
            gate_fn = RandomGate(rates[cfg.seed], random.Random(train_seed + GATE_RNG_OFFSET))
        gate_opens, gate_calls = 0, 0
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
            stage_fit = {"critic_sse": 0.0, "return_sum": 0.0,
                         "return_sq_sum": 0.0, "critic_n": 0,
                         "critic_within_ss": 0.0, "return_within_ss": 0.0}
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
                    grad_brain=finetune,
                    critic=critic, critic_optimizer=critic_optimizer,
                    flatten_critic=cfg.flatten_critic,
                    encoder_optimizer=encoder_optimizer, gate_fn=gate_fn,
                )
                baseline = ema(baseline, stats["mean_return"], cfg.baseline_beta)
                entropies.append(stats["mean_entropy"])
                stage_ents.append(stats["mean_entropy"])
                stage_solved += int(stats["reached_goal"])
                steps_total += stats["steps"]
                revisits += len(env.visited) - len(set(env.visited))
                stored_counts.append(agent.hippo.n_stored if use_memory else 0)
                unshuffled_steps += readout.unshuffled_steps
                if encoder_optimizer is not None:
                    gate_calls += 1
                    gate_opens += int(stats["gate_open"])
                if critic is not None:
                    for k in stage_fit:
                        stage_fit[k] += stats[k]
            if stage_ents:
                tenth = max(1, len(stage_ents) // 10)
                stage_trace.append({
                    "depth": stage_depth,
                    "episodes": stage_episodes,
                    "entropy_first_10pct": sum(stage_ents[:tenth]) / tenth,
                    "entropy_last_10pct": sum(stage_ents[-tenth:]) / tenth,
                    "entropy_min": min(stage_ents),
                    "train_solved_frac": stage_solved / len(stage_ents),
                    **({"critic_ev": explained_variance(stage_fit),
                        "critic_n": stage_fit["critic_n"],
                        # EXP-056 validity gate: how much within-episode variation does `V`
                        # actually have, against the returns it is subtracted from?
                        "critic_within_rms": within_rms(stage_fit, "critic_within_ss"),
                        "return_within_rms": within_rms(stage_fit, "return_within_ss")}
                       if critic is not None else {}),
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
        # EXP-047. Additive: a new key, so every prior record simply lacks it and no existing
        # field or code path changes. `encoder_lr` itself rides along inside `config`.
        "trainable_params": trainable_params,
        # EXP-053. Additive: absent from every prior record. The FINAL stage's figure, which
        # is the deepest one and the regime the critic lr was selected in.
        **({"critic_ev": stage_trace[-1]["critic_ev"],
            "critic_n": stage_trace[-1]["critic_n"]} if critic is not None else {}),
        # EXP-053. The realized fraction of episodes on which the encoder actually stepped.
        # Arm R is rate-matched to this per seed, and a rate far from 0.5 is itself
        # diagnostic - it would mean the median threshold is not tracking the distribution.
        **({"gate_rate": gate_opens / gate_calls if gate_calls else 0.0}
           if cfg.plasticity_gate is not None else {}),
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
        # EXP-047. A fine-tuned encoder is a RESULT, not a byproduct: Claim 2 re-probes it to
        # ask whether RL improved the representation or merely fitted the head to it, and that
        # question cannot be asked afterwards if the weights are gone. Frozen runs write
        # nothing here - their encoder is already on disk as `encoder_state_path`, or is
        # reproducible from `encoder_seed`.
        if cfg.encoder_lr is not None:
            torch.save(agent.sensory.state_dict(), out_dir / encoder_filename(cfg))
        if cfg.critic_lr is not None:
            torch.save(critic.state_dict(), out_dir / critic_filename(cfg))
    return record
