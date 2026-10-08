"""Look-ahead move selection over the true simulator (EXP-070).

The PROCEDURE lives here: enumerate move sequences with the pure simulator, take a solve if one is
within reach, otherwise score the sequences and play the first move of the best. It is general:
nothing here knows the cube has 6 moves, and nothing here may read the BFS distance table, which
is an instrument and never an input (spec section 2.3). A test enforces the second rule at the
source level.
"""

from __future__ import annotations

import itertools

import torch

from neuromorphic.envs.cube import apply_move, is_solved


def solving_prefix(state, k, n_actions, *, apply_fn=apply_move, goal_fn=is_solved):
    """The shortest sequence of length 1..k that reaches the goal, or None.

    Lengths are tried in increasing order and sequences in `itertools.product` order, so the
    result is deterministic and ties go to the lowest action indices.
    """
    for length in range(1, k + 1):
        for seq in itertools.product(range(n_actions), repeat=length):
            s = state
            for a in seq:
                s = apply_fn(s, a)
            if goal_fn(s):
                return seq
    return None


def tree_levels(state, k, n_actions, *, apply_fn=apply_move):
    """levels[l] = the states every length-l sequence reaches, in product order."""
    levels = [[state]]
    for _ in range(k):
        levels.append([apply_fn(s, a) for s in levels[-1] for a in range(n_actions)])
    return levels


def sequence_scores(level_logp, k, n_actions):
    """Summed log-probability of every length-k sequence, indexed in product order.

    For sequence index j, the move at level l is (j // n**(k-l-1)) % n and the state it is taken
    from is level l's index j // n**(k-l).
    """
    j = torch.arange(n_actions ** k)
    scores = torch.zeros(n_actions ** k, dtype=level_logp[0].dtype)
    for l in range(k):
        parent = j // n_actions ** (k - l)
        move = (j // n_actions ** (k - l - 1)) % n_actions
        scores = scores + level_logp[l][parent, move]
    return scores


import numpy as np

from neuromorphic.envs.cube import CubeEnv
from neuromorphic.training.cube_baseline import max_steps_for, modal_action_fraction
from neuromorphic.training.reinforce import action_distribution

MODES = ("G", "E", "P", "R", "C")


def imag_seed_for(base: int, state_index: int, step: int) -> int:
    """A distinct, deterministic seed per (cell, held-out state, real step)."""
    return (base * 1_000_003 + state_index) * 1_009 + step


def imagined_logp(agent, head, states, *, generator):
    """Log-probabilities over moves for many states, in ONE batched brain call.

    Must read exactly what `action_distribution` reads for a single state: the sensory concept's
    mean rate over the window, with recall off. A test pins this against the evaluation path.
    """
    with torch.no_grad():
        out = agent.step(np.array(states), recall=False, generator=generator)
        features = out["concept"].mean(dim=0)  # [B, content]
        return torch.log_softmax(head(features), dim=-1)


def imagined_values(agent, critic, states, *, generator):
    """The critic's value for many states, in ONE batched call, recall off.

    A critic with `reads_states = True` (EXP-074's judge) is called on the states themselves; it
    owns its own encoder and readout. Any other critic reads `agent.step`'s concept as before.
    """
    with torch.no_grad():
        if getattr(critic, "reads_states", False):
            return critic(states, generator=generator)
        out = agent.step(np.array(states), recall=False, generator=generator)
        return critic(out["concept"].mean(dim=0)).squeeze(-1)


def sequence_eligibility(levels, k, n_actions, visited):
    """True where a length-k sequence passes through no state in `visited`."""
    j = torch.arange(n_actions ** k)
    ok = torch.ones(n_actions ** k, dtype=torch.bool)
    for l in range(1, k + 1):
        bad = torch.tensor([s in visited for s in levels[l]], dtype=torch.bool)
        ok &= ~bad[j // n_actions ** (k - l)]
    return ok


def _greedy_unvisited(state, root_logits, n_actions, visited, info):
    eligible = [a for a in range(n_actions) if apply_move(state, a) not in visited]
    if not eligible:
        if info is not None:
            info["fallback"] = True
        return int(root_logits.argmax())
    return max(eligible, key=lambda a: (float(root_logits[a]), -a))


def choose_move(mode, state, root_logits, k, n_actions, *, agent=None, head=None,
                imag_generator=None, critic=None, visited=None, info=None):
    """One real move. Returns (action, goal_fired). Never sees the evaluation generator."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    greedy = int(root_logits.argmax())
    if mode == "G":
        if visited is not None:
            return _greedy_unvisited(state, root_logits, n_actions, visited, info), False
        return greedy, False
    if k < 1:
        raise ValueError(f"mode {mode} needs k >= 1, got {k}")
    prefix = solving_prefix(state, k, n_actions)
    if prefix is not None:
        return prefix[0], True
    if mode == "E":
        if visited is not None:
            return _greedy_unvisited(state, root_logits, n_actions, visited, info), False
        return greedy, False
    levels = None
    if mode == "R":
        scores = torch.rand(n_actions ** k, generator=imag_generator)
    elif mode == "C":
        if critic is None:
            raise ValueError("mode C needs a critic")
        levels = tree_levels(state, k, n_actions)
        scores = imagined_values(agent, critic, levels[k], generator=imag_generator)
    else:  # "P"
        levels = tree_levels(state, k, n_actions)
        level_logp = [torch.log_softmax(root_logits.reshape(1, -1), dim=-1)]
        for l in range(1, k):
            level_logp.append(imagined_logp(agent, head, levels[l], generator=imag_generator))
        scores = sequence_scores(level_logp, k, n_actions)
    if visited is not None:
        if levels is None:
            levels = tree_levels(state, k, n_actions)
        mask = sequence_eligibility(levels, k, n_actions, visited)
        if bool(mask.any()):
            scores = torch.where(mask, scores, torch.full_like(scores, float("-inf")))
        elif info is not None:
            info["fallback"] = True
    best = int(scores.argmax())  # first maximum, so ties go to the lowest product index
    return best // n_actions ** (k - 1), False


def evaluate_lookahead(agent, head, states, *, depth, mode, k, generator, rng_seed=0,
                       imag_seed=0, trace=None, critic=None, no_revisit=False):
    """`evaluate_states` with a look-ahead move rule. At mode G, k 0 it IS `evaluate_states`.

    Every mode draws the real state's logits from `generator` exactly once per real move, the
    same call `greedy_action` makes. That is what keeps G's stream, and therefore Gate 0, intact,
    and what makes E and P see bit-identical real encodings while they play the same moves.
    """
    if mode == "G" and k != 0:
        raise ValueError("mode G is the published evaluator and takes k=0")
    limit = max_steps_for(depth)
    env = CubeEnv(scramble_depth=depth, max_steps=limit, scramble_seed=rng_seed)
    n_actions = env.action_space.n
    solved = 0
    steps_solved: list[int] = []
    eval_revisits = 0
    eval_steps = 0
    fired_moves = 0
    fallback_moves = 0
    modal_fracs: list[float] = []
    for i, state in enumerate(states):
        obs, _ = env.reset(options={"state": state})
        visited = [env._state]
        visited_set = set(visited)
        actions: list[int] = []
        state_trace: list[tuple[int, bool, list[float]]] = []
        for t in range(1, limit + 1):
            with torch.no_grad():
                _, logits = action_distribution(agent, head, obs, generator=generator)
            imag = torch.Generator().manual_seed(imag_seed_for(imag_seed, i, t))
            info = {}
            action, fired = choose_move(mode, env._state, logits, k, n_actions,
                                        agent=agent, head=head, imag_generator=imag,
                                        critic=critic,
                                        visited=visited_set if no_revisit else None,
                                        info=info)
            fallback_moves += int(info.get("fallback", False))
            fired_moves += int(fired)
            actions.append(int(action))
            state_trace.append((int(action), bool(fired), logits.reshape(-1).tolist()))
            obs, _, terminated, truncated, _ = env.step(action)
            visited.append(env._state)
            visited_set.add(env._state)
            eval_steps += 1
            if terminated:
                solved += 1
                steps_solved.append(t)
                break
            if truncated:
                break
        eval_revisits += len(visited) - len(set(visited))
        modal_fracs.append(modal_action_fraction(actions))
        if trace is not None:
            trace.append(state_trace)
    n = len(states)
    total_steps = sum(steps_solved)
    return {
        "success_rate": solved / n if n else 0.0,
        "mean_steps": total_steps / len(steps_solved) if steps_solved else 0.0,
        "optimality": (depth * len(steps_solved) / total_steps) if total_steps else 0.0,
        "n": n,
        "eval_revisit_rate": (eval_revisits / eval_steps) if eval_steps else 0.0,
        "greedy_modal_action_frac": (sum(modal_fracs) / len(modal_fracs)) if modal_fracs else 0.0,
        "solved": solved,
        "goal_fired_frac": (fired_moves / eval_steps) if eval_steps else 0.0,
        "mode": mode,
        "k": k,
        "no_revisit": bool(no_revisit),
        "fallback_frac": (fallback_moves / eval_steps) if eval_steps else 0.0,
    }
