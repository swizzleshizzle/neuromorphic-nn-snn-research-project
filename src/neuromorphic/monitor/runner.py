"""Drive one Gymnasium episode with a ``Brain`` and stream Frames to a ``TraceSink``."""

from __future__ import annotations

import torch

from neuromorphic.monitor.frame import build_frame
from neuromorphic.monitor.schema import REGION_OUTPUT_KEY, build_header, region_specs

DEFAULT_ACTION_LABELS = ("up", "right", "down", "left")


def record_episode(
    brain,
    env,
    sink,
    *,
    seed: int = 0,
    action_labels=DEFAULT_ACTION_LABELS,
    max_steps: int | None = None,
    store_first: bool = True,
    recall: bool = True,
    generator=None,
) -> dict:
    """Record one episode to ``sink`` (header + one Frame per env step).

    Returns a summary dict: ``steps``, ``total_reward``, ``reached_goal``.
    """
    sink.open(build_header(brain, seed=seed, action_labels=action_labels))

    # seed reseeds the env; the brain's Poisson stochasticity comes from `generator`.
    obs, _ = env.reset(seed=seed)
    if store_first:
        brain.remember(obs, generator=generator)

    total_reward = 0.0
    reached_goal = False
    steps = 0
    limit = max_steps if max_steps is not None else getattr(env, "max_steps", 100)

    while steps < limit:
        out = brain.step(obs, store=False, recall=recall, record=True, generator=generator)
        action = int(out["action"])
        next_obs, reward, terminated, truncated, _ = env.step(action)
        brain.learn(reward)
        total_reward += float(reward)

        task = {
            "agent": [int(obs[0]), int(obs[1])],
            "goal": [int(obs[2]), int(obs[3])],
            "action": action,
            "action_label": action_labels[action],
            "reward": float(reward),
            "return": total_reward,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        frame = build_frame(
            out, episode=0, step=steps, t=float(steps), task=task,
            store=False, recall=recall, grid_n=brain.grid_n,
        )
        sink.write(frame)

        obs = next_obs
        steps += 1
        if terminated:
            reached_goal = True
            break
        if truncated:
            break

    sink.close()
    return {"steps": steps, "total_reward": total_reward, "reached_goal": reached_goal}


def _pad_bypassed_recordings(out: dict, brain) -> None:
    """Ensure every region has an output recording; zero-fill ones bypassed this step."""
    recs = out.setdefault("recordings", {})
    t_b = None
    for rid, key in REGION_OUTPUT_KEY.items():
        r = recs.get(rid)
        if r is not None and key in r:
            tens = r[key]
            t_b = (tens.shape[0], tens.shape[1])
            break
    if t_b is None:
        return
    T, B = t_b
    sizes = {rid: n for rid, _, n, _ in region_specs(brain)}
    for rid, key in REGION_OUTPUT_KEY.items():
        r = recs.get(rid)
        if r is None or key not in r:
            recs.setdefault(rid, {})[key] = torch.zeros(T, B, sizes[rid])


def record_policy_episode(
    brain,
    head,
    env,
    sink,
    *,
    seed: int = 0,
    action_labels=DEFAULT_ACTION_LABELS,
    max_steps: int | None = None,
    recall: bool = False,
    policy_regions=("sensory",),
    generator=None,
) -> dict:
    """Record one episode driven by the trained head (sensory concept -> action logits).

    Unlike ``record_episode`` (which uses the brain's internal ``out['action']``), this
    records the actual trained policy. Region activity is still captured via ``record=True``;
    regions bypassed under ``recall=False`` are zero-filled so frames build without error.
    """
    sink.open(build_header(brain, seed=seed, action_labels=action_labels, policy_regions=list(policy_regions)))
    obs, _ = env.reset(seed=seed)

    total_reward = 0.0
    reached_goal = False
    steps = 0
    limit = max_steps if max_steps is not None else getattr(env, "max_steps", 100)

    while steps < limit:
        out = brain.step(obs, store=False, recall=recall, record=True, generator=generator)
        concept = out["concept"].mean(dim=0)[0]  # inline (monitor must not import training)
        action = int(head(concept).argmax())
        _pad_bypassed_recordings(out, brain)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)

        task = {
            "agent": [int(obs[0]), int(obs[1])],
            "goal": [int(obs[2]), int(obs[3])],
            "action": action,
            "action_label": action_labels[action],
            "reward": float(reward),
            "return": total_reward,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        frame = build_frame(out, episode=0, step=steps, t=float(steps), task=task,
                            store=False, recall=recall, grid_n=brain.grid_n)
        sink.write(frame)

        obs = next_obs
        steps += 1
        if terminated:
            reached_goal = True
            break
        if truncated:
            break

    sink.close()
    return {"steps": steps, "total_reward": total_reward, "reached_goal": reached_goal}
