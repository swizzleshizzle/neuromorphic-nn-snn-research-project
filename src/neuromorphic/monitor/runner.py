"""Drive one Gymnasium episode with a ``Brain`` and stream Frames to a ``TraceSink``."""

from __future__ import annotations

import torch

from neuromorphic.monitor.frame import build_frame
from neuromorphic.monitor.schema import REGION_OUTPUT_KEY, build_header, region_specs
from neuromorphic.monitor.tasks import GRID_ACTION_LABELS, GridworldAdapter

DEFAULT_ACTION_LABELS = GRID_ACTION_LABELS


def record_episode(
    brain,
    env,
    sink,
    *,
    seed: int = 0,
    max_steps: int | None = None,
    store_first: bool = True,
    recall: bool = True,
    adapter=None,
    generator=None,
) -> dict:
    """Record one episode to ``sink`` (header + one Frame per env step).

    Returns a summary dict: ``steps``, ``total_reward``, ``reached_goal``.
    """
    if adapter is None:
        adapter = GridworldAdapter(brain.grid_n)
    sink.open(build_header(brain, seed=seed, adapter=adapter))

    # seed reseeds the env; the brain's Poisson stochasticity comes from `generator`.
    total_reward = 0.0
    reached_goal = False
    steps = 0
    try:
        obs, _ = env.reset(seed=seed)
        if store_first:
            brain.remember(obs, generator=generator)

        limit = max_steps if max_steps is not None else getattr(env, "max_steps", 100)

        while steps < limit:
            out = brain.step(obs, store=False, recall=recall, record=True, generator=generator)
            action = int(out["action"])
            next_obs, reward, terminated, truncated, info = env.step(action)
            brain.learn(reward)
            total_reward += float(reward)

            task = adapter.frame_task(
                obs, action=action, reward=reward, total=total_reward,
                terminated=terminated, truncated=truncated, info=info,
            )
            frame = build_frame(
                out, episode=0, step=steps, t=float(steps), task=task,
                store=False, recall=recall, adapter=adapter,
            )
            sink.write(frame)

            obs = next_obs
            steps += 1
            if terminated:
                reached_goal = True
                break
            if truncated:
                break
    finally:
        # Always flush/close so an exception mid-episode can't leak the handle
        # (Windows file lock) or leave a silently-truncated trace behind.
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
    max_steps: int | None = None,
    recall: bool = False,
    policy_regions=("sensory",),
    adapter=None,
    generator=None,
) -> dict:
    """Record one episode driven by the trained head (sensory concept -> action logits).

    Unlike ``record_episode`` (which uses the brain's internal ``out['action']``), this
    records the actual trained policy. Region activity is still captured via ``record=True``;
    regions bypassed under ``recall=False`` are zero-filled so frames build without error.
    """
    if adapter is None:
        adapter = GridworldAdapter(brain.grid_n)
    sink.open(build_header(brain, seed=seed, adapter=adapter, policy_regions=list(policy_regions)))

    total_reward = 0.0
    reached_goal = False
    steps = 0
    try:
        obs, _ = env.reset(seed=seed)
        limit = max_steps if max_steps is not None else getattr(env, "max_steps", 100)

        while steps < limit:
            out = brain.step(obs, store=False, recall=recall, record=True, generator=generator)
            concept = out["concept"].mean(dim=0)[0]  # inline (monitor must not import training)
            action = int(head(concept).argmax())
            _pad_bypassed_recordings(out, brain)
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)

            task = adapter.frame_task(
                obs, action=action, reward=reward, total=total_reward,
                terminated=terminated, truncated=truncated, info=info,
            )
            frame = build_frame(out, episode=0, step=steps, t=float(steps), task=task,
                                store=False, recall=recall, adapter=adapter)
            sink.write(frame)

            obs = next_obs
            steps += 1
            if terminated:
                reached_goal = True
                break
            if truncated:
                break
    finally:
        # Always flush/close so an exception mid-episode can't leak the handle
        # (Windows file lock) or leave a silently-truncated trace behind.
        sink.close()
    return {"steps": steps, "total_reward": total_reward, "reached_goal": reached_goal}
