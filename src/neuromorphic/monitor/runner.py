"""Drive one Gymnasium episode with a ``Brain`` and stream Frames to a ``TraceSink``."""

from __future__ import annotations

from neuromorphic.monitor.frame import build_frame
from neuromorphic.monitor.schema import build_header

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

    obs, _ = env.reset()
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
