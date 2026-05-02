"""
Script 2: Save an episode as an animated GIF

Uses render_mode="rgb_array" to grab frames as numpy arrays, then stitches
them into a GIF with PIL. Useful for:
  - Showing your work in YouTube videos / blog posts (Phase 5 deliverable)
  - Comparing policies side-by-side post-hoc
  - Embedding in Obsidian notes (Obsidian renders GIFs inline)

Run:  python3 02_save_episode_gif.py
Output: cartpole_random.gif, frozenlake_optimal.gif
"""
import gymnasium as gym
from PIL import Image


def record_episode(env, policy_fn, max_steps=500):
    """Run an episode and collect frames. Returns list of PIL Images."""
    frames = []
    obs, _ = env.reset(seed=0)
    frames.append(Image.fromarray(env.render()))
    for _ in range(max_steps):
        action = policy_fn(obs, env)
        obs, reward, terminated, truncated, _ = env.step(action)
        frames.append(Image.fromarray(env.render()))
        if terminated or truncated:
            break
    return frames


def save_gif(frames, path, fps=30):
    """Save list of PIL Images as an animated GIF."""
    duration_ms = int(1000 / fps)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,  # 0 = loop forever
        optimize=True,
    )
    print(f"  saved {len(frames)} frames to {path}")


# --- CartPole with random policy ---
env = gym.make("CartPole-v1", render_mode="rgb_array")
frames = record_episode(env, lambda obs, env: env.action_space.sample())
save_gif(frames, "cartpole_random.gif", fps=30)
env.close()

# --- FrozenLake with the hand-coded optimal policy ---
env = gym.make("FrozenLake-v1", is_slippery=False, map_name="4x4",
               render_mode="rgb_array")
optimal_actions = iter([1, 1, 2, 2, 1, 2])  # Down,Down,Right,Right,Down,Right


def optimal_frozen(obs, env):
    return next(optimal_actions)


frames = record_episode(env, optimal_frozen, max_steps=10)
save_gif(frames, "frozenlake_optimal.gif", fps=2)  # slower for FrozenLake
env.close()

print("\nDone. Drop these GIFs into Obsidian — drag-and-drop into a note,")
print("they render inline like images. Useful for the weekly notes.")