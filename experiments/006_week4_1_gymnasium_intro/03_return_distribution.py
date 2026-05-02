"""
Script 3: Visualize the return distribution under different policies

Plots a histogram of returns across N episodes, for two policies:
random and 'chase-the-pole' heuristic. This is the visual version of
the random-policy baseline you set in the original tutorial.

The point: when you train a learning agent next week, you'll plot its
returns over training on the same axes. If the learned curve doesn't
clearly separate from the random distribution, the agent hasn't learned.

Run:  python3 03_return_distribution.py
Output: return_distribution.png
"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def run_n_episodes(env, policy_fn, n=200):
    returns = []
    for ep in range(n):
        obs, _ = env.reset(seed=ep)
        G = 0.0
        while True:
            action = policy_fn(obs, env)
            obs, r, term, trunc, _ = env.step(action)
            G += r
            if term or trunc:
                break
        returns.append(G)
    return np.array(returns)


env = gym.make("CartPole-v1")

random_returns = run_n_episodes(
    env, lambda obs, env: env.action_space.sample(), n=200
)

heuristic_returns = run_n_episodes(
    env, lambda obs, env: 1 if obs[2] > 0 else 0, n=200
)

env.close()

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 6))

bins = np.linspace(0, 500, 51)
ax.hist(random_returns, bins=bins, alpha=0.6, label=f"Random  (μ={random_returns.mean():.1f})", color="#888")
ax.hist(heuristic_returns, bins=bins, alpha=0.6, label=f"Heuristic 'chase pole'  (μ={heuristic_returns.mean():.1f})", color="#2c7fb8")

ax.axvline(475, color="red", linestyle="--", linewidth=1, label="Solved threshold (475)")

ax.set_xlabel("Return per episode (steps survived)")
ax.set_ylabel("Number of episodes")
ax.set_title("CartPole-v1: Return distribution under two non-learning policies\n(200 episodes each)")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("return_distribution.png", dpi=120)
print("Saved return_distribution.png")
print(f"\nRandom:    mean={random_returns.mean():.1f}, std={random_returns.std():.1f}, max={random_returns.max():.0f}")
print(f"Heuristic: mean={heuristic_returns.mean():.1f}, std={heuristic_returns.std():.1f}, max={heuristic_returns.max():.0f}")
print(f"\nThe heuristic is dramatically better than random but still far below")
print(f"the 475 'solved' threshold. That's the headroom a learning agent fills.")