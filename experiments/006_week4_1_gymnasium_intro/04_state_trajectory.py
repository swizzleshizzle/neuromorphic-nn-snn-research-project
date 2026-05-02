"""
Script 4: Plot CartPole's state variables over time

CartPole's state is 4 floats — [cart_pos, cart_vel, pole_angle, pole_ang_vel].
Plotting them over time shows you exactly *how* the pole falls and what the
controller has to compensate for.

This is the same view a SCADA HMI would give you for a 4-channel process —
trends of each variable side by side. Useful for:
  - Diagnosing why a policy fails (which variable went out of bounds first?)
  - Comparing a learned policy's behavior to a hand-crafted one
  - Sanity-checking your understanding of what 'state' means

Run:  python3 04_state_trajectory.py
Output: state_trajectory.png
"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def collect_trajectory(env, policy_fn, seed=0):
    """Run one episode and return arrays of state, action, reward over time."""
    obs, _ = env.reset(seed=seed)
    states = [obs]
    actions = []
    rewards = []
    while True:
        action = policy_fn(obs, env)
        obs, r, term, trunc, _ = env.step(action)
        actions.append(action)
        rewards.append(r)
        states.append(obs)
        if term or trunc:
            break
    return np.array(states), np.array(actions), np.array(rewards)


env = gym.make("CartPole-v1")

# Two policies for comparison
random_states, random_actions, _ = collect_trajectory(
    env, lambda obs, env: env.action_space.sample(), seed=42
)
heuristic_states, heuristic_actions, _ = collect_trajectory(
    env, lambda obs, env: 1 if obs[2] > 0 else 0, seed=42
)
env.close()

# --- Plot ---
fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex="col")

var_names = ["Cart position (m)", "Cart velocity (m/s)",
             "Pole angle (rad)", "Pole ang. velocity (rad/s)"]
# Failure thresholds from the env source
limits = [(-2.4, 2.4), None, (-0.2095, 0.2095), None]

for col, (states, actions, label) in enumerate([
    (random_states, random_actions, "Random policy"),
    (heuristic_states, heuristic_actions, "Heuristic 'chase pole'"),
]):
    t = np.arange(len(states))
    for row in range(4):
        ax = axes[row, col]
        ax.plot(t, states[:, row], color="#2c7fb8", linewidth=1.5)
        ax.set_ylabel(var_names[row], fontsize=9)
        ax.grid(alpha=0.3)

        # Draw failure-threshold lines for vars that have them
        if limits[row] is not None:
            lo, hi = limits[row]
            ax.axhline(lo, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.axhline(hi, color="red", linestyle="--", linewidth=0.8, alpha=0.6)

        if row == 0:
            ax.set_title(f"{label} — episode lasted {len(t)-1} steps")
        if row == 3:
            ax.set_xlabel("Timestep")

plt.suptitle("CartPole-v1 state trajectories\n(red dashed lines = termination thresholds)",
             y=1.00, fontsize=12)
plt.tight_layout()
plt.savefig("state_trajectory.png", dpi=120)
print("Saved state_trajectory.png")
print("\nWhat to look for:")
print("- Random policy: pole angle drifts past ±0.21 rad → episode ends")
print("- Heuristic policy: pole angle oscillates but stays bounded longer")
print("- The cart position drifts in both cases — the heuristic doesn't")
print("  consider cart position, only pole angle. That's a bug a learned")
print("  policy can fix because it sees the full 4-D state.")