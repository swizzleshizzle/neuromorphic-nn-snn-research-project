"""
Script 1: Live rendering — see CartPole with your eyeballs

Uses render_mode="human" to pop up a window showing CartPole live.
Compares two policies:
  1. Random policy (the baseline)
  2. A simple hand-crafted heuristic: "lean toward the pole's tilt direction"

The heuristic isn't optimal but illustrates that even a 5-line rule can
massively outperform random — and gives you a feel for why pole balancing
is a control problem before it's a learning problem.

Run:  python3 01_live_render_cartpole.py
Press Ctrl+C in the terminal to quit early.
"""
import gymnasium as gym
import time


def random_policy(obs):
    """Pick uniformly at random from {0, 1}."""
    return env.action_space.sample()


def heuristic_policy(obs):
    """
    obs = [cart_pos, cart_vel, pole_angle, pole_angular_vel]

    Rule: if the pole is tilting right (positive angle), push the cart right.
    If tilting left (negative angle), push left. The cart 'chases' the pole's
    base under it. Crude but effective — usually gets 40-200 steps.
    """
    pole_angle = obs[2]
    return 1 if pole_angle > 0 else 0  # 1 = push right, 0 = push left

def better_heuristic(obs):
    """
    obs = [cart_pos, cart_vel, pole_angle, pole_angular_vel]

    Weighted sum of all 4 state variables. Push right when the
    weighted score is positive, left when negative. Hand-tuned
    weights that emphasize pole angle + angular velocity (the
    important variables) and gently corrects for cart drift.
    """
    cart_pos, cart_vel, pole_angle, pole_ang_vel = obs
    score = (0.1 * cart_pos
             + 1.0 * cart_vel
             + 10.0 * pole_angle
             + 5.0 * pole_ang_vel)
    return 1 if score > 0 else 0


def run_episode(env, policy, label, slowdown=0.02):
    """Run one episode and print the return."""
    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0
    while True:
        action = policy(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        time.sleep(slowdown)  # slow it down so you can watch
        if terminated or truncated:
            break
    print(f"  [{label}] survived {steps} steps, return = {total_reward}")
    return total_reward


# render_mode="human" pops up a window. Keep one env open the whole time.
env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=2000)

print("=" * 60)
print("CartPole Live — two policies, three episodes each")
print("=" * 60)

print("\nRandom policy (baseline):")
for ep in range(3):
    run_episode(env, random_policy, f"random ep{ep+1}")

print("\nHeuristic policy ('chase the pole'):")
for ep in range(3):
    run_episode(env, heuristic_policy, f"heuristic ep{ep+1}")
    
print("\nBetter Heuristic policy:")
for ep in range(10):
    run_episode(env, better_heuristic, f"heuristic ep{ep+1}")

env.close()
print("\nDone. Notice how the heuristic isn't *optimal* — it overshoots and")
print("oscillates. That's the gap a learned policy will close.")