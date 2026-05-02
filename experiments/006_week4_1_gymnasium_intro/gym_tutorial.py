"""
Module L4: Hands-on with Gymnasium
Walks through the API one piece at a time, mapping each piece back to
Silver's lecture vocabulary (agent, environment, state, action, reward).
"""
import gymnasium as gym
import numpy as np


def section(title):
    print(f"\n{'=' * 60}\n {title}\n{'=' * 60}")


# ---------------------------------------------------------------
# 1. The MDP loop, made concrete
# ---------------------------------------------------------------
section("1. CartPole — what the MDP actually looks like")

env = gym.make("CartPole-v1")

# In Silver's notation:
#   S = state space  -> env.observation_space
#   A = action space -> env.action_space
#   P = transition dynamics, R = reward -> hidden inside env.step()
print(f"State space S:  {env.observation_space}")
print(f"  - 4 floats: cart position, cart velocity, pole angle, pole angular velocity")
print(f"Action space A: {env.action_space}")
print(f"  - Discrete(2): 0 = push cart left, 1 = push cart right")

# Initial state s_0 ~ initial state distribution
obs, info = env.reset(seed=42)
print(f"\nInitial state s_0 = {obs}")
print(f"info dict (diagnostic, NOT for the agent): {info}")

# ---------------------------------------------------------------
# 2. One step of the loop
# ---------------------------------------------------------------
section("2. One step: a_t -> (s_{t+1}, r_t, done)")

action = 1  # push right
obs, reward, terminated, truncated, info = env.step(action)
print(f"action a_0 = {action}")
print(f"next state s_1 = {obs}")
print(f"reward r_0 = {reward}")
print(f"terminated = {terminated}  (did the pole fall / cart leave bounds?)")
print(f"truncated  = {truncated}   (did we hit the time limit?)")
print(f"\nNote: Gymnasium splits the old `done` into two flags. Silver's MDPs only")
print(f"have `terminated` (true terminal state). `truncated` means 'we cut the")
print(f"episode short for practical reasons' — value bootstrapping should still happen.")

# ---------------------------------------------------------------
# 3. Full episode with a random policy (the baseline of all baselines)
# ---------------------------------------------------------------
section("3. A full episode under a random policy pi(a|s) = uniform")

obs, info = env.reset(seed=0)
total_reward = 0.0
steps = 0
while True:
    action = env.action_space.sample()  # uniform random policy
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1
    if terminated or truncated:
        break

print(f"Episode length: {steps} steps")
print(f"Return G_0 = sum of rewards = {total_reward}")
print(f"(Random policy averages ~20 steps. Solved threshold = 475.)")

# ---------------------------------------------------------------
# 4. Many episodes — what the *return distribution* looks like
# ---------------------------------------------------------------
section("4. 100 episodes of random policy — return distribution")

returns = []
for ep in range(100):
    obs, _ = env.reset(seed=ep)
    G = 0.0
    while True:
        a = env.action_space.sample()
        obs, r, term, trunc, _ = env.step(a)
        G += r
        if term or trunc:
            break
    returns.append(G)

returns = np.array(returns)
print(f"Mean return:    {returns.mean():.1f}")
print(f"Std deviation:  {returns.std():.1f}")
print(f"Min / Max:      {returns.min():.0f} / {returns.max():.0f}")
print(f"\nThis distribution is your baseline. Any learned policy you build")
print(f"must beat ~22 mean return to demonstrate it learned anything at all.")

env.close()

# ---------------------------------------------------------------
# 5. FrozenLake — discrete state space, where tabular RL lives
# ---------------------------------------------------------------
section("5. FrozenLake — discrete states, the world Silver mostly lectures about")

env = gym.make("FrozenLake-v1", is_slippery=False, map_name="4x4")
# is_slippery=False makes it deterministic, easier to reason about first.

print(f"State space:  {env.observation_space}  (16 cells in a 4x4 grid)")
print(f"Action space: {env.action_space}        (0=Left, 1=Down, 2=Right, 3=Up)")
print(f"Reward:       +1 only on reaching the goal, else 0")
print(f"This is the canonical tabular MDP — small enough that you can")
print(f"compute V*(s) and Q*(s,a) exactly via value iteration.")

obs, _ = env.reset(seed=0)
print(f"\nStart state: {obs}  (cell 0, top-left)")

# Solve it manually with the optimal action sequence on the 4x4 deterministic map
# Layout:  S F F F
#          F H F H
#          F F F H
#          H F F G
# Optimal: Down, Down, Right, Right, Down, Right
optimal = [1, 1, 2, 2, 1, 2]
for t, a in enumerate(optimal):
    obs, r, term, trunc, _ = env.step(a)
    print(f"  t={t}: action={a}, next_state={obs}, reward={r}, done={term or trunc}")
    if term or trunc:
        break

env.close()
print(f"\nReached the goal with the hand-crafted optimal policy.")
print(f"Next step in the curriculum: have an algorithm *find* this policy.")