"""
Tabular Q-Learning on a 5x5 Grid World
Phase 0, Module L4 — Neuromorphic Project

Goal: Agent learns shortest path from start (0,0) to goal (4,4).
Actions: 0=up, 1=right, 2=down, 3=left
Reward: -1 per step, +10 at goal (encourages short paths)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


# ============================================================
# 1. ENVIRONMENT  (provided — boilerplate)
# ============================================================
class GridWorld:
    def __init__(self, size: int = 5,
                 start: Tuple[int, int] = (0, 0),
                 goal: Tuple[int, int] = (4, 4)):
        self.size = size
        self.start = start
        self.goal = goal
        self.state = start

    def reset(self) -> Tuple[int, int]:
        self.state = self.start
        return self.state

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """Return (next_state, reward, done)."""
        r, c = self.state
        if   action == 0: r -= 1   # up
        elif action == 1: c += 1   # right
        elif action == 2: r += 1   # down
        elif action == 3: c -= 1   # left

        # clamp to grid (bumping a wall = stay in place)
        r = max(0, min(self.size - 1, r))
        c = max(0, min(self.size - 1, c))
        self.state = (r, c)

        done = (self.state == self.goal)
        reward = 10.0 if done else -1.0
        return self.state, reward, done


# ============================================================
# 2. Q-TABLE INITIALIZATION  (provided)
# ============================================================
def make_q_table(size: int, n_actions: int = 4) -> np.ndarray:
    """Q-table shape: (rows, cols, actions). All zeros to start."""
    return np.zeros((size, size, n_actions))


# ============================================================
# 3. EPSILON-GREEDY ACTION SELECTION   <-- TODO (you)
# ============================================================
def select_action(Q: np.ndarray,
                  state: Tuple[int, int],
                  epsilon: float,
                  n_actions: int = 4) -> int:
    """
    With probability epsilon: pick a random action (explore).
    Otherwise: pick the action with the highest Q-value at this state (exploit).

    Hints:
      - np.random.rand() returns a uniform [0,1) sample.
      - np.random.randint(n_actions) returns a random action.
      - np.argmax(Q[state]) returns the greedy action.
        Note: Q[state] indexes with a tuple — Q[(r,c)] gives a length-4 vector.
    """
    # TODO (you): write the 3-4 lines that implement epsilon-greedy
    if np.random.rand() < epsilon:
        #explore
        return np.random.randint(n_actions)
    else:
        #exploit
        return int(np.argmax(Q[state]))
        


# ============================================================
# 4. Q-LEARNING UPDATE   <-- TODO (you) — THE CORE
# ============================================================
def q_update(Q: np.ndarray,
             state: Tuple[int, int],
             action: int,
             reward: float,
             next_state: Tuple[int, int],
             alpha: float,
             gamma: float,
             done: bool) -> None:
    """
    Apply the Q-learning update rule IN-PLACE on Q:

        Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

    Special case: if `done` is True, there is no s' to bootstrap from,
    so the target is just `reward` (no gamma * max term).

    Hints:
      - Q[state][action]  reads the current estimate.
      - np.max(Q[next_state])  gives max over actions at s'.
      - You don't return anything — modify Q in place.
    """
    # TODO (you): implement the update
    #Step 1 - whats the current estimate?
    current = Q[state][action]
    
    #Step 2 - whats the target?
    if done:
        #terminal - nothing comes after, target is just reward
        target = reward
    else:
        #bootstrap - reward + gamma * (max future Q-value)
        target = reward + gamma * np.max(Q[next_state])
    
    #Step 3 - nudge Q toward target by step size alpha
    Q[state][action] = current + alpha * (target - current)
    


# ============================================================
# 5. TRAINING LOOP   (provided — calls your two functions)
# ============================================================
def train(env: GridWorld,
          episodes: int = 1000,
          alpha: float = 0.1,
          gamma: float = 0.95,
          epsilon_start: float = 1.0,
          epsilon_end: float = 0.05,
          max_steps: int = 200) -> Tuple[np.ndarray, list, list]:
    """
    Returns: (Q-table, episode_lengths, episode_rewards)
    """
    Q = make_q_table(env.size)
    lengths, rewards = [], []

    # linearly decay epsilon from start -> end across all episodes
    eps_schedule = np.linspace(epsilon_start, epsilon_end, episodes)

    for ep in range(episodes):
        state = env.reset()
        eps = eps_schedule[ep]
        total_reward, steps = 0.0, 0

        for _ in range(max_steps):
            action = select_action(Q, state, eps)
            next_state, reward, done = env.step(action)
            q_update(Q, state, action, reward, next_state, alpha, gamma, done)

            state = next_state
            total_reward += reward
            steps += 1
            if done:
                break

        lengths.append(steps)
        rewards.append(total_reward)

    return Q, lengths, rewards


# ============================================================
# 6. VISUALIZATION   (provided)
# ============================================================
def plot_results(Q: np.ndarray, lengths: list, rewards: list,
                 title_suffix: str = "") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # (a) Heatmap of V(s) = max_a Q(s,a) — "how good is each cell?"
    V = np.max(Q, axis=2)
    im = axes[0].imshow(V, cmap='viridis')
    axes[0].set_title(f'State value V(s) = max_a Q(s,a) {title_suffix}')
    plt.colorbar(im, ax=axes[0])

    # arrows for greedy policy
    best_actions = np.argmax(Q, axis=2)
    arrow_map = {0: (0, -0.3), 1: (0.3, 0), 2: (0, 0.3), 3: (-0.3, 0)}
    for r in range(Q.shape[0]):
        for c in range(Q.shape[1]):
            if (r, c) == (4, 4):
                axes[0].text(c, r, 'G', ha='center', va='center',
                             color='white', fontsize=14, fontweight='bold')
                continue
            dx, dy = arrow_map[best_actions[r, c]]
            axes[0].arrow(c, r, dx, dy, head_width=0.15,
                          color='white', alpha=0.85)

    # (b) Episode lengths (smoothed)
    axes[1].plot(lengths, alpha=0.3, label='raw')
    if len(lengths) >= 50:
        smooth = np.convolve(lengths, np.ones(50)/50, mode='valid')
        axes[1].plot(smooth, label='smoothed (50 ep)')
    axes[1].set_xlabel('Episode'); axes[1].set_ylabel('Steps to goal')
    axes[1].set_title('Episode length over training')
    axes[1].legend()

    # (c) Episode rewards (smoothed)
    axes[2].plot(rewards, alpha=0.3, label='raw')
    if len(rewards) >= 50:
        smooth = np.convolve(rewards, np.ones(50)/50, mode='valid')
        axes[2].plot(smooth, label='smoothed (50 ep)')
    axes[2].set_xlabel('Episode'); axes[2].set_ylabel('Total reward')
    axes[2].set_title('Episode reward over training')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f'qlearn_results{title_suffix.replace(" ", "_")}.png', dpi=110)
    plt.show()


# ============================================================
# 7. MAIN
# ============================================================
if __name__ == "__main__":
    env = GridWorld(size=5, start=(0, 0), goal=(4, 4))
    Q, lengths, rewards = train(env, episodes=1000)

    print(f"First 10 episodes avg length:  {np.mean(lengths[:10]):.1f}")
    print(f"Last  10 episodes avg length:  {np.mean(lengths[-10:]):.1f}")
    print(f"Optimal length (Manhattan):    8")

    plot_results(Q, lengths, rewards)