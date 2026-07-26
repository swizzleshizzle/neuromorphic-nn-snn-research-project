"""``Brain`` — the full five-region assembly orchestrator (spec §3, Week-11 S1).

Instantiates all five regions plus the neuromodulatory bus, owns the inter-region
wiring from the architecture spec, and exposes one environment decision as
``step(obs) -> action``. The composition is **window-batched** (the EXP-020 pattern):
each region consumes a full ``[T, B, N]`` spike window and hands its output window to
the next region, so one ``step`` = one environment decision over the ``T``-step
inference window.

Signal flow per ``step`` (matches the session brief and spec §3 pathways):

1. **Environment → Sensory** — `encode_gridworld(obs)` → Poisson spikes → Sensory concept.
2. **Sensory → Hippocampus** (gated, "via thalamus") — store the sensory snapshot
   (pathway 3) and/or read the recall code (the attractor driven by the current cue).
3. **Sensory + Hippocampus → Prefrontal** (gated) — PFC integrates the sensory concept
   and the gated hippocampal recall via its two summed afferents (pathway 2 + 4).
4. **Prefrontal → Motor** — the Thalamic Router gates PFC→Motor (pathway 5); Motor's
   winner-take-all resolves a single action.
5. **Motor → action** — the winning neuron (argmax spike count) is the discrete action.
6. **Reward → learnable connections** — ``learn(reward)`` pushes reward onto the
   dopamine bus (the R-STDP third factor). Actual plasticity is **deferred** (designed
   L11 / EXP-021); ``learn`` is the wired hook, per this session's chosen scope.

v1 simplifications (flagged for later): the store/recall commands on pathways 3/4 are
explicit ``store`` / ``recall`` flags rather than router-issued — the router's Stage-A
selection needs PFC utilities, which need the recall, so issuing store/recall from the
router in the same pass is a chicken-and-egg left for a later closed-loop pass. The
router *is* used for its real job, gating pathway 5 (PFC→Motor).
"""

from __future__ import annotations

import numpy as np
import torch

from neuromorphic.connections import apply_gate
from neuromorphic.encoders import grid_encoder
from neuromorphic.neuromod import NeuromodBus
from neuromorphic.regions import (
    Hippocampus,
    MotorCortex,
    Prefrontal,
    SensoryCortex,
    ThalamicRouter,
)


class Brain:
    """The five-region spiking brain wired into one orchestrator.

    Args:
        grid_n: grid side length (sets ``n_obs = 2 * grid_n**2``).
        content: width of the sensory concept / hippocampal content code.
        n_hippo: hippocampus attractor population size.
        n_actions: number of discrete actions (grid-world = 4).
        num_steps: inference window ``T`` (one env step = one ``T``-window).
        seed: shared RNG seed for reproducible region weights.
        bus: an existing ``NeuromodBus``, or ``None`` to create a fresh one.
    """

    def __init__(
        self,
        grid_n: int = 5,
        content: int = 64,
        n_hippo: int = 150,
        n_actions: int = 4,
        num_steps: int = 32,
        seed: int = 0,
        bus: NeuromodBus | None = None,
        encoder=None,
        n_obs: int | None = None,
        obs_width: int = 4,
    ):
        self.grid_n = grid_n
        self.content = content
        self.n_actions = n_actions
        self.T = num_steps
        self.bus = bus if bus is not None else NeuromodBus()

        # Encoder seam: default reproduces the grid behavior exactly, so every existing
        # caller (Brain(grid_n=5)) is untouched. A cube passes cube_encoder() with
        # n_obs=144 and obs_width=24. See docs/.../2026-07-25-cube-baseline-design.md.
        self.obs_width = obs_width
        self._encoder = encoder if encoder is not None else grid_encoder(grid_n)
        self.n_obs = n_obs if n_obs is not None else 2 * grid_n * grid_n

        self.sensory = SensoryCortex(
            n_obs=self.n_obs, concept=content, num_steps=num_steps, seed=seed
        )
        self.hippo = Hippocampus(
            content_dim=content, n_neurons=n_hippo, num_steps=num_steps, seed=seed
        )
        self.pfc = Prefrontal(
            concept_dim=content,
            recall_dim=content,
            n_actions=n_actions,
            num_steps=num_steps,
            seed=seed,
        )
        self.router = ThalamicRouter(n_actions=n_actions, num_steps=num_steps)
        self.motor = MotorCortex(n_actions=n_actions, num_steps=num_steps, bus=self.bus)

        self._regions = {
            "sensory": self.sensory,
            "hippocampus": self.hippo,
            "prefrontal": self.pfc,
            "router": self.router,
            "motor": self.motor,
        }

    @property
    def n_neurons(self) -> int:
        """Total neurons across all regions. The matching budget for MonolithicBrain."""
        return sum(r.n_neurons for r in self._regions.values())

    # ------------------------------------------------------------------ #
    def _to_obs_tensor(self, obs) -> torch.Tensor:
        """Coerce an observation to a ``[B, obs_width]`` int tensor."""
        arr = np.asarray(obs)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[1] != self.obs_width:
            raise ValueError(
                f"obs must be [{self.obs_width}] or [B, {self.obs_width}]; got {arr.shape}"
            )
        return torch.as_tensor(arr, dtype=torch.long)

    def remember(self, obs, generator: torch.Generator | None = None) -> torch.Tensor:
        """Imprint the current sensory snapshot into the hippocampal attractor.

        Returns the stored sparse pattern ``p`` (``[n_hippo]``).
        """
        obs_t = self._to_obs_tensor(obs)
        spikes = self._encoder(obs_t, T=self.T, generator=generator)
        concept = self.sensory(spikes)
        snapshot = concept.mean(dim=0)  # [B, content] sensory snapshot
        return self.hippo.store(snapshot)

    def step(
        self,
        obs,
        *,
        store: bool = False,
        recall: bool = True,
        record: bool = False,
        generator: torch.Generator | None = None,
    ) -> dict:
        """One environment decision: obs → action, over the ``T``-window.

        Args:
            obs: ``[obs_width]`` or ``[B, obs_width]`` integer observation, where
                ``obs_width`` is set at construction (default 4, the grid's
                ``(ax, ay, gx, gy)``; a cube-configured ``Brain`` uses 24 raw facelets).
            store: imprint the current sensory snapshot into the hippocampus (pathway 3).
            recall: feed the (gated) hippocampal recall into PFC's memory afferent
                (pathway 4). ``False`` → sensory-only PFC (recall=None).
            record: enable per-region state logging; recordings are returned under
                ``"recordings"`` for the viz toolkit.
            generator: RNG for reproducible Poisson encoding.

        Returns:
            dict with ``action`` (int for a single agent, else ``[B]`` tensor),
            ``utilities``/``recall``/``gate``/``action_spikes`` ``[T, B, N]`` tensors,
            ``obs_spikes`` (the encoder's output, ``[T, B, n_obs]``; ``n_obs`` is set at
            construction, e.g. ``2*grid_n**2`` for the default grid encoder or 144 for
            the cube encoder), and ``recordings`` (per-region ``{key: [T,B,N]}``) when
            ``record=True``.
        """
        obs_t = self._to_obs_tensor(obs)
        B = obs_t.shape[0]

        if record:
            for r in self._regions.values():
                r.enable_recording(True)

        # 1. Environment → Sensory Cortex
        obs_spikes = self._encoder(obs_t, T=self.T, generator=generator)
        concept = self.sensory(obs_spikes)  # [T, B, content]

        # 2. Sensory → Hippocampus (gated store / recall — "via thalamus")
        if store:
            self.hippo.store(concept.mean(dim=0))  # one-shot Hebbian imprint of the snapshot
        recall_code = self.hippo(concept) if recall else None  # [T, B, content] or None

        # 3. Sensory + Hippocampus → Prefrontal (two summed afferents)
        utilities = self.pfc(concept, recall_spikes=recall_code)  # [T, B, n_actions]

        # 4. Prefrontal → Motor, gated by the Thalamic Router (pathway 5)
        gate_closed = self.router(utilities)  # [T, B, n_actions]
        action_spikes = self.motor(apply_gate(utilities, gate_closed))  # [T, B, n_actions]

        # 5. Motor → action (winner = argmax spike count over the window)
        winners = self.motor.winner(action_spikes)  # [B]
        action = int(winners[0]) if B == 1 else winners

        out = {
            "action": action,
            "concept": concept,
            "recall": recall_code,
            "utilities": utilities,
            "gate_closed": gate_closed,
            "action_spikes": action_spikes,
            "obs_spikes": obs_spikes,
        }
        if record:
            out["recordings"] = {
                name: r.get_recording() for name, r in self._regions.items()
            }
            for r in self._regions.values():
                r.enable_recording(False)
        return out

    def learn(self, reward: float, baseline: float = 0.0) -> bool:
        """Wire a reward onto the dopamine bus (the R-STDP third factor).

        Sets ``dopamine = reward - baseline`` (reward-minus-expectation). Actual
        plasticity is **deferred** (R-STDP designed in L11; eligibility-trace taste in
        EXP-021) — this is the hook the future weight update reads.

        Returns whether learning would be enabled (``dopamine >= learning_threshold``).
        """
        self.bus.set(dopamine=float(reward - baseline))
        return self.bus.learning_enabled

    def run_episode(
        self,
        env,
        *,
        max_steps: int | None = None,
        store_first: bool = True,
        generator: torch.Generator | None = None,
    ) -> dict:
        """Drive a Gymnasium env to termination/truncation with this brain.

        Returns a summary dict: ``steps``, ``total_reward``, ``reached_goal``,
        ``actions``.
        """
        obs, _ = env.reset()
        if store_first:
            self.remember(obs, generator=generator)

        actions: list[int] = []
        total_reward = 0.0
        reached_goal = False
        steps = 0
        limit = max_steps if max_steps is not None else getattr(env, "max_steps", 100)

        while steps < limit:
            out = self.step(obs, recall=True, generator=generator)
            action = out["action"]
            obs, reward, terminated, truncated, _ = env.step(action)
            self.learn(reward)
            actions.append(int(action))
            total_reward += float(reward)
            steps += 1
            if terminated:
                reached_goal = True
                break
            if truncated:
                break

        return {
            "steps": steps,
            "total_reward": total_reward,
            "reached_goal": reached_goal,
            "actions": actions,
        }


def _demo() -> None:
    """Run one untrained episode end-to-end and print a summary."""
    from neuromorphic.envs import GridWorldEnv

    env = GridWorldEnv()
    brain = Brain(grid_n=env.size, seed=0)
    summary = brain.run_episode(env, generator=torch.Generator().manual_seed(0))
    print("five-region brain — one untrained episode")
    print(f"  steps        : {summary['steps']}")
    print(f"  total reward : {summary['total_reward']:.0f}")
    print(f"  reached goal : {summary['reached_goal']}")
    print(f"  actions      : {summary['actions']}")
    print("(untrained → action is a fixed structural favourite; the point is the loop runs)")


if __name__ == "__main__":
    _demo()
