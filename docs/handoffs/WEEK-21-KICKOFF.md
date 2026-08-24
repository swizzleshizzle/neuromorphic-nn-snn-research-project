# Week 21 kickoff plan - written 2026-08-24, before EXP-050 landed

**Purpose: when EXP-050 finishes there should be nothing to decide.** Dispatch EXP-051, then
design Stage 3 against the sketch below.

## 0. The queue

| | what | status |
|---|---|---|
| EXP-050 | objective vs more gradient | **running**, phase 1 from 19:02, expect ~01:00 |
| EXP-051 | depth-7 transfer | **pre-registered and staged on the laptop** (`launch051.ps1`) |
| Stage 3 | dense signal on the neuromod bus | **to design**, sketch below |

**Dispatch EXP-051 the moment EXP-050's 12 arm-F records exist.** Its launcher refuses to start
if more than 2 python processes are alive, so it cannot accidentally share the machine.

```bash
ssh -n laptop 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\launch051.ps1'
```

## 1. Read EXP-050 through its grid, then act

Its spec fixes the grid; two of the three plausible branches point at the **pretraining
objective** rather than the RL loop, so the branch matters for Stage 3's priority.

| EXP-050 result | consequence for week 21 |
|---|---|
| **REFUTED** (more pretraining does nothing) | "More gradient" closes. EXP-047/048/049 lift their caveats. **Stage 3 proceeds as the main build.** |
| **just more gradient** (F ~ B) | Deflates the fine-tuning line. A better *pretraining objective* becomes the cheapest lever, and Stage 3 competes with it rather than following it. |
| **F beats B** | The line is superseded by something offline and cheaper. **Reprioritise before building Stage 3.** Note the spec's own caveat: this cell cannot attribute the win to the objective, because pretraining also applies ~100x more parameter movement per update. |

## 2. Stage 3 sketch - a dense signal, and the honesty trap

### The measured problem it targets

**EXP-045 is the reason this stage exists.** At depth 7 the policy *collapsed* under sparse
reward: deepest-stage entropy fell **0.591 -> 0.098** (min 2.7e-06) on a **2.2%** training solve
rate. When the agent almost never solves, REINFORCE gets almost no signal and what it does get is
noise. **The failure is signal density, and it is measured, not assumed.**

### The mechanism, and why it is genuinely neuromorphic rather than a retrofit

A learned value function `V(s)` gives a per-step TD error:

```
delta = r + gamma * V(s') - V(s)
```

**`delta` is the dopamine signal.** Reward-prediction-error-as-dopamine is the actual
neuroscience story, not an analogy invented to justify using the bus. `NeuromodBus` already has
a `dopamine` field and `Brain.learn()` already writes `reward - baseline` into it - a crude
scalar stand-in for exactly this quantity.

### Two increments, and only the second is neuromorphic

**3a - the substance.** Replace the EMA baseline with a learned critic and use `delta` as the
advantage. `V` as `Linear(64 -> 1)` on the same concept the policy head reads: **65 parameters**.
Test at **depth 7 at 10,000 episodes**, where the collapse was measured, against EXP-044 arm A
(0.0621) and now also EXP-051's number. Primary claim on success; **mechanism claim on the
entropy trace**, which is the thing EXP-045 showed dying.

**3b - the neuromorphic step.** Route `delta` onto the bus and **gate encoder plasticity on it**:
`NeuromodBus.learning_enabled` already exists and is unused. The encoder updates only when
`|delta|` clears the threshold - neuromodulated plasticity, and a direct synthesis with week 20's
result. Testable against always-on fine-tuning.

> [!danger] THE TRAP, NAMED IN ADVANCE
> **3a is actor-critic. Putting `delta` on a bus does not make it neuromorphic.** If nothing
> *reads* the bus, the routing is decoration and the honest description of 3a is
> *"a learned critic reduces gradient variance"* - a real result, and an RL one.
>
> **3b is the increment where the bus becomes load-bearing**, because `learning_enabled` would
> actually gate something. Only 3b licenses "a neuromodulatory signal is on the critical path".
>
> `CLAUDE.md` and [[road-to-a-solved-cube]] both record that **nothing neuromorphic participates
> in the learning yet**. Week 20 changed that once - the spiking encoder now trains. Do not let
> 3a be written up as if it changed it twice.

### Design questions to settle before writing the spec

1. **Critic input**: the frozen concept (65 params, matches the policy head) or its own spiking
   region? The cheap version first, on the EXP-025 precedent that a bigger readout did not beat a
   linear one.
2. **Shaping vs critic-as-baseline.** Potential-based shaping (`r + gamma*V(s') - V(s)` added to
   reward) is **policy-invariant** and so cannot change the optimum - a strong property worth
   having. Using `delta` purely as an advantage is the smaller change. **Prefer the smaller
   change first**; the invariance argument is only needed if shaping is added to the reward.
3. **The confound, again.** A critic adds parameters and compute per step. The same discipline
   as EXP-047: measure the per-step cost, price it against the budget curve
   (**0.22/log10 at depth 6, 0.210 at depth 7**), and pre-register the ambiguous band.
4. **What would refute it.** If entropy still collapses with a critic, signal density was not the
   binding constraint and Wall 2 needs a different answer.

## 3. Standing items, unchanged

- **The probe is retired as a policy predictor.** Use `revisit_rate` and `optimality`. Do not add
  a probe number to a Stage 3 spec.
- Redo the probe-based inferences in EXP-033/039/047 with trajectory metrics beside them.
- Re-ask the memory question - EXP-030 was measured on a 2.2% policy and is due a re-run against
  a working one.
- **Depth 8+ is priced, not free**: ~194,000 episodes on the old recipe. Any frontier push should
  wait for EXP-051 to say whether the encoder changes that exchange rate.
