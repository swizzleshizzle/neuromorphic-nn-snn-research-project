# The five retired instruments

**Standing note, not an experiment.** No data was generated for it. It collects a pattern that has
now recurred five times across eight experiments, each time discovered separately and each time
costing something.

> **THE RULE: measure the policy's own behaviour. Use `revisit_rate` and `optimality`.**
> None of the five below may gate a decision, cancel an arm, select a pilot, or appear as a claim
> in a new spec.

## The five

| instrument | what it measures | what retired it |
|---|---|---|
| **the EXP-033 probe** | linear decodability of the concept vector | EXP-050: two objectives moved it in **opposite directions, both unanimous at p 0.0005** |
| **pretraining move-accuracy** | how often the pretrained encoder predicts the right move | EXP-052: climbs **0.383 -> 0.452** across 10 to 80 epochs **while the depth-6 policy degrades** |
| **the entropy trace** | per-episode policy entropy during training | EXP-032: an entropy bonus raises it by **injecting randomness**, not by teaching the policy to read its input |
| **`S`** | distance structure in encoder space | EXP-055: at one epoch `S` falls **below its own initialisation** (11 of 12 seeds, p 0.0010) while policy rises from **0.0000 to 0.0854** (12 of 12) |
| **`critic_ev`** | how well the critic predicts pooled stage-level returns | EXP-056: the **worse-performing arm had the better-fitting critic at every stage** |

## What they have in common, and it is not "they are bad measurements"

**Every one of them works as a THRESHOLD and fails as a GRADIENT.** They reliably detect that an
intervention happened at all. None of them measures how much it helped, and past the threshold
several of them point the wrong way.

EXP-054 said it first and most exactly, about `S`: *"a threshold, not a gradient - it detects that
pretraining happened at all"*. The probe re-analysis reached the identical shape from different
records: the probe **got the direction right** - pretraining genuinely raised success at depths 4, 5
and 6 - and across **15 correlations between a probe movement and a behavioural one, 2 were
nominally significant, 0 survived Bonferroni, and both nominal hits ran the wrong way.**

So the recurring error is not trusting a bad number. It is observing that an instrument and the
outcome move together across **one coarse contrast** (pretrained versus not, critic versus no
critic) and then reading that as if the relationship held **per seed** or **per increment**. It does
not. The coarse contrast is the only place the relationship was ever measured.

### Unanimity is the trap that makes this feel safe

**"Unanimous at p 0.0005" measures the consistency of the INSTRUMENT, not its connection to the
outcome.** The probe was unanimous at every depth in EXP-039 and still ranked the seeds no better
than chance. A perfectly reproducible measurement of the wrong quantity is perfectly reproducible.

That is worth stating because unanimity is exactly what makes a proxy persuasive, and four of the
five were adopted on the strength of it.

### Why this system is unusually good at generating them

**`recall=False` means only the sensory region is on the policy path** - the policy head reads the
sensory concept, computed upstream of the hippocampus, so 318 of the five-region brain's 510 neurons
are off-path. An instrument reading a representation can therefore be measuring something the policy
never consults. The architecture makes intermediate measurements cheap to take and easy to
over-read, which is the whole failure mode in one sentence.

`critic_ev` is the same story in the training objective rather than the architecture: **a baseline in
REINFORCE is unbiased for any function of state.** Its job is variance reduction across the timesteps
of an episode, not accurate prediction of the pooled stage-level return distribution - and
`critic_ev` measures the latter. It was never a measurement of the thing that helps.

## What to use instead

**`revisit_rate` and `optimality`.** Both are computed from the policy's own trajectories, so there
is no gap between what is measured and what is claimed. `revisit_rate` has the additional property
that it is a per-episode rate rather than a success count, so its within-arm spread is smaller and it
is better powered than success at the same n - which is why EXP-059 makes it a claim rather than a
footnote.

## Before adopting a new instrument

1. **Ask what it would take for this number to move while the policy got worse.** If you can
   construct that story, the instrument is a threshold detector. All five here fail this.
2. **Validate per seed, not per arm.** A between-arm correlation over 3 or 4 points is one
   observation wearing a hat. The probe re-analysis is the worked example of how to do it: rank the
   seeds by the instrument, rank them by behaviour, and correlate.
3. **Do not let it gate anything until step 2 passes.** EXP-053's lr pilot predicted arm B would be
   null on `critic_ev` grounds. Arm B turned out to be the entire effect. **A pilot that selects on
   an unvalidated proxy can cancel the arm that was going to work.**
4. **Prefer a mechanism measurement on the policy path** to a representation measurement upstream
   of it.

## What is NOT claimed

- **Not that these numbers are meaningless.** Each remains valid for the narrow question it was
  built for. EXP-033's width finding stands; it must simply be quoted narrowly, because it cannot be
  checked against behaviour without new runs.
- **Not that the interventions they motivated were wrong.** Pretraining genuinely works. A learned
  critic genuinely raises depth-7 success. **The instruments failed to explain WHY, which is a
  different failure from the intervention failing.**
- **Not that `critic_ev`'s retirement rests on a controlled comparison.** EXP-056's arms `F` and `B`
  have different policies, so their critics see different trajectories and the two rows are not
  matched. The direction is notable and is the opposite of what "fit the critic better, learn
  better" predicts, but that caveat travels with the finding.
- **Not that `revisit_rate` and `optimality` are exempt from this note.** They are on the policy
  path, which is the reason to prefer them, not a proof. Hold them to the four checks above.

## Sources

`experiments/probe_reanalysis/RESULTS.md` (why unanimity is not relevance) ·
`experiments/050_objective_vs_gradient/RESULTS.md` (the probe's direction failure) ·
`experiments/052_pretraining_optimum/RESULTS.md` (move-accuracy against policy) ·
`experiments/032_collapse_sweep/RESULTS.md` (entropy by randomisation) ·
`experiments/054_sequence_blindness/RESULTS.md` (`S` as a threshold) ·
`experiments/055_pretraining_left_edge/RESULTS.md` (`S` below its own initialisation) ·
`experiments/056_flattened_critic/RESULTS.md` (`critic_ev` inverted)
