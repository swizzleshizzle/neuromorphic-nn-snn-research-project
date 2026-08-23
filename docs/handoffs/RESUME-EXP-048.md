# Resume EXP-048 after the laptop went to sleep

**State as of 2026-08-23:** EXP-048 was dispatched and started cleanly (6 workers up, banner
correct, laptop 23:42:04). The laptop then dropped off the tailnet roughly 20-40 minutes in and
has been `offline` since. The VPS's own network path was verified healthy at the time, so this
is the laptop sleeping, not a network or code fault.

## Nothing is lost

- Windows **suspends** processes rather than killing them.
- Completed cells are already written to `experiments/048_fresh_head/outputs/`.
- Runs are **seeded and byte-identical across worker scheduling**, so a resumed run reproduces
  the same numbers as an uninterrupted one.
- The launcher passes `--skip-existing`, so finished cells are never recomputed.

## To resume: wake the laptop, then run ONE of these

If the original workers survived the sleep, they may simply carry on. Check first:

```bash
ssh -n laptop 'powershell -NoProfile -Command "@(Get-Process | Where-Object { $_.Name -match \"^python\" }).Count; @(Get-ChildItem C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project\experiments\048_fresh_head\outputs\exp048_freshhead_d6_*.json).Count"'
```

- **pyprocs > 2** -> it is still running. Leave it alone.
- **pyprocs 0-2 and records < 12** -> re-dispatch; it resumes from what is on disk:

```bash
ssh -n laptop 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\launch048.ps1'
```

- **records = 12** -> it finished. Fetch and aggregate:

```bash
scp 'laptop:C:/Users/mlgbr/Desktop/Projects/neuromorphic-nn-snn-research-project/experiments/048_fresh_head/outputs/*.json' experiments/048_fresh_head/outputs/
.venv/bin/python experiments/048_fresh_head/aggregate.py
```

## Fix the cause, once

This is the second time sleep has threatened an overnight run (EXP-047's 42 h chain survived by
luck). **Set the laptop to never sleep on AC**, or these dispatches stay a coin flip:

```
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
```

Tailscale **cannot wake a sleeping machine**, so there is no remote recovery path. This is the
one dependency of the whole remote-run workflow that lives entirely outside the VPS.

## What EXP-048 is

See `docs/superpowers/specs/2026-08-23-exp048-fresh-head-design.md` (pre-registered at
`652b517`). Arm B: EXP-047's fine-tuned encoders, **frozen**, with a **fresh** head. Tests
whether EXP-047's +0.0900 was a genuinely better encoder or encoder-head co-adaptation.
Arms A (0.1800, EXP-043) and C (0.2700, EXP-047) already exist and are not re-run.
