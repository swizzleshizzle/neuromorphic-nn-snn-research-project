"""Pre-flight probe for EXP-064, run by `launch064_wt.ps1` from the worktree.

This is a FILE and not a `python -c` one-liner on purpose. The one-liner version had to embed an
f-string with braces, pipes and nested quotes inside a PowerShell double-quoted argument, and
PowerShell parsed the `|` as a pipeline before python ever saw it. The project's own playbook
already says this: for anything non-trivial, put it in a file and run the file.

Prints three space-separated numbers: trainable_params, region_drift, mean_n_stored.
The launcher checks all three, because each catches a different way a stale worktree can produce
a complete, plausible, entirely wrong result:

  trainable_params  the regions are in the optimizer at all
  region_drift      they ACTUALLY MOVED (EXP-047 reported a 70x surface while its parameter
                    moved by exactly 0.0, and the run looked perfectly ordinary)
  mean_n_stored     the hippocampus is OFF, since memory hurts on this task
"""

from __future__ import annotations

import sys
from pathlib import Path

from neuromorphic.training.cube_baseline import CubeConfig, run_cube_baseline


def main() -> None:
    out = Path(sys.argv[1])
    r = run_cube_baseline(CubeConfig(
        arm="regionalized", readout="motor", region_lr=0.01, tag="launchcheck",
        depth=1, seed=3, sigma=0.0, episodes=6, entropy_beta=0.0, max_depth=1, out_dir=out,
    ))
    drift = r["region_drift"] or 0.0
    sys.stdout.write(f"{r['trainable_params']} {drift:.6f} {r['mean_n_stored']}")


if __name__ == "__main__":
    main()
