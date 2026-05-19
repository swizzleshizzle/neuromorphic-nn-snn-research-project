"""Run all three section scripts for the week-7 recurrent-SNN experiment.

Each section's ``main()`` writes a plot under ``outputs/`` and prints its
own diagnostic numbers. This driver just sequences them.

Usage:
    python run_all.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import section2_v_sweep                 # noqa: E402
import section2b_10neuron_population    # noqa: E402
import section3_ff_vs_recurrent         # noqa: E402
import section4_rsynaptic_check         # noqa: E402


SECTIONS = [
    ("Section 2 — V sweep",                  section2_v_sweep.main),
    ("Section 2b — 10-neuron population",    section2b_10neuron_population.main),
    ("Section 3 — feedforward vs recurrent", section3_ff_vs_recurrent.main),
    ("Section 4 — RSynaptic API check",      section4_rsynaptic_check.main),
]


def main() -> None:
    for title, fn in SECTIONS:
        print("\n" + "=" * 72)
        print(title)
        print("=" * 72)
        fn()
    print("\nAll sections complete. Plots in outputs/.")


if __name__ == "__main__":
    main()
