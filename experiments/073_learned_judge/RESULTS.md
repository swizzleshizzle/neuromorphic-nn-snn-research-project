# EXP-073 results: superseded by EXP-074, no evaluation

EXP-073 (spec `docs/superpowers/specs/2026-10-07-exp073-learned-judge-design.md`) trained a
64-unit learned judge by value iteration. Two pilots (seeds 12 and 13; records in `outputs/` and
`outputs_pilot2/`, reports in spec sections 10 and 11) predicted that its training gate, Gate T,
would fail for both arms. Diagnostics then showed Gate T guarded the wrong quantity and the 64-unit
readout was a measured bottleneck (issue #14). Michael chose to replace it with EXP-074.

**No EXP-073 evaluation seed was ever trained and no EXP-073 claim was tested.** Its arms A and B
were carried into EXP-074 unchanged (EXP-074 Gate 0(c) re-ran them and reproduced pilot 1 exactly),
and EXP-074 answered EXP-073's two questions as secondary patterns at depth 9: the 64-unit judge
beats P3V (+0.066) and training its encoder helps (+0.063).

Results: `experiments/074_wide_judge/RESULTS.md`.

Launcher note: `launch073.ps1`'s evaluation pool has the null-`ExitCode` defect fixed in EXP-074's
launcher (`5ca5ead`). It never ran an evaluation phase.
