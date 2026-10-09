# EXP-074 diagnostics (throwaway, 2026-10-07)

Run on the VPS against EXP-073 pilot judges copied to `/root/scratch/exp073pilot/judge_*`. Kept for
provenance of spec sections 1, 2 and 6; none of these is part of the method.

| script | what it measured | result |
|---|---|---|
| `noise.py` | per-state Poisson sd of pilot J vs between-state sd | 0.19-0.25 vs 0.68-0.83 |
| `local.py` | lowest-J child closer, 50 states per distance | exploratory, n too small |
| `ceiling.py N which` | heads fitted on TRUE distance over concept / concept+hidden / raw | spearman_7_11 0.26 / 0.34-0.37 / 0.39-0.40 at 500k |
| `oracle_search.py` | raw judge fitted on true distance, eval states excluded, through J3V | d9 s0 88/200 vs P3V 5/200 |
| `pilot_search.py` | EXP-073 pilot judges through J3V on evaluation cells (DISCLOSED, spec section 2) | see spec |
| `gate_l.py` | Gate L margin on pilot judges, pilot seeds only | A 0.148/0.143, B 0.104/0.043 |
| `oracle_wide.py` | heads fitted on TRUE distance over frozen readouts, through J3V at d9 s0 (already disclosed), 2026-10-09, EXP-075 motivation | raw 88/200; E1 192 33; E1 192 x8 draws 34; random 128 30; random 512 44 (spec EXP-075 section 1) |
