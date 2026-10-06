"""Copy the 36 published solved counts into a COMMITTED table.

The published JSON records are gitignored, so without this Gate 0(b) would depend on files that
exist only on the machines that ran EXP-053 and EXP-062. Run once, from the repo root.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("exp070_cells", HERE / "cells.py")
cells = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cells)


def main() -> None:
    table = {}
    for d in cells.DEPTHS:
        for s in cells.SEEDS:
            rec = json.loads(cells.published_record_path(d, s).read_text())
            n = int(rec["n"])
            table[f"d{d}_s{s}"] = {"solved": round(float(rec["success_rate"]) * n), "n": n}
    (HERE / "published_counts.json").write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
    print(f"wrote {len(table)} cells")


if __name__ == "__main__":
    main()
