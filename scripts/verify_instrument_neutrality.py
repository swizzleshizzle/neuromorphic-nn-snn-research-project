"""Verify the collapse instrument did not perturb EXP-030.

The instrument (`greedy_modal_action_frac`, `mean_train_entropy`) consumes no randomness,
so re-running the identical configuration must reproduce every pre-existing field exactly.
If it does not, the instrument changed the experiment and its numbers cannot be trusted.

This is the check the playbook calls free: EXP-030's 36 concept records were already shown
byte-identical across two independent invocations at different worker counts, so any
difference here is attributable to the code change rather than to scheduling.

Usage:
    .venv/bin/python scripts/verify_instrument_neutrality.py ORIGINAL_DIR NEW_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Fields the instrument adds. Everything else must match exactly.
NEW_FIELDS = {"greedy_modal_action_frac", "mean_train_entropy"}

# `out_dir` necessarily differs: the re-run wrote to a separate directory precisely so the
# original records were not overwritten. It is provenance, not a measurement.
CONFIG_EXEMPT = {"out_dir"}


def strip(record: dict) -> dict:
    """Drop the new fields and the one config key that is expected to differ."""
    out = {k: v for k, v in record.items() if k not in NEW_FIELDS}
    if isinstance(out.get("config"), dict):
        out["config"] = {k: v for k, v in out["config"].items() if k not in CONFIG_EXEMPT}
    return out


def diff_keys(a: dict, b: dict) -> list[str]:
    """Names of top-level keys whose values differ, including missing on either side."""
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2

    orig_dir, new_dir = Path(sys.argv[1]), Path(sys.argv[2])
    orig = {p.name: p for p in orig_dir.glob("*.json")}
    new = {p.name: p for p in new_dir.glob("*.json")}

    print(f"original: {len(orig)} records in {orig_dir}")
    print(f"new:      {len(new)} records in {new_dir}")

    only_orig = sorted(set(orig) - set(new))
    only_new = sorted(set(new) - set(orig))
    if only_orig:
        print(f"\nMISSING from new ({len(only_orig)}): {only_orig[:5]}")
    if only_new:
        print(f"\nEXTRA in new ({len(only_new)}): {only_new[:5]}")

    shared = sorted(set(orig) & set(new))
    identical, differing, missing_instrument = 0, [], []

    for name in shared:
        a = json.loads(orig[name].read_text())
        b = json.loads(new[name].read_text())

        if not NEW_FIELDS <= set(b):
            missing_instrument.append(name)

        if strip(a) == strip(b):
            identical += 1
        else:
            differing.append((name, diff_keys(strip(a), strip(b))))

    print(f"\ncompared:  {len(shared)}")
    print(f"identical: {identical}")
    print(f"differing: {len(differing)}")

    if missing_instrument:
        print(f"\nFAIL: {len(missing_instrument)} new records lack the instrument fields")
        for name in missing_instrument[:5]:
            print(f"  {name}")

    if differing:
        print("\nFAIL: pre-existing fields changed. The instrument perturbed the experiment.")
        for name, keys in differing[:10]:
            print(f"  {name}: {keys}")
        return 1

    if missing_instrument:
        return 1

    if not shared:
        print("\nFAIL: no filenames in common, nothing was actually compared")
        return 1

    print("\nPASS: every pre-existing field reproduced exactly. Instrument is neutral.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
