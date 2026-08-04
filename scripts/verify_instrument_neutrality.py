"""Verify a code change did not perturb an experiment it was supposed to only instrument.

The original use (EXP-030, 2026-07-31): the collapse instrument
(`greedy_modal_action_frac`, `mean_train_entropy`) consumes no randomness, so re-running the
identical configuration must reproduce every pre-existing field exactly. If it does not, the
instrument changed the experiment and its numbers cannot be trusted.

EXP-036 needed the same check ACROSS TAGS, which the filename-keyed original could not do.
`record_filename` encodes the tag, so `exp035_curriculum_e10000_*` and `exp036_d3_e10000_*`
never share a filename and the comparison reported "no filenames in common" - a vacuous fail.
Hence `--key-by cell`, which keys on (arm, depth, seed, sigma): the coordinates that actually
identify a run, independent of what the sweep chose to call itself.

Usage:
    # EXP-030 style: same tag, same filenames, new instrument fields
    .venv/bin/python scripts/verify_instrument_neutrality.py ORIGINAL_DIR NEW_DIR

    # EXP-036 style: different tags, matched on run coordinates
    .venv/bin/python scripts/verify_instrument_neutrality.py OLD NEW \\
        --key-by cell \\
        --new-fields train_success_rate,n_train_eval,generalisation_gap \\
        --exempt tag

Exit 0 only if at least one record was compared AND every pre-existing field matched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Fields the EXP-030 collapse instrument added. Overridable with --new-fields.
DEFAULT_NEW_FIELDS = ("greedy_modal_action_frac", "mean_train_entropy")

# `out_dir` necessarily differs: a re-run writes to a separate directory precisely so the
# original records are not overwritten. It is provenance, not a measurement. Always exempt.
ALWAYS_EXEMPT = {"out_dir"}

CELL_KEYS = ("arm", "depth", "seed", "sigma")


def cell_key(record: dict) -> tuple:
    """The coordinates that identify a run, independent of its tag.

    Deliberately NOT the filename. `record_filename` encodes the tag, so two sweeps that ran
    the same cell under different tags produce disjoint filename sets and a filename-keyed
    comparison reports a vacuous "nothing in common" rather than comparing them.
    """
    return tuple(record.get(k) for k in CELL_KEYS)


def strip(record: dict, new_fields: set[str], exempt: set[str]) -> dict:
    """Drop the newly added fields and any key expected to differ, at both levels."""
    drop = new_fields | exempt
    out = {k: v for k, v in record.items() if k not in drop}
    if isinstance(out.get("config"), dict):
        out["config"] = {k: v for k, v in out["config"].items() if k not in drop}
    return out


def diff_keys(a: dict, b: dict) -> list[str]:
    """Names of top-level keys whose values differ, including missing on either side."""
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def load(directory: Path, key_by: str) -> tuple[dict, list[str]]:
    """Map key -> record. Returns (mapping, collisions).

    A collision means two records in one directory share a key, which for `cell` means a
    sweep varied something the key does not capture (`episodes`, `curriculum`, the three
    seeds). Reported rather than silently letting one shadow the other.
    """
    mapping: dict = {}
    collisions: list[str] = []
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text())
        key = path.name if key_by == "filename" else cell_key(record)
        if key in mapping:
            collisions.append(f"{key} ({path.name})")
        mapping[key] = record
    return mapping, collisions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("original_dir", type=Path)
    ap.add_argument("new_dir", type=Path)
    ap.add_argument("--key-by", choices=("filename", "cell"), default="filename",
                    help="match records by filename (default) or by (arm, depth, seed, sigma)")
    ap.add_argument("--new-fields", default=",".join(DEFAULT_NEW_FIELDS),
                    help="comma-separated fields the change ADDS, exempt from comparison")
    ap.add_argument("--exempt", default="",
                    help="comma-separated extra keys expected to differ, e.g. tag")
    args = ap.parse_args()

    new_fields = {f.strip() for f in args.new_fields.split(",") if f.strip()}
    exempt = ALWAYS_EXEMPT | {f.strip() for f in args.exempt.split(",") if f.strip()}

    orig, orig_dupes = load(args.original_dir, args.key_by)
    new, new_dupes = load(args.new_dir, args.key_by)

    print(f"original: {len(orig)} records in {args.original_dir}")
    print(f"new:      {len(new)} records in {args.new_dir}")
    print(f"keyed by: {args.key_by}")
    print(f"exempt:   {sorted(new_fields | exempt)}")

    for label, dupes in (("original", orig_dupes), ("new", new_dupes)):
        if dupes:
            print(f"\nFAIL: {len(dupes)} key collisions in {label}; the key does not identify")
            print("a run here, so some records shadowed others. Widen the key or split the tags.")
            for d in dupes[:5]:
                print(f"  {d}")
            return 1

    only_orig = sorted(set(orig) - set(new), key=str)
    only_new = sorted(set(new) - set(orig), key=str)
    if only_orig:
        print(f"\nMISSING from new ({len(only_orig)}): {only_orig[:5]}")
    if only_new:
        print(f"\nEXTRA in new ({len(only_new)}): {only_new[:5]}")

    shared = sorted(set(orig) & set(new), key=str)
    identical, differing, missing_instrument = 0, [], []

    for key in shared:
        a, b = orig[key], new[key]
        if not new_fields <= set(b):
            missing_instrument.append(str(key))
        if strip(a, new_fields, exempt) == strip(b, new_fields, exempt):
            identical += 1
        else:
            differing.append(
                (str(key), diff_keys(strip(a, new_fields, exempt), strip(b, new_fields, exempt)))
            )

    print(f"\ncompared:  {len(shared)}")
    print(f"identical: {identical}")
    print(f"differing: {len(differing)}")

    # Checked BEFORE the pass path. An empty intersection must never read as success: that is
    # the vacuous pass this script exists to avoid, and the reason --key-by cell was added.
    if not shared:
        print("\nFAIL: no records in common, nothing was actually compared.")
        if args.key_by == "filename":
            print("If the two sweeps used different tags their filenames cannot match.")
            print("Retry with --key-by cell --exempt tag.")
        return 1

    if missing_instrument:
        print(f"\nFAIL: {len(missing_instrument)} new records lack the added fields")
        for key in missing_instrument[:5]:
            print(f"  {key}")

    if differing:
        print("\nFAIL: pre-existing fields changed. The change perturbed the experiment.")
        for key, keys in differing[:10]:
            print(f"  {key}: {keys}")
        return 1

    if missing_instrument:
        return 1

    print("\nPASS: every pre-existing field reproduced exactly. The change is neutral.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
