#!/usr/bin/env bash
# EXP-085 VPS stage, run from the main checkout before the laptop chain (amendment 2026-11-06):
#   1. copy X16k's four checkpoint files per seed from the laptop into outputs/x16k_src/;
#   2. Gate C on the VPS (judge.pt and state.json checked against EXP-082's tracked files);
#   3. the walks pre-flight (distance shares, instrument only);
#   4. force-add and commit the Gate C and pre-flight records, push;
#   5. pull on the laptop, then copy outputs/start/ there. The round trip keeps Gate C's check
#      against git on the VPS, where the spec puts it.
# Every step skips what is already done, so a re-run after a failure is safe.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
EXP=experiments/085_outward_walks
LAPTOP_SRC="C:/Users/mlgbr/Desktop/Projects/neuromorphic-nn-snn-research-project/experiments/082_x_long_budget/outputs"
LAPTOP_REPO='C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project'
LAPTOP_OUT="C:/Users/mlgbr/Desktop/Projects/neuromorphic-nn-snn-research-project/$EXP/outputs"
cd "$REPO"
for s in $(seq 0 11); do
  d="$EXP/outputs/x16k_src/judge_X_s$s"
  mkdir -p "$d"
  for f in judge.pt jt.pt opt.pt state.json; do
    [ -s "$d/$f" ] || scp -q "laptop:$LAPTOP_SRC/judge_X_s$s/$f" "$d/$f"
  done
done
PYTHONPATH=src .venv/bin/python -u "$EXP/gatec.py" --src-root "$EXP/outputs/x16k_src"
[ -s "$EXP/outputs/exp085_preflight_walks.json" ] || \
  PYTHONPATH=src .venv/bin/python -u "$EXP/preflight.py" --part walks
git add -f "$EXP"/outputs/exp085_gatec_s*.json "$EXP/outputs/exp085_preflight_walks.json"
git diff --cached --quiet || git commit -qm "EXP-085 Gate C records (12 seeds) and the walks pre-flight"
git push -q
ssh -n laptop "cd /d $LAPTOP_REPO && git pull -q"
scp -rq "$EXP/outputs/start" "laptop:$LAPTOP_OUT/"
echo "staged: Gate C passed, start checkpoints on the laptop"
