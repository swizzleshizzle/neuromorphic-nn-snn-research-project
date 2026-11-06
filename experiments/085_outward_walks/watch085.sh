#!/usr/bin/env bash
# EXP-085 watcher (VPS). Polls the laptop every $WATCH085_SLEEP s (default 600) for up to
# $WATCH085_ROUNDS rounds (default 12). Since the 2026-11-06 amendment the run is on the laptop
# and the VPS runs this from a systemd timer, one round per run, so every count lives in
# $WATCH085_STATE:
#   - verdict file present: push once, exit 0. A STOP line other than a failed stage: push once,
#     exit 3 (the chain stopped on purpose, e.g. a failed check; read it). Either writes
#     $WATCH085_STATE/done, and a later run exits with its code and pushes nothing.
#   - empty status (ssh failed): the laptop is UNREACHABLE (asleep, rebooting, off the network).
#     Never relaunch on it and never count it as a crash; on the third unreachable poll in a
#     row, push once so Michael can wake it.
#   - reachable with no chain process and no python (dead, or stopped on a failed stage): a
#     reboot when the boot time changed since the last poll (Windows Update, most nights around
#     03:31), else a crash. Either way relaunch from the top (every stage resumes). Only crashes
#     count; on the third: push, exit 4, no relaunch.
#   - rounds used up while running: exit 5.
set -u
SLEEP=${WATCH085_SLEEP:-600}
ROUNDS=${WATCH085_ROUNDS:-12}
STATE=${WATCH085_STATE:-/root/scratch/exp085}
HOST=laptop
DIR='C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project\experiments\085_outward_walks'
STATUS="powershell -NoProfile -ExecutionPolicy Bypass -File $DIR\\status085.ps1"
LAUNCH="powershell -NoProfile -ExecutionPolicy Bypass -File $DIR\\launch085.ps1"
mkdir -p "$STATE"
push() {
  if [ -n "${WATCH085_PUSH:-}" ]; then "$WATCH085_PUSH" "$1" "$2"; return; fi
  local cfg=/root/.claude/secretary/pm-config.json server topic
  read -r server topic < <(python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); print(c.get("ntfy_server","https://ntfy.sh"), c["ntfy_topic"])' "$cfg")
  curl -sf -m 15 -H "Title: $1" --data-binary "$2" "$server/$topic" >/dev/null || echo "push FAILED: $1"
}
finish() {  # code, push title, push body
  echo "$1" > "$STATE/done"
  push "$2" "$3"
  exit "$1"
}
if [ -s "$STATE/done" ]; then
  code=$(cat "$STATE/done"); echo "already finished (exit $code)"; exit "$code"
fi
crashes=$(cat "$STATE/crashes" 2>/dev/null || echo 0)
unreach=$(cat "$STATE/unreach" 2>/dev/null || echo 0)
for i in $(seq 1 "$ROUNDS"); do
  s=$(ssh -n -o ConnectTimeout=20 -o BatchMode=yes "$HOST" "$STATUS" 2>/dev/null | tr -d '\r')
  echo "$(date -u '+%F %T') ${s:-UNREACHABLE} (crashes $crashes)"
  if [ -z "$s" ]; then
    unreach=$((unreach + 1)); echo "$unreach" > "$STATE/unreach"
    [ "$unreach" -eq 3 ] && push "EXP-085: laptop unreachable" "3 polls in a row with no answer. If it is asleep or off the network, wake it; the run resumes where it stopped."
  else
    unreach=0; echo 0 > "$STATE/unreach"
    case "$s" in True\|*) echo "VERDICT FILE PRESENT"; finish 0 "EXP-085: verdict ready" "$s";; esac
    boot=$(grep -o 'boot [^|]*' <<< "$s")
    last=$(cat "$STATE/boot" 2>/dev/null || true)
    [ -n "$boot" ] && echo "$boot" > "$STATE/boot"
    dead=0
    case "$s" in *"|py 0|ch 0|"*) dead=1;; esac
    # A stage that exited non-zero (a python crash) is a crash; any other STOP is deliberate.
    case "$s" in *"STOP: stage "*" failed"*) ;; *STOP*) echo "CHAIN STOPPED"; finish 3 "EXP-085: chain stopped" "$s";; esac
    case "$dead" in 1)
      if [ -n "$last" ] && [ -n "$boot" ] && [ "$boot" != "$last" ]; then
        echo "REBOOTED ($last -> $boot), not a crash"
      else
        crashes=$((crashes + 1)); echo "$crashes" > "$STATE/crashes"
        if [ "$crashes" -ge 3 ]; then
          echo "THIRD CRASH"; finish 4 "EXP-085: third crash, not relaunching" "$s"
        fi
      fi
      ssh -n -o ConnectTimeout=20 -o BatchMode=yes "$HOST" "$LAUNCH" 2>&1 | tr -d '\r'
      echo "RELAUNCHED (crashes $crashes)";;
    esac
  fi
  [ "$i" -lt "$ROUNDS" ] && sleep "$SLEEP"
done
echo "ROUNDS DONE, still running"; exit 5
