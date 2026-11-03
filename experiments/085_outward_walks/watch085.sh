#!/usr/bin/env bash
# EXP-085 watcher (VPS). Polls the desktop every $WATCH085_SLEEP s (default 600) for up to
# $WATCH085_ROUNDS rounds (default 12).
#   - verdict file present: exit 0. A STOP line other than a failed stage: exit 3 (the chain
#     stopped on purpose, e.g. a failed check; read it).
#   - empty status (ssh failed): the desktop is UNREACHABLE, which after a bugcheck can last
#     hours. Never relaunch on it and never count it as a crash; after 3 unreachable polls in a
#     row, push once so Michael can bring Tailscale back.
#   - reachable with no chain process and no python (dead, or stopped on a failed stage): a
#     crash. Relaunch from the top (every stage resumes); the count lives in
#     $WATCH085_STATE/crashes. On the third crash: exit 4, no relaunch.
#   - rounds used up while running: exit 5 (re-arm).
set -u
SLEEP=${WATCH085_SLEEP:-600}
ROUNDS=${WATCH085_ROUNDS:-12}
STATE=${WATCH085_STATE:-/root/scratch/exp085}
STATUS='powershell -NoProfile -ExecutionPolicy Bypass -File D:\gpu083\repo\experiments\085_outward_walks\status085.ps1'
LAUNCH='powershell -NoProfile -ExecutionPolicy Bypass -File D:\gpu083\repo\experiments\085_outward_walks\launch085.ps1'
mkdir -p "$STATE"
push() {
  if [ -n "${WATCH085_PUSH:-}" ]; then "$WATCH085_PUSH" "$1" "$2"; return; fi
  local cfg=/root/.claude/secretary/pm-config.json server topic
  read -r server topic < <(python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); print(c.get("ntfy_server","https://ntfy.sh"), c["ntfy_topic"])' "$cfg")
  curl -sf -m 15 -H "Title: $1" --data-binary "$2" "$server/$topic" >/dev/null || echo "push FAILED: $1"
}
crashes=$(cat "$STATE/crashes" 2>/dev/null || echo 0)
unreach=0
for i in $(seq 1 "$ROUNDS"); do
  s=$(ssh -n -o ConnectTimeout=20 -o BatchMode=yes desktop "$STATUS" 2>/dev/null | tr -d '\r')
  echo "$(date -u '+%F %T') ${s:-UNREACHABLE} (crashes $crashes)"
  if [ -z "$s" ]; then
    unreach=$((unreach + 1))
    [ "$unreach" -eq 3 ] && push "EXP-085: desktop unreachable" "3 polls in a row with no answer. If it rebooted, open Tailscale on it."
  else
    unreach=0
    case "$s" in True\|*) echo "VERDICT FILE PRESENT"; exit 0;; esac
    dead=0
    case "$s" in *"|py 0|ch 0|"*) dead=1;; esac
    # A stage that exited non-zero (a python crash) is a crash; any other STOP is deliberate.
    case "$s" in *"STOP: stage "*" failed"*) ;; *STOP*) echo "CHAIN STOPPED"; exit 3;; esac
    case "$dead" in 1)
      crashes=$((crashes + 1)); echo "$crashes" > "$STATE/crashes"
      if [ "$crashes" -ge 3 ]; then
        push "EXP-085: third crash, not relaunching" "$s"; echo "THIRD CRASH"; exit 4
      fi
      ssh -n -o ConnectTimeout=20 -o BatchMode=yes desktop "$LAUNCH" 2>&1 | tr -d '\r'
      echo "RELAUNCHED after crash $crashes";;
    esac
  fi
  sleep "$SLEEP"
done
echo "ROUNDS DONE, still running: re-arm"; exit 5
