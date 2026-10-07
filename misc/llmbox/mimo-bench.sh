#!/usr/bin/env bash
# mimo-bench.sh [QUEUE] [OUTDIR] -- the MiMo benchmark queue, run unattended.
#
# Every step borrows the card through ~/claude-tmp/gpu-window.sh, which evicts
# the live tenants, runs the step under a hard timeout, and wakes gpud on its
# own EXIT trap. This script adds the two things a long campaign needs on top:
#
#   resume  a step is recorded in $OUT/done only after it exits 0, so re-running
#           this script picks up at the first step that has not landed; and
#   recovery  a window killed with SIGKILL cannot run its own trap, so gpud may
#           be left asleep. Before the first step, and on our own EXIT, we wake
#           it if it is asleep. The card is never left closed by this script.
#
# Metrics land as one JSON object per step in $OUT/results.jsonl, so a sweep is
# diffable and the numbers in docs/RAM_EXPERT_TIER.md come from a file, not from
# scrolling terminal output.
#
#   ~/claude-tmp/gpu-window.sh is the only thing that touches the card.
#   Env: MAX=900 default window; ONLY="a b" run only those labels; FORCE=1
#        re-runs labels already in done.
set -u

QUEUE=${1:-baseline}
OUT=${2:-$HOME/claude-tmp/mimo-bench/$QUEUE}
export OUT                                # the step scripts build their paths from it
MAX=${MAX:-900}
mkdir -p "$OUT/steps"

WT=$(cd "$(dirname "$0")/../.." && pwd)
GW=$HOME/claude-tmp/gpu-window.sh
GPU=/home/evan/claude/tools/gpu.sh
DONE=$OUT/done
RESULTS=$OUT/results.jsonl
[ -f "$GW" ] || { echo "mimo-bench: need $GW" >&2; exit 2; }

# --- gpud is asleep only while a window is open; wake it if we find it that way.
is_asleep() { "$GPU" status 2>/dev/null | grep -q '^SLEEPING'; }
ensure_awake() { is_asleep && { echo "mimo-bench: gpud asleep, waking" >&2; "$GPU" wake >/dev/null 2>&1; }; }
on_exit() { rc=$?; ensure_awake; [ $rc -ne 0 ] && echo "mimo-bench: stopped rc=$rc; completed steps are in $DONE" >&2; }
trap on_exit EXIT
trap 'exit 130' INT TERM
ensure_awake                            # recovery from a previous SIGKILLed window

# --- the queue: label | window seconds | command ---
# Commands run with the worktree as cwd so the relative misc/ paths resolve.
declare -a STEPS=()
case $QUEUE in
baseline)
  STEPS=(
    'story|900|misc/llmbox/mimo2-story.sh "$OUT/steps/story"'
    'prefill|1500|misc/llmbox/mimo2-prefill.sh "$OUT/steps/prefill"'
    'score|900|misc/llmbox/mimo2-scoretext.py "$OUT/steps/story/story-llama.txt" "$OUT/steps/score-text.txt" && misc/llmbox/mimo2-score-decode.sh "$OUT/steps/score-text.txt"'
  ) ;;
mtp)
  STEPS=(
    'mtp|1500|CPUS="0 1" MODES="plain mtp1 mtp3 mtp3g70" misc/llmbox/mimo2-mtp.sh "$OUT/steps/mtp"'
  ) ;;
sweep)
  STEPS=(
    'sweep|1500|misc/llmbox/mimo2-sweep.sh "$OUT/steps/sweep"'
  ) ;;
profile)
  STEPS=(
    'profile|1500|misc/llmbox/mimo2-profile.sh "$OUT/steps/profile"'
  ) ;;
*) echo "mimo-bench: unknown queue '$QUEUE' (baseline|mtp|sweep)" >&2; exit 2 ;;
esac

for spec in "${STEPS[@]}"; do
  IFS='|' read -r label secs cmd <<< "$spec"
  if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $label "* ]]; then continue; fi
  if [ "${FORCE:-0}" != 1 ] && grep -qxF "$label" "$DONE" 2>/dev/null; then
    echo "=== $label: already done, skipping"
    continue
  fi
  echo "=== $label: borrowing the card for at most ${secs}s"
  # shellcheck disable=SC2024
  "$GW" --max "$secs" -- bash -c "cd '$WT' && $cmd" \
    > "$OUT/$label.out" 2> "$OUT/$label.err"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "=== $label: rc=$rc, NOT recorded; rerun this script to retry" >&2
    tail -5 "$OUT/$label.err" >&2
    exit $rc
  fi
  python3 "$(dirname "$0")/mimo-bench-tally.py" "$OUT/$label.out" "$OUT/$label.err" \
    > "$OUT/$label.jsonl" || { echo "=== $label: tally failed, NOT recorded" >&2; exit 1; }
  # A step that ran and printed no metric is a step that did not run: the engine
  # died at load, or wrote its story somewhere we are not looking. Refuse to
  # record it, because a green rc with no numbers is how a campaign lies.
  if ! grep -qE '"(decode_tps|prefill_tps|llama_decode|llama_prefill|score)' "$OUT/$label.jsonl"; then
    echo "=== $label: no metric parsed, NOT recorded" >&2
    tail -5 "$OUT/$label.err" >&2
    exit 1
  fi
  sed "s/^{/{$\"step\":\"$label\",/" "$OUT/$label.jsonl" >> "$RESULTS"
  echo "$label" >> "$DONE"
  echo "=== $label: recorded $(wc -l < "$OUT/$label.jsonl") run(s)"
  cat "$OUT/$label.jsonl"
done
echo "mimo-bench: queue '$QUEUE' complete; results in $RESULTS"
