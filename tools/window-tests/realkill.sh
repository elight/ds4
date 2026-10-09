#!/usr/bin/env bash
# realkill.sh — kill -9 the harness in a REAL window and measure the gap between the
# kill and gpud waking up, against gpud's own journal.
#
# $1 = mode: sock (SIGKILL the harness only) | group (SIGKILL the whole process
# group, which is what a tool timeout does)
set -uo pipefail
MODE=${1:-sock}
W=/tmp/realkill
mkdir -p "$W"
LOG="$W/harness-$MODE.log"
: > "$LOG"
HARNESS_SH=/home/evan/src/ds4/tools/ds4-window.sh

# Two windows in a row would collide; refuse rather than race.
if [ -n "$(pgrep -f 'ds4-bench --cuda' 2>/dev/null)" ]; then
    echo "REFUSING: a bench is already running"; exit 1
fi

setsid bash -c "exec $HARNESS_SH \
  --cuda -m '/srv/models/gguf/ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf' \
  --prompt-file /tmp/bench-prompt-32k.txt --ssd-streaming --ctx-max 2048 --gen-tokens 16" \
  > "$LOG" 2>&1 &
HARNESS=$!
PGID=$(ps -o pgid= -p "$HARNESS" 2>/dev/null | tr -d ' ')

for _ in $(seq 1 240); do
    grep -q "card is free; bench starting" "$LOG" 2>/dev/null && break
    sleep 1
done
if ! grep -q "card is free; bench starting" "$LOG" 2>/dev/null; then
    echo "FAIL: the window never started:"; tail -4 "$LOG"; exit 1
fi
sleep 8   # bench is in prefill, holding the card

echo "== before =="
echo "gpud sleeping: $(gpu.sh json | sed -n 's/.*\"sleeping\": *\([a-z]*\).*/\1/p' | head -1)"
echo "bench procs:   $(pgrep -cf 'ds4-bench --cuda')"

T0=$(date +%s.%N)
if [ "$MODE" = group ]; then
    kill -9 -"$PGID" 2>/dev/null; echo "KILL -9 to the process group -$PGID (harness $HARNESS + bench)"
else
    kill -9 "$HARNESS" 2>/dev/null; echo "KILL -9 to the harness only ($HARNESS); the bench is left orphaned on purpose"
fi

for _ in $(seq 1 60); do
    [ "$(pgrep -cf 'ds4-bench --cuda')" = 0 ] && { T1=$(date +%s.%N); echo "bench gone after $(python3 -c "print(round($T1-$T0,2))")s"; break; }
    sleep 0.25
done
for _ in $(seq 1 60); do
    if gpu.sh json 2>/dev/null | grep -q '"sleeping": *false'; then
        T2=$(date +%s.%N); echo "gpud AWAKE after $(python3 -c "print(round($T2-$T0,2))")s"; break
    fi
    sleep 0.25
done

echo "== gpud's own journal =="
journalctl -u llmbox-gpud --since "-5min" --no-pager 2>&1 | grep -E "going to sleep|waking up|starting strata" | tail -4 | cut -c1-95
echo "== watchdog log =="
newest=$(ls -t /srv/models/gguf/ds4/window-logs/watchdog-*.log 2>/dev/null | head -1)
[ -n "$newest" ] && { echo "($newest)"; tail -8 "$newest"; }
echo "== leftover benches (want none) =="
pgrep -af "ds4-bench --cuda" 2>/dev/null | head -3
echo "restore log: $(ls -t /srv/models/gguf/ds4/window-logs/restore-*.log | head -1)"
echo "(end $MODE; strata's restore runs on, detached)"
