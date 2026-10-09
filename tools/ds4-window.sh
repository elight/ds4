#!/usr/bin/env bash
# ds4-window.sh — run ONE ds4 benchmark window on llmbox's 3090 and give the box
# back the instant the bench stops.
#
# WHY THIS FILE EXISTS. When pi runs the local model (~/.pi/agent/settings.json:
# defaultProvider=gpud, defaultModel=swift-1.5-iq3_xxs), strata IS the agent's
# brain. Sleeping gpud to free the card therefore stops the agent's own inference,
# and the agent cannot act again until the card comes back. Three failures were
# measured on 2026-10-09 before this rewrite:
#
#   * A prompt below ds4-bench's floor (16690 tokens vs 32768) was discovered only
#     after the card had been slept, and exited 1: the window was already spent.
#     -> the floor is now checked BEFORE the card is touched.
#   * `kill -9` on the harness skipped the EXIT trap, killed the shell but left the
#     bench running, so gpud stayed asleep for the bench's remaining 59 s. A trap
#     is not a guarantee; SIGKILL cannot run one.
#     -> a detached watchdog survives the kill AND kills the bench.
#   * The exit path polled `systemctl is-active llmbox-strata` for 150 s, which
#     kept the card down and the agent offline for minutes. gpud does not start
#     strata on wake — it starts it when a request arrives, so the poll could
#     never succeed, and it burned the time it was waiting for.
#     -> the exit path does no sleeping at all, and strata is brought back by
#        issuing the request that actually brings it back.
#
#   ds4-window.sh [ds4-bench args...]        # run the bench in a window
#   ds4-window.sh --probe [ds4-bench args]   # report the prompt's token count only
#   ds4-window.sh --status                   # last window's restore result
#   ds4-window.sh --floor FILE               # is FILE above the floor? no card needed
#
# WHO STOPS AND STARTS WHAT (measured, gpud.py:2884 stops tenants, gpud.py:2944
# starts them). gpud is the single writer for VRAM tenancy and the only thing on
# this box with the sudo to touch tenant units. `gpu.sh sleep` makes gpud run
# `systemctl stop` on every running tenant — strata's 22.4 GB VRAM and 52.9 GiB RAM
# both go (`systemctl show -p MemoryCurrent --value llmbox-strata` = 56728317952).
# gpud itself never stops: sleep/wake are its authenticated HTTP control routes
# (ADR-097). A plain `systemctl start llmbox-strata` from an agent shell fails with
# "Interactive authentication required", so this script calls gpu.sh and nothing
# else. And `gpu.sh wake` alone does NOT start strata: measured 04:00:25, wake with
# no job waiting, strata still inactive 200 s later. Strata comes back when a
# request reaches gpud, which is why the restore below is a real completion.
set -uo pipefail

GPU_SH=${GPU_SH:-/home/evan/claude/tools/gpu.sh}
STRATA_UNIT=${STRATA_UNIT:-llmbox-strata.service}
DS4=${DS4:-/home/evan/src/ds4/ds4-bench}
GPUD_URL=${GPUD_URL:-http://127.0.0.1:8080}
GPUD_MODEL=${GPUD_MODEL:-swift-1.5-iq3_xxs}

# Prompt sizing. ds4-bench requires prompt tokens >= --ctx-max, plus --gen-tokens
# when teacher-forced decode is on (ds4_bench.c:741), and ctx_alloc is
# ctx_max + gen_tokens + 1 (ds4_bench.c:415). Defaults: ctx_max 32768,
# gen_tokens 128, ctx_start 2048 (ds4_bench.c:220-223).
PROMPT_TOKENS=${DS4_WINDOW_TOKENS:-32800}
PROMPT_BAND=${DS4_WINDOW_BAND:-120}
CALIBRATION=${DS4_WINDOW_CORPUS:-/tmp/bench-prompt.txt}
# Measured on this tokenizer: 90820 B / 16690 tokens = 5.4427 B/token.
CALIB_BPT=${DS4_WINDOW_BPT:-5.4427}

LOG_DIR=${DS4_WINDOW_LOGDIR:-/srv/models/gguf/ds4/window-logs}
PROBE_TIMEOUT=${DS4_WINDOW_PROBE_TIMEOUT:-300}
RUN_TIMEOUT=${DS4_WINDOW_TIMEOUT:-3600}
RESTORE_TIMEOUT=${DS4_WINDOW_RESTORE_TIMEOUT:-300}
WATCH_INTERVAL=${DS4_WINDOW_WATCH_INTERVAL:-1}

HARNESS_PID=$$
BENCH_LEADER=""
BENCH_PGID=""
WATCHDOG_PID=""
RESTORE_LOG=""
WOKEN=0

# Every line carries the time it happened, so a window's log times its own exit
# path: the gap from the bench's last line to "woke gpud" is readable off the log
# rather than reconstructed from the journal afterwards.
say() { printf '%s ds4-window: %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { echo "ds4-window: $*" >&2; exit 2; }

# --- the exit path: bounded, no sleeping -------------------------------------
# Everything here is instantaneous except gpu.sh wake itself (a local HTTP call).
# A sleep in this path is what kept the card down for 150 s.
#
# The trap is installed ONLY when this invocation is going to open a window (see
# open_window below). --floor, --status and any future read-only mode must not
# wake, sleep, or chat with anything: a query is not an operation.
wake() {
    [ "$WOKEN" = 1 ] && return 0
    WOKEN=1
    [ -n "$WATCHDOG_PID" ] && kill "$WATCHDOG_PID" 2>/dev/null
    say "bench gone (rc=${1:-0}) -> waking gpud"
    "$GPU_SH" wake || say "WARN gpu.sh wake failed (rc=$?)"
    kick_restore
    say "gpud awake. strata restore requested; check it with: $0 --status"
}
open_window() {
    trap 'wake $?' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
}

# The restore: gpud starts strata when a request arrives, so the request is the
# fix, not a poll. Detached, because the engine takes 40-70 s to load and the
# agent should not be made to wait for it — gpud is already awake and the card is
# already back.
kick_restore() {
    [ -z "${STAMP:-}" ] && STAMP=$(date +%Y%m%d-%H%M%S)
    RESTORE_LOG="$(do_restore "$STAMP")"
    say "restore request issued (a 1-token completion to $GPUD_MODEL); result -> $RESTORE_LOG"
}

# The restore itself, shared by every path (the normal exit, and the watchdog's
# path when this shell was killed). Detached: the engine takes 40-70 s to load and
# nobody should be made to wait on it, because gpud is already awake and the card
# is already back. Prints the log path.
do_restore() {
    local stamp=${1:-$(date +%Y%m%d-%H%M%S)}
    RESTORE_LOG="$LOG_DIR/restore-$stamp.log"
    setsid bash -c '
        url=$1; model=$2; out=$3; tmo=$4
        # Record the attempt BEFORE the call: the request blocks while the engine
        # loads (40-70 s), and if this process is killed in the meantime the log
        # stays empty and a window looks like it never asked. Measured 04:21.
        printf "%s issued pid=%s\\n" "$(date -Is)" "$$" >>"$out"
        code=$(curl -s --max-time "$tmo" -o "${out}.body" -w "%{http_code}" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"${model}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1}" \
            "${url}/v1/chat/completions" 2>/dev/null)
        printf "%s http=%s %s\\n" "$(date -Is)" "${code:-none}" \
            "$([ -s "${out}.body" ] && echo body=yes || echo body=no)" >>"$out"
    ' _ "$GPUD_URL" "$GPUD_MODEL" "$RESTORE_LOG" "$RESTORE_TIMEOUT" \
        >/dev/null 2>&1 </dev/null &
    printf '%s\n' "$RESTORE_LOG"
}

# The watchdog: a detached process in its own session, so it outlives this shell.
# A SIGKILLed harness runs no trap, and a bench left running holds the card for
# as long as it runs — measured at 59 s. So the watchdog covers both directions:
#   bench gone        -> wake gpud
#   harness gone      -> kill the bench, then wake gpud
start_watchdog() {
    # The watchdog logs every decision it makes. An exit path that cannot be
    # audited is how a bench got left holding the card for 59 s with nothing on
    # the record to say why, so each kill and its exit status goes to a file.
    local wlog="$LOG_DIR/watchdog-$STAMP.log"
    setsid bash -c '
        harness=$1; leader=$2; pgid=$3; gpu=$4; interval=$5; self=$6; stamp=$7; wlog=$8
        log() { printf "%s %s\n" "$(date +%H:%M:%S.%3N)" "$*" >>"$wlog"; }
        log "watchdog start: harness=$harness leader=$leader pgid=$pgid"
        while kill -0 "$leader" 2>/dev/null; do
            if ! kill -0 "$harness" 2>/dev/null; then
                # The harness is gone and nobody is recording numbers: stop the
                # bench, then give the card back. Never leave a bench holding it.
                log "harness $harness gone -> TERM bench group -$pgid"
                kill -TERM -"$pgid" 2>>"$wlog"; log "  TERM rc=$?"
                for _ in 1 2 3 4 5 6 7 8 9 10; do
                    kill -0 "$leader" 2>/dev/null || break
                    sleep 0.5
                done
                if kill -0 "$leader" 2>/dev/null; then
                    log "  bench survived TERM -> KILL group -$pgid"
                    kill -KILL -"$pgid" 2>>"$wlog"; log "  KILL rc=$?"
                else
                    log "  bench gone after TERM"
                fi
                break
            fi
            sleep "$interval"
        done
        log "waking gpud"
        "$gpu" wake >/dev/null 2>&1; log "  wake rc=$?"
        # gpud starts strata only when a request arrives, so waking gpud is half
        # the job: the same restore chat the normal exit path sends, from here, so
        # a SIGKILLed window still leaves strata coming back.
        DS4_WINDOW_STAMP="$stamp" "$self" --restore >/dev/null 2>&1; log "  restore rc=$?"
        log "watchdog done"
    ' _ "$HARNESS_PID" "$BENCH_LEADER" "$BENCH_PGID" "$GPU_SH" "$WATCH_INTERVAL" \
        "$(readlink -f "$0")" "${STAMP:-}" "$wlog" >/dev/null 2>&1 </dev/null &
    WATCHDOG_PID=$!
    say "watchdog $WATCHDOG_PID (own session): wakes gpud within ${WATCH_INTERVAL}s of the bench exiting, kills the bench and restores strata if this shell is killed; decisions -> $wlog"
}

# --- status / floor: both need no card ---------------------------------------
do_status() {
    local newest
    newest=$(ls -t "$LOG_DIR"/restore-*.log 2>/dev/null | head -1)
    echo "gpud:    $(systemctl is-active "$STRATA_UNIT" 2>/dev/null | sed 's/^/strata=/' ) $(systemctl is-active llmbox-gpud 2>/dev/null | sed 's/^/gpud=/')"
    echo "sleeping: $("$GPU_SH" json 2>/dev/null | sed -n 's/.*"sleeping": *\([a-z]*\).*/\1/p' | head -1)"
    if [ -n "$newest" ]; then
        echo "restore: $(tail -1 "$newest")  ($newest)"
    else
        echo "restore: no window restore log in $LOG_DIR"
    fi
}

do_floor() {
    local f="$1" manifest want
    [ -r "$f" ] || die "no such prompt file: $f"
    manifest="$f.floor"
    want="$(stat -c '%s' "$f") $(sha256sum "$f" | cut -c1-16)"
    if [ -r "$manifest" ]; then
        printf '%s\n' "$(cat "$manifest")"
        if [ "$(cut -d' ' -f1-2 "$manifest")" = "$want" ]; then
            say "matches this file's size+sha256: no card needed, this count is valid"
        else
            say "STALE: the file changed since this count was recorded; re-probe with --probe"
        fi
    else
        say "no recorded count for $f (${want}); probe it once with --probe"
    fi
}

# --- args ---------------------------------------------------------------------
MODE=bench
ARGS=()
MODEL=""
PROMPT=""
FLOOR_FILE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --probe) MODE=probe; shift ;;
        --status) MODE=status; shift ;;
        --restore) MODE=restore; shift ;;
        --floor) MODE=floor; FLOOR_FILE=${2:-}; shift 2 ;;
        -m|--model) MODEL=${2:-}; ARGS+=("$1" "$2"); shift 2 ;;
        --prompt-file) PROMPT=${2:-}; ARGS+=("$1" "$2"); shift 2 ;;
        *) ARGS+=("$1"); shift ;;
    esac
done

[ "$MODE" = status ] && { do_status; exit 0; }
if [ "$MODE" = restore ]; then
    mkdir -p "$LOG_DIR" 2>/dev/null || LOG_DIR=/tmp
    do_restore "${DS4_WINDOW_STAMP:-$(date +%Y%m%d-%H%M%S)}"
    exit 0
fi
if [ "$MODE" = floor ]; then
    [ -n "$FLOOR_FILE" ] || die "--floor needs a file"
    do_floor "$FLOOR_FILE"
    exit 0
fi

[ -z "$MODEL" ] && die "no -m MODEL given"
[ -z "$PROMPT" ] && die "no --prompt-file given"

# --- preflight, before the card is touched ------------------------------------
[ -x "$DS4" ] || die "$DS4 is not an executable"
[ -r "$MODEL" ] || die "model not readable: $MODEL"
[ -x "$GPU_SH" ] || die "missing $GPU_SH"
mkdir -p "$LOG_DIR" 2>/dev/null || LOG_DIR=/tmp

matches=$(pgrep -af "$DS4" 2>/dev/null \
    | awk -v self="$$" '$1 != self && $2 != "pgrep" && $2 != "awk" && $2 != "grep"' \
    | head -3)
if [ -n "$matches" ]; then
    # Refuse on evidence, not on pgrep's exit code.
    #
    # Excluding $2==pgrep is not cosmetic: pgrep's own argv contains the pattern,
    # so a concurrent pgrep (a second window's check, or a caller's wait loop
    # polling for the bench) lands in the match list and the checker sees the
    # checker. Measured 2026-10-09: the harness refused to open a window because
    # "824487 pgrep -f /tmp/watchtest/stub-bench.sh" was in the table, and that
    # phantom cost a full test cycle. Only a real bench process counts.
    say "a bench is already running; the match is:"
    printf '%s\n' "$matches"
    die "refusing to open a second window on the same card"
fi

# What the bench will demand of the prompt: --ctx-max from the args, else 32768.
CTX_MAX=32768
GEN_TOKENS=128
for (( i = 0; i < ${#ARGS[@]}; i++ )); do
    case "${ARGS[i]}" in
        --ctx-max)   CTX_MAX=${ARGS[i+1]:-32768} ;;
        --gen-tokens) GEN_TOKENS=${ARGS[i+1]:-128} ;;
    esac
done
FLOOR=$(( CTX_MAX ))

build_prompt() {
    [ -r "$CALIBRATION" ] || die "no corpus to build a prompt from: $CALIBRATION"
    python3 - "$CALIBRATION" "$PROMPT" "$PROMPT_TOKENS" "$CALIB_BPT" <<'PY'
import sys
src, dst, tokens, bpt = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
target = int(tokens * bpt)
body = open(src, 'rb').read()
out = bytearray()
while len(out) < target:
    out += body
open(dst, 'wb').write(bytes(out[:target]))
PY
    say "built $PROMPT at $(wc -c <"$PROMPT") bytes (~${PROMPT_TOKENS} tok at ${CALIB_BPT} B/tok)"
}

# The floor check that must not need the card. A recorded count is only trusted
# for the exact bytes it was measured on (size + sha256), because a stale count is
# worse than none: it would pass a prompt the bench will refuse.
floor_known() {
    local manifest="$PROMPT.floor" want have n
    [ -r "$manifest" ] || return 1
    want="$(stat -c '%s' "$PROMPT") $(sha256sum "$PROMPT" | cut -c1-16)"
    have="$(cut -d' ' -f1-2 "$manifest")"
    [ "$want" = "$have" ] || return 1
    n="$(cut -d' ' -f4 "$manifest")"
    [ -n "$n" ] || return 1
    PROMPT_SEEN="$n"
    return 0
}

PROMPT_SEEN=""
# Teacher-forced decode adds --gen-tokens to the floor (ds4_bench.c:741), so a
# prompt that clears ctx_max can still be refused by that run mode. Say so before
# the card is slept rather than after.
note_teacher_forced_margin() {
    local need=$(( FLOOR + GEN_TOKENS ))
    [ "$PROMPT_SEEN" -ge "$need" ] && return 0
    say "note: $PROMPT_SEEN tokens clears ctx-max=$FLOOR but not $need; a teacher-forced decode run would be refused"
}
probe_tokens() {
    # --ctx-max far above any real context makes the length check fail on purpose,
    # so the bench prints "prompt has N tokens" and exits before doing any work.
    local out rc seen
    out=$(timeout "$PROBE_TIMEOUT" "$DS4" "${ARGS[@]}" --ctx-max 100000000 --gen-tokens 0 2>&1)
    rc=$?
    seen=$(printf '%s\n' "$out" | sed -n 's/.*prompt has \([0-9]*\) tokens.*/\1/p' | tail -1)
    if [ -z "$seen" ]; then
        say "probe printed no token count (rc=$rc); first lines:"
        printf '%s\n' "$out" | head -5
        return 1
    fi
    PROMPT_SEEN="$seen"
    say "probe: prompt has $seen tokens (floor $FLOOR, target ${PROMPT_TOKENS} +/-${PROMPT_BAND})"
    if [ "$seen" -lt $(( PROMPT_TOKENS - PROMPT_BAND )) ] || [ "$seen" -gt $(( PROMPT_TOKENS + PROMPT_BAND )) ]; then
        say "probe: out of band, rebuilding from the measured ratio"
        CALIB_BPT=$(python3 - "$CALIB_BPT" "$seen" "$PROMPT_TOKENS" <<'PY'
import sys
print(round(float(sys.argv[1]) * int(sys.argv[2]) / int(sys.argv[3]), 4))
PY
)
        build_prompt
        probe_tokens && return 0
        return 1
    fi
    printf '%s %s %s %s\n' "$(stat -c '%s' "$PROMPT")" "$(sha256sum "$PROMPT" | cut -c1-16)" \
        "$FLOOR" "$seen" > "$PROMPT.floor"
    return 0
}

say "preflight: model=$MODEL prompt=$PROMPT"
if [ ! -r "$PROMPT" ]; then
    build_prompt
fi
say "preflight ok: $(wc -c <"$PROMPT") bytes in the prompt file"

if floor_known; then
    if [ "$PROMPT_SEEN" -lt "$FLOOR" ]; then
        die "prompt has $PROMPT_SEEN tokens, the bench needs $FLOOR — the card was NOT touched; rebuild the prompt or lower --ctx-max"
    fi
    say "floor ok before sleeping: $PROMPT_SEEN tokens >= $FLOOR (recorded for this exact file)"
    note_teacher_forced_margin
else
    say "no recorded count for this prompt; it will be probed inside the window, before the bench does any work"
fi

# --- the window ---------------------------------------------------------------
say "sleeping gpud (gpud stops strata and frees its VRAM and RAM)"
open_window
"$GPU_SH" sleep || die "gpu.sh sleep failed"

if [ "$MODE" = probe ]; then
    probe_tokens
    RC=$?
    exit "$RC"
fi

if ! floor_known; then
    probe_tokens || die "prompt probe failed; not spending the window on the bench"
    [ "$PROMPT_SEEN" -ge "$FLOOR" ] || die "prompt has $PROMPT_SEEN tokens, the bench needs $FLOOR; aborting before the bench"
    note_teacher_forced_margin
fi

STAMP=$(date +%Y%m%d-%H%M%S)
LOG="$LOG_DIR/$STAMP.log"
say "card is free; bench starting -> $LOG"

# The bench gets its own session so the watchdog can kill exactly it and never the
# shell that started it. setsid -w stays alive for the bench's lifetime, which is
# what the watchdog watches. The session leader records its own PID, and that PID
# IS the process group — deterministic, rather than hoping a pgrep lands on the
# right child while the bench is still coming up.
PGIDFILE="$LOG_DIR/$STAMP.pgid"
rm -f "$PGIDFILE"
setsid -w bash -c 'echo $$ > "$1"; shift; exec "$@"' _ "$PGIDFILE" \
    timeout -k 30 "$RUN_TIMEOUT" "$DS4" "${ARGS[@]}" >"$LOG" 2>&1 &
BENCH_LEADER=$!
BENCH_PGID=""
for _ in $(seq 1 25); do
    BENCH_PGID=$(cat "$PGIDFILE" 2>/dev/null)
    [ -n "$BENCH_PGID" ] && break
    sleep 0.2
done
if [ -n "$BENCH_PGID" ]; then
    say "bench pid $BENCH_PGID (its own session and process group); leader $BENCH_LEADER"
else
    say "WARN the bench never recorded its pid; the watchdog will still wake gpud and restore strata"
fi
start_watchdog
wait "$BENCH_LEADER"
RC=$?
[ "$RC" = 124 ] && say "bench hit the ${RUN_TIMEOUT}s timeout"
tail -n 40 "$LOG"
say "bench rc=$RC log=$LOG"
exit "$RC"
