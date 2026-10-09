#!/usr/bin/env bash
# ds4-window.sh — run ONE ds4 benchmark window on llmbox's 3090 and give the box
# back the instant the bench stops.
#
# Why this file exists: a bench that crashed used to leave the card asleep and
# strata down until somebody remembered to wake them, and the box sat unusable
# for minutes. Here the wake is a trap, so it runs the moment the bench exits by
# any route: clean finish, CUDA OOM, segfault, timeout, Ctrl-C, SIGTERM. The
# window is never longer than the bench and never outlives it.
#
#   ds4-window.sh [ds4-bench args...]        # run the bench in a window
#   ds4-window.sh --probe [ds4-bench args]   # report the prompt's token count only
#
# Preflight runs BEFORE the card is touched, so a prompt that is too short costs
# a second, not a window.
#
# WHO STOPS AND STARTS WHAT (measured 2026-10-09, gpud.py:2884 and gpud.py:2944):
# gpud is the single writer for VRAM tenancy, and it is the only thing on this box
# with the sudo to touch tenant units. `gpu.sh sleep` makes gpud run
# `systemctl stop` on every running tenant — strata's 22.4 GB of VRAM and 52.9 GiB
# of RAM both go. `gpu.sh wake` makes gpud re-admit and `systemctl start` strata.
# gpud itself never stops: sleep/wake are its HTTP control routes (ADR-097).
# So this script calls gpu.sh sleep and gpu.sh wake and NOTHING else. A plain
# `systemctl start llmbox-strata` from an agent shell fails with "Interactive
# authentication required" — verified — and a failed attempt here is noise, not a
# fix. gpud brings strata back; the script only checks that it did.
set -uo pipefail

GPU_SH=${GPU_SH:-/home/evan/claude/tools/gpu.sh}
STRATA_UNIT=${STRATA_UNIT:-llmbox-strata.service}
DS4=${DS4:-/home/evan/src/ds4/ds4-bench}

# Prompt sizing. ds4-bench requires prompt tokens >= --ctx-max, plus --gen-tokens
# when teacher-forced decode is on (ds4_bench.c:741), and ctx_alloc is
# ctx_max + gen_tokens + 1 (ds4_bench.c:415). So the prompt must clear the floor
# and not run far past it. Target just over the default 32768 floor.
PROMPT_TOKENS=${DS4_WINDOW_TOKENS:-32800}
PROMPT_BAND=${DS4_WINDOW_BAND:-120}
CALIBRATION=${DS4_WINDOW_CORPUS:-/tmp/bench-prompt.txt}
# Measured on this tokenizer: 90820 B / 16690 tokens = 5.4427 B/token.
CALIB_BPT=${DS4_WINDOW_BPT:-5.4427}

LOG_DIR=${DS4_WINDOW_LOGDIR:-/srv/models/gguf/ds4/window-logs}
PROBE_TIMEOUT=${DS4_WINDOW_PROBE_TIMEOUT:-300}
RUN_TIMEOUT=${DS4_WINDOW_TIMEOUT:-3600}
# Strata costs ~69 s to come back. Verify it did, and say so on the record.
VERIFY_SECS=${DS4_WINDOW_VERIFY_SECS:-150}

say() { echo "ds4-window: $*"; }
die() { echo "ds4-window: $*" >&2; exit 2; }

# --- wake: exactly once, as the first thing every exit path does --------------
WOKEN=0
wake() {
    [ "$WOKEN" = 1 ] && return 0
    WOKEN=1
    say "bench exited (rc=${1:-0}) -> waking gpud now"
    "$GPU_SH" wake || say "WARN gpu.sh wake failed (rc=$?)"
    verify_strata
}
verify_strata() {
    local i
    for (( i = 0; i < VERIFY_SECS; i += 5 )); do
        if [ "$(systemctl is-active "$STRATA_UNIT" 2>/dev/null)" = active ]; then
            say "restored: gpud awake, $STRATA_UNIT active after ${i}s"
            "$GPU_SH" status 2>/dev/null | sed -n '1,6p'
            return 0
        fi
        sleep 5
    done
    say "WARN $STRATA_UNIT not active after ${VERIFY_SECS}s — gpud is awake, strata did not come back; check: journalctl -u $STRATA_UNIT -n 50"
}
trap 'wake $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# --- args ---------------------------------------------------------------------
MODE=bench
ARGS=()
MODEL=""
PROMPT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --probe) MODE=probe; shift ;;
        -m|--model) MODEL=${2:-}; ARGS+=("$1" "$2"); shift 2 ;;
        --prompt-file) PROMPT=${2:-}; ARGS+=("$1" "$2"); shift 2 ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
[ -z "$MODEL" ] && die "no -m MODEL given"
[ -z "$PROMPT" ] && die "no --prompt-file given"

# --- preflight, before the card is touched ------------------------------------
[ -x "$DS4" ] || die "$DS4 is not an executable"
[ -r "$MODEL" ] || die "model not readable: $MODEL"
[ -x "$GPU_SH" ] || die "missing $GPU_SH"
mkdir -p "$LOG_DIR" 2>/dev/null || LOG_DIR=/tmp

# Build the prompt to the target size by repeating a corpus whose bytes-per-token
# is already measured on this tokenizer. Repeating the same corpus keeps the ratio
# stable, which is what makes one probe enough.
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
    say "probe: prompt has $seen tokens (target ${PROMPT_TOKENS} +/-${PROMPT_BAND})"
    if [ "$seen" -lt $(( PROMPT_TOKENS - PROMPT_BAND )) ] || [ "$seen" -gt $(( PROMPT_TOKENS + PROMPT_BAND )) ]; then
        say "probe: out of band, rebuilding from the measured ratio"
        CALIB_BPT=$(python3 - "$CALIB_BPT" "$seen" "$PROMPT_TOKENS" <<'PY'
import sys
print(round(float(sys.argv[1]) * int(sys.argv[2]) / int(sys.argv[3]), 4))
PY
)
        build_prompt
        probe_tokens && return 0
    fi
    return 0
}

say "preflight: model=$MODEL prompt=$PROMPT"
if [ ! -r "$PROMPT" ]; then
    build_prompt
fi
say "preflight ok: $(wc -c <"$PROMPT") bytes in the prompt file"

# --- the window ---------------------------------------------------------------
say "sleeping gpud (gpud stops strata and frees its VRAM and RAM)"
"$GPU_SH" sleep || die "gpu.sh sleep failed"
say "card is free; bench starting"

if [ "$MODE" = probe ]; then
    probe_tokens
    RC=$?
    exit "$RC"
fi

STAMP=$(date +%Y%m%d-%H%M%S)
LOG="$LOG_DIR/$STAMP.log"
timeout "$RUN_TIMEOUT" "$DS4" "${ARGS[@]}" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
[ "$RC" = 124 ] && say "bench hit the ${RUN_TIMEOUT}s timeout"
say "bench rc=$RC log=$LOG"
exit "$RC"
