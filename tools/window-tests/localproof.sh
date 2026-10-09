#!/usr/bin/env bash
# localproof.sh — the acceptance test for the harness: a pi agent running on the
# LOCAL model (strata, through gpud) takes a bench window from inside its own turn,
# loses its model for the length of the window, and has to finish the turn on the
# restored strata.
#
#   tools/window-tests/localproof.sh OUT_DIR
set -uo pipefail
OUT=${1:?out dir}
mkdir -p "$OUT"
MODEL=/srv/models/gguf/ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
CMD="/home/evan/src/ds4/tools/ds4-window.sh --cuda -m $MODEL --prompt-file /tmp/bench-prompt-32k.txt --ssd-streaming --ctx-max 2048 --gen-tokens 16"
PROMPT="Use the bash tool to run exactly this one command, once, and wait for it to finish:

$CMD

When it has finished, reply with two lines and nothing else: the line from its output that contains 'bench rc=', and then the word LOCALPROOF-DONE."

date -Is > "$OUT/start"
echo "start $(cat "$OUT/start")"
timeout 900 pi -p --no-session --provider gpud --model swift-1.5-iq3_xxs --tools bash \
    "$PROMPT" > "$OUT/pi.out" 2>&1
echo "pi rc=$?" | tee "$OUT/rc"
date -Is > "$OUT/end"
echo "end $(cat "$OUT/end")"
echo "--- pi's final answer ---"; tail -5 "$OUT/pi.out"
echo "--- gpud timeline ---"
journalctl -u llmbox-gpud --since "$(cat "$OUT/start")" --until "$(cat "$OUT/end")" --no-pager 2>&1 \
    | grep -E "going to sleep|waking up|starting strata|strata ready|path=/v1/chat/completions" \
    | sed -E 's/^([A-Za-z]+ [0-9]+ )([0-9:]+).*msg="([^"]*)".*/\2 \3/; s/^([A-Za-z]+ [0-9]+ )([0-9:]+).*path=(\S+).*status=([0-9]+).*duration_ms=([0-9.]+).*/\2 \3 status=\4 \5ms/' \
    | tee "$OUT/timeline"
