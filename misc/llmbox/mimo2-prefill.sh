#!/usr/bin/env bash
# MiMo V2.6 Flash long-prompt prefill: ds4-mimo2 and llama.cpp on the same
# 2203-token prompt (llama.cpp docs/build.md, cut with llama-tokenize), same
# ctx and ubatch, greedy, 64 tokens decoded.
# Run inside the GPU window:
#   ~/claude-tmp/gpu-window.sh --max 1500 -- misc/llmbox/mimo2-prefill.sh OUTDIR
# Env: PROMPT2K path to the prompt; UBATCH/CTX/N to move the batch shape.
set -u
OUT=${1:?usage: $0 OUTDIR}
mkdir -p "$OUT" || exit 2
[ -d "$OUT" ] || { echo "mimo2-prefill: cannot write $OUT" >&2; exit 2; }
HERE=$(cd "$(dirname "$0")/../.." && pwd)
M=/srv/models/gguf/mimo/MiMo-V2.6-Flash-RL-Q2_K-00001-of-00002.gguf
LLAMA=/home/evan/src/llama.cpp/build/bin/llama-completion
P=${PROMPT2K:-$HOME/claude-tmp/ds4/prompt-2k.txt}
CTX=${CTX:-8192}
UB=${UBATCH:-1024}
N=${N:-64}
[ -s "$P" ] || { echo "mimo2-prefill: no prompt at $P" >&2; exit 2; }

for cpu in ${CPUS:-1}; do
  echo "=== ds4 cpu-experts $cpu"
  "$HERE/ds4-mimo2" -m "$M" -f "$P" --raw -n "$N" --ctx "$CTX" --ubatch "$UB" \
    --cpu-experts "$cpu" ${EXTRA:-} > "$OUT/prefill-ds4-cpu$cpu.txt" 2> "$OUT/prefill-ds4-cpu$cpu.err"
  grep -E "t/s|experts:|phases:" "$OUT/prefill-ds4-cpu$cpu.err"
done
if [ "${LLAMA_RUN:-1}" = 1 ]; then
  echo "=== llama.cpp"
  "$LLAMA" -m "$M" -f "$P" -n "$N" --temp 0 -ngl 99 -ot exps=CPU -c "$CTX" \
    -b "$UB" -ub "$UB" -t 10 --no-warmup -no-cnv > "$OUT/prefill-llama.txt" 2> "$OUT/prefill-llama.err"
  grep -E "eval time" "$OUT/prefill-llama.err"
fi
