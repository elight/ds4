#!/usr/bin/env bash
# MiMo V2.6 Flash: ds4-mimo2 vs llama.cpp, greedy, same chat prompt.
# Usage: mimo2-runs.sh ds4|llama|bench   (run inside ~/claude-tmp/gpu-window.sh)
set -u
M=/srv/models/gguf/mimo/MiMo-V2.6-Flash-RL-Q2_K-00001-of-00002.gguf
WT=$(cd "$(dirname "$0")/../.." && pwd)
Q='What is the capital of France? Answer in one sentence.'
RAW="<|im_start|>user
${Q}<|im_end|><|im_start|>assistant
<think></think>"
case "${1:-}" in
ds4)   shift; "$WT/ds4-mimo2" -m "$M" -p "$Q" -n 24 --ctx 4096 --ubatch 512 --top-logits 5 "$@" ;;
llama) /home/evan/src/llama.cpp/build/bin/llama-completion -m "$M" -p "$RAW" -n 24 --temp 0 \
         -ngl 99 -ot exps=CPU -c 4096 -t 10 --no-warmup -no-cnv 2>&1 | tail -25 ;;
bench) "$WT/ds4-mimo2" -m "$M" --ctx 8192 --bench "${2:-2048}" -n "${3:-64}" --ubatch "${4:-1024}" ;;
real)  shift; "$WT/ds4-mimo2" -m "$M" -f "${PROMPT2K:-$HOME/claude-tmp/ds4/prompt-2k.txt}" --raw -n 64 --ctx 8192 --ubatch 1024 "$@" > /dev/null ;;
lreal) /home/evan/src/llama.cpp/build/bin/llama-completion -m "$M" -f "${PROMPT2K:-$HOME/claude-tmp/ds4/prompt-2k.txt}" -n 64 --temp 0 \
         -ngl 99 -ot exps=CPU -c 8192 -b 1024 -ub 1024 -t 10 --no-warmup -no-cnv 2>&1 >/dev/null | grep -E "prompt eval time|eval time" ;;
*) echo "usage: $0 ds4|llama|bench|real|lreal [prefill decode ubatch]" >&2; exit 2 ;;
esac
