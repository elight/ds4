#!/usr/bin/env bash
# Longer greedy run: ds4-mimo2 with lookahead 0/1/2, then llama.cpp, same prompt.
set -u
M=/srv/models/gguf/mimo/MiMo-V2.6-Flash-RL-Q2_K-00001-of-00002.gguf
WT=$(cd "$(dirname "$0")/../.." && pwd)
S=${S:-$HOME/claude-tmp/ds4-mimo2}; mkdir -p "$S"
Q='Write a short story, about 150 words, about a lighthouse keeper who finds a message in a bottle.'
RAW="<|im_start|>user
${Q}<|im_end|><|im_start|>assistant
<think></think>"
for la in ${LAS:-0 1 2}; do
  echo "=== ds4 lookahead $la"
  "$WT/ds4-mimo2" -m "$M" -p "$Q" -n 160 --ctx 4096 --lookahead "$la" --profile-out "$S/profile.bin" ${EXTRA:-} > "$S/story-ds4-la$la.txt"
done
if [ -n "${LLAMA:-}" ]; then
  echo "=== llama"
  /home/evan/src/llama.cpp/build/bin/llama-completion -m "$M" -p "$RAW" -n 160 --temp 0 \
    -ngl 99 -ot exps=CPU -c 4096 -t 10 --no-warmup -no-cnv > "$S/story-llama.txt" 2> "$S/story-llama.err"
  grep "eval time" "$S/story-llama.err"
fi
