#!/usr/bin/env bash
# Token parity through the decode path: teacher-forces llama.cpp's greedy
# story (TEXT) one token per step, so RAM-tier experts run on the CPU, and
# reports perplexity and top-1 agreement with the CPU split on and off.
# Run inside the GPU window:
#   ~/claude-tmp/gpu-window.sh --max 600 -- misc/llmbox/mimo2-score-decode.sh TEXT
set -u
TEXT=${1:?usage: $0 TEXT}
HERE=$(cd "$(dirname "$0")/../.." && pwd)
M=/srv/models/gguf/mimo/MiMo-V2.6-Flash-RL-Q2_K-00001-of-00002.gguf
for cpu in ${CPUS:-1 0}; do
  echo "=== ds4 cpu-experts $cpu"
  "$HERE/ds4-mimo2" -m "$M" -f "$TEXT" --raw --score --score-from 34 --ubatch 1 --ctx 4096 \
    --cpu-experts "$cpu" ${EXTRA:-} 2>&1 | grep -E "^score|^pos|RAM->CPU"
done
