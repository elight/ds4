#!/usr/bin/env bash
# Qwen3.8 Flash Next Q2 on llmbox (RTX 3090, i9-10900K, 64 GB): CPU/GPU hybrid
# decode measurements. Each case runs in its own GPU window, because the card
# is shared with live services (gpu-window.sh pauses gpud and always wakes it).
#
#   qwen-hybrid.sh bench <on|off> <ctx>     ds4-bench, prefill + 128 decode tokens
#   qwen-hybrid.sh ppl <on|off>             teacher-forced NLL over ~480 tokens
#   qwen-hybrid.sh greedy <on|off>          96 greedy tokens on a short prompt
#
# Output: $OUT (default ~/claude-tmp/ds4-qwen-hybrid). Extra env passes through
# (DS4_CPU_HYBRID_PCIE, DS4_STREAM_STATS, ...).
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
M=${M:-/srv/models/gguf/ds4/Qwen3.8-Flash-Next-Q2.gguf}
W=${W:-$HOME/claude-tmp/gpu-window.sh}
OUT=${OUT:-$HOME/claude-tmp/ds4-qwen-hybrid}
TAG=${TAG:-}
mkdir -p "$OUT"
cd "$ROOT"

mode=${2:-on}
case "$mode" in
  on)  export DS4_CPU_HYBRID=1 ;;
  off) export DS4_CPU_HYBRID=0 ;;
  *) echo "mode must be on or off" >&2; exit 2 ;;
esac
export DS4_STREAM_STATS=${DS4_STREAM_STATS:-64}

case "${1:-}" in
  bench)
    ctx=${3:-8192}
    name="bench-$mode-$ctx$TAG"
    # The story prompt is ~31K tokens; repeat it so 32K frontiers fit. The
    # first 31K tokens are unchanged, so shorter frontiers measure the same text.
    cat tests/long_context_story_prompt.txt tests/long_context_story_prompt.txt > "$OUT/story-x2.txt"
    "$W" --max 600 -- ./ds4-bench -m "$M" --cuda --ssd-streaming \
      --prompt-file "$OUT/story-x2.txt" \
      --ctx-start "$ctx" --ctx-max "$ctx" --gen-tokens 128 \
      --csv "$OUT/$name.csv" > "$OUT/$name.log" 2>&1
    grep -E "CPU hybrid|RAM->CPU|expert slots|VRAM" "$OUT/$name.log" | tail -4
    cat "$OUT/$name.csv" ;;
  ppl)
    name="ppl-$mode$TAG"
    head -c 2400 tests/long_context_story_prompt.txt > "$OUT/ppl-text.txt"
    "$W" --max 600 -- ./ds4 -m "$M" --cuda --ssd-streaming -c 512 \
      --perplexity-file "$OUT/ppl-text.txt" > "$OUT/$name.out" 2> "$OUT/$name.log"
    tail -3 "$OUT/$name.out"; grep -i -E "perplex|nll" "$OUT/$name.log" | tail -2 ;;
  greedy)
    name="greedy-$mode$TAG"
    "$W" --max 600 -- ./ds4 -m "$M" --cuda --ssd-streaming -c 8192 \
      -p "Explain in three sentences why the sky is blue." -n 96 --temp 0 \
      > "$OUT/$name.out" 2> "$OUT/$name.log"
    cat "$OUT/$name.out"; grep -E "CPU hybrid|t/s" "$OUT/$name.log" | tail -3 ;;
  *) sed -n 2,13p "$0"; exit 2 ;;
esac
