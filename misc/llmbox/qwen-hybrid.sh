#!/usr/bin/env bash
# Qwen3.8 Flash Next Q2 on llmbox (RTX 3090, i9-10900K, 64 GB): CPU/GPU hybrid
# decode measurements. Each case runs in its own GPU window, because the card
# is shared with live services (gpu-window.sh pauses gpud and always wakes it).
#
#   qwen-hybrid.sh bench <on|off> <ctx>     ds4-bench, prefill + 128 decode tokens
#   qwen-hybrid.sh ppl <on|off>             teacher-forced NLL over ~480 tokens
#   qwen-hybrid.sh greedy <on|off>          96 greedy tokens on a short prompt
#   qwen-hybrid.sh mtp <on|off> <ctx> [depth...]  128 greedy tokens per MTP draft depth
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
  mtp)
    # The story, cut to about <ctx> tokens, as one chat turn; 128 greedy tokens
    # per MTP depth, all in one GPU window. Depth 1 is plain decoding; 2 and 3
    # force DS4_QWEN4_MTP_DEPTH. Acceptance comes from DS4_QWEN4_SPEC_TRACE.
    ctx=${3:-8192}; shift 3; depths=${*:-1 2 3}
    name="mtp-$mode-$ctx$TAG"
    # The story file carries DeepSeek chat markers; the CLI applies Qwen's.
    cat tests/long_context_story_prompt.txt tests/long_context_story_prompt.txt |
      sed -e 's/<｜[^｜]*｜>//g' -e 's#</think>##g' | head -c $(( (ctx - 256) * 449 / 100 )) > "$OUT/story-$ctx.txt"
    printf '\n\nRetell the story above in detail, scene by scene.\n' >> "$OUT/story-$ctx.txt"
    "$W" --max 600 -- "$0" _mtprun "$mode" "$ctx" "$name" $depths
    for d in $depths; do
      log="$OUT/$name-d$d.log"
      printf 'depth %s: %s  cycles: accept %s reject %s plain %s\n' "$d" \
        "$(grep -o 'generation: [0-9.]* t/s' "$log")" \
        "$(grep -c 'draft.* accept' "$log")" "$(grep -c 'draft.* reject' "$log")" \
        "$(grep -c ' plain$' "$log")"
    done ;;
  _mtprun)
    ctx=$3; name=$4; shift 4
    for d in "$@"; do
      if [ "$d" = 1 ]; then mtp=(); else mtp=(--mtp); fi
      DS4_QWEN4_MTP_DEPTH=$d DS4_QWEN4_SPEC_TRACE=1 ./ds4 -m "$M" --cuda --ssd-streaming \
        -c $((ctx + 512)) "${mtp[@]}" --prompt-file "$OUT/story-$ctx.txt" -n 128 --temp 0 \
        > "$OUT/$name-d$d.out" 2> "$OUT/$name-d$d.log"
    done ;;
  *) sed -n 2,13p "$0"; exit 2 ;;
esac
