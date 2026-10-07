#!/usr/bin/env bash
# MiMo V2.6 Flash parity and decode baseline: the same greedy 160-token story
# from ds4-mimo2 with the CPU expert split on and off, then from llama.cpp.
# Run inside the GPU window:
#   ~/claude-tmp/gpu-window.sh --max 600 -- misc/llmbox/mimo2-story.sh OUTDIR
# Prints each run's t/s and how many leading words of each ds4 story match
# llama.cpp's.
set -u
OUT=${1:?usage: $0 OUTDIR}
mkdir -p "$OUT" || exit 2
[ -d "$OUT" ] || { echo "mimo2-story: cannot write $OUT" >&2; exit 2; }
HERE=$(cd "$(dirname "$0")/../.." && pwd)
M=/srv/models/gguf/mimo/MiMo-V2.6-Flash-RL-Q2_K-00001-of-00002.gguf
LLAMA=/home/evan/src/llama.cpp/build/bin/llama-completion
Q='Write a short story, about 150 words, about a lighthouse keeper who finds a message in a bottle.'
RAW="<|im_start|>user
${Q}<|im_end|><|im_start|>assistant
<think></think>"
N=${N:-160}

for cpu in ${CPUS:-1 0}; do
  echo "=== ds4 cpu-experts $cpu"
  "$HERE/ds4-mimo2" -m "$M" -p "$Q" -n "$N" --ctx 4096 --cpu-experts "$cpu" ${EXTRA:-} \
    > "$OUT/story-ds4-cpu$cpu.txt" 2> "$OUT/story-ds4-cpu$cpu.err"
  grep -E "t/s|RAM->CPU|experts:" "$OUT/story-ds4-cpu$cpu.err"
done
if [ "${LLAMA_RUN:-1}" = 1 ]; then
  echo "=== llama.cpp"
  "$LLAMA" -m "$M" -p "$RAW" -n "$N" --temp 0 -ngl 99 -ot exps=CPU -c 4096 -t 10 --no-warmup -no-cnv \
    > "$OUT/story-llama.txt" 2> "$OUT/story-llama.err"
  grep -E "eval time" "$OUT/story-llama.err"
fi

# Leading words in common with llama.cpp's continuation (its stdout echoes the prompt).
python3 - "$OUT" <<'EOF'
import sys, pathlib
d = pathlib.Path(sys.argv[1])
ref = (d / "story-llama.txt")
if not ref.exists(): sys.exit()
r = ref.read_text(errors="replace").split("<think></think>")[-1].split()
for f in sorted(d.glob("story-ds4-cpu*.txt")):
    w = f.read_text(errors="replace").split()
    n = 0
    while n < min(len(w), len(r)) and w[n] == r[n]: n += 1
    print(f"{f.name}: {n} of {len(r)} llama words match before the first difference")
EOF
