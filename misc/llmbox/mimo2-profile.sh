#!/usr/bin/env bash
# MiMo V2.6 Flash: routing profile, then the story run seeded from it.
#
# Tiers start from profile_seed, which ranks experts by the counts in a profile
# file. With no file it falls back to a uniform order that just interleaves
# layers, so every run measured so far began with the weakest placement the
# engine offers and never saw a profile at all.
#
# The profile is built from a decode-heavy pass, not a prefill-heavy one: the
# tiers are what decode pays for, and a 2203-token prefill touches nearly every
# expert of every layer, which flattens the ranking to noise.
#
#   mimo2-profile.sh OUTDIR
# Run inside the GPU window:
#   ~/claude-tmp/gpu-window.sh --max 1200 -- misc/llmbox/mimo2-profile.sh OUTDIR
set -u
OUT=${1:?usage: $0 OUTDIR}
mkdir -p "$OUT" || exit 2
[ -d "$OUT" ] || { echo "mimo2-profile: cannot write $OUT" >&2; exit 2; }
HERE=$(cd "$(dirname "$0")/../.." && pwd)
M=/srv/models/gguf/mimo/MiMo-V2.6-Flash-RL-Q2_K-00001-of-00002.gguf
P=${PROMPT2K:-$HOME/claude-tmp/ds4/prompt-2k.txt}
Q='Write a short story, about 150 words, about a lighthouse keeper who finds a message in a bottle.'
PROF=$OUT/mimo-profile.bin
N=${N:-512}
[ -s "$P" ] || { echo "mimo2-profile: no prompt at $P" >&2; exit 2; }

echo "=== build profile (prefill 2203 + decode $N)"
"$HERE/ds4-mimo2" -m "$M" -f "$P" --raw -n "$N" --ctx 8192 --ubatch 1024 \
  --profile-out "$PROF" ${SEEDARGS:-} > /dev/null 2> "$OUT/build.err"
grep -E "t/s|experts:|phases:|seeded" "$OUT/build.err"
ls -l "$PROF" 2>/dev/null || { echo "mimo2-profile: no profile written" >&2; exit 1; }

for seed in 0 1; do
  echo "=== story, profile seed=$seed"
  args=()
  # ${seed:+...} is true for the string "0", so the arm is chosen explicitly.
  [ "$seed" = 1 ] && args=(--profile "$PROF")
  "$HERE/ds4-mimo2" -m "$M" -p "$Q" -n 160 --ctx 4096 --cpu-experts 1 \
    "${args[@]}" > "$OUT/story-seed$seed.txt" 2> "$OUT/story-seed$seed.err"
  grep -E "t/s|experts:|phases:|seeded" "$OUT/story-seed$seed.err"
done
