#!/usr/bin/env bash
set -u
M=/srv/models/gguf/mimo/MiMo-V2.6-Flash-RL-Q2_K-00001-of-00002.gguf
WT=$(cd "$(dirname "$0")/../.." && pwd)
S=${S:-$HOME/claude-tmp/ds4-mimo2}; mkdir -p "$S"
for t in ${TEXTS:-llama ds4}; do
  echo "=== $t text"
  "$WT/ds4-mimo2" -m "$M" -f "$S/score-$t.txt" --raw --score --score-from 34 --ctx 4096 --profile "$S/profile.bin" ${EXTRA:-} 2>&1 | grep -v "^ds4-mimo2: \(non-routed\|tiers\|seeded\)"
done
