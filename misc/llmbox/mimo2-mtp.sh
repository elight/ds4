#!/usr/bin/env bash
# MiMo V2.6 Flash MTP correctness and speed: the same greedy story with and
# without --mtp, for each CPU-expert setting. Greedy output with MTP must
# equal greedy output without it.
# Run inside the GPU window:
#   ~/claude-tmp/gpu-window.sh --max 600 -- misc/llmbox/mimo2-mtp.sh OUTDIR
# Env: CPUS="0 1" (CPU-expert settings), MODES="plain mtp3", N=128 (tokens),
# EXTRA=... (pass --slots N so every mode gets the same VRAM tiers).
# Modes: plain (any name starting "plain"), mtpK = K drafts, mtpKg70 = gate
# later drafts at p 0.70, mtpKg70f50 = also gate the first at 0.50. A mode
# whose output file exists is reused unless REUSE=0.
set -u
OUT=${1:?usage: $0 OUTDIR}
mkdir -p "$OUT"
HERE=$(cd "$(dirname "$0")/../.." && pwd)
M=/srv/models/gguf/mimo/MiMo-V2.6-Flash-RL-Q2_K-00001-of-00002.gguf
MTP=/srv/models/gguf/mimo/mtp-MiMo-V2.6-Flash-RL-Q8_0.gguf
Q='Write a short story, about 150 words, about a lighthouse keeper who finds a message in a bottle.'
N=${N:-128}

for cpu in ${CPUS:-0 1}; do
  for mode in ${MODES:-plain mtp3}; do
    f="$OUT/mtp-cpu$cpu-$mode"
    [ -s "$f.txt" ] && [ "${REUSE:-1}" = 1 ] && { echo "=== cpu-experts $cpu, $mode (reused)"; continue; }
    args=()
    case $mode in
      mtp*) d=${mode#mtp}; args=(--mtp "$MTP" --mtp-draft "${d%%[gf]*}")
            [[ $mode =~ g([0-9]+) ]] && args+=(--mtp-gate "0.${BASH_REMATCH[1]}")
            [[ $mode =~ f([0-9]+) ]] && args+=(--mtp-gate-first "0.${BASH_REMATCH[1]}") ;;
    esac
    echo "=== cpu-experts $cpu, $mode"
    "$HERE/ds4-mimo2" -m "$M" -p "$Q" -n "$N" --ctx 4096 --cpu-experts "$cpu" "${args[@]}" ${EXTRA:-} \
      > "$f.txt" 2> "$f.err"
    echo "rc=$?"
    grep -E "decode .*t/s|MTP|RAM->CPU|die|error|failed" "$f.err"
  done
  for mode in ${MODES:-plain mtp3}; do
    [ "$mode" = plain ] && continue
    a="$OUT/mtp-cpu$cpu-plain.txt" b="$OUT/mtp-cpu$cpu-$mode.txt"
    [ -f "$a" ] || continue
    if cmp -s "$a" "$b"; then
      echo "cpu-experts $cpu $mode: output IDENTICAL to plain greedy"
    else
      python3 - "$a" "$b" <<'EOF2'
import sys
a = open(sys.argv[1], errors="replace").read(); b = open(sys.argv[2], errors="replace").read()
n = 0
while n < min(len(a), len(b)) and a[n] == b[n]: n += 1
print(f"{sys.argv[2]}: DIFFERS from plain at char {n} of {len(a)}: plain {a[n:n+40]!r} vs mtp {b[n:n+40]!r}")
EOF2
    fi
  done
done
