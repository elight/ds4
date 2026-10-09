#!/usr/bin/env bash
# repeat.sh — run N bench windows of ONE configuration back to back through
# ds4-window.sh, then report each run's rows and the median and spread per context
# size. Each window waits for the previous window's strata restore to finish
# (settle_before_sleep), so the runs never overlap and strata is never stranded.
#
#   tools/window-tests/repeat.sh N OUT_DIR [ds4-bench args...]
set -uo pipefail
N=${1:?runs}; OUT=${2:?out dir}; shift 2
H=/home/evan/src/ds4/tools/ds4-window.sh
mkdir -p "$OUT"
: > "$OUT/runs.txt"
for i in $(seq 1 "$N"); do
    echo "=== run $i of $N, $(date +%H:%M:%S) ===" | tee -a "$OUT/progress.txt"
    "$H" "$@" > "$OUT/run-$i.out" 2>&1
    rc=$?
    log=$(sed -n 's/.*bench rc=[0-9]* log=\(.*\)$/\1/p' "$OUT/run-$i.out" | tail -1)
    echo "run $i rc=$rc log=$log" | tee -a "$OUT/progress.txt"
    [ -n "$log" ] && grep -E '^[0-9]+,[0-9]+,' "$log" | sed "s/^/$i,/" >> "$OUT/runs.txt"
done
python3 - "$OUT/runs.txt" <<'PY' | tee "$OUT/summary.txt"
import sys, statistics as st
rows = [l.strip().split(',') for l in open(sys.argv[1]) if l.strip()]
# run,ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,gen_steady_tokens,gen_steady_tps,kv
by = {}
for r in rows:
    by.setdefault(int(r[1]), []).append((int(r[0]), float(r[3]), float(r[5])))
print(f"{'ctx':>6} {'runs':>4}  {'prefill t/s per run':<26} {'median':>7} {'spread':>7}   {'decode t/s per run':<20} {'median':>6} {'spread':>6}")
for ctx in sorted(by):
    v = by[ctx]
    pf = [x[1] for x in v]; dc = [x[2] for x in v]
    def spread(a): return f"{(max(a)-min(a))/st.median(a)*100:.1f}%"
    print(f"{ctx:>6} {len(v):>4}  {', '.join(f'{x:.2f}' for x in pf):<26} {st.median(pf):>7.2f} {spread(pf):>7}   "
          f"{', '.join(f'{x:.2f}' for x in dc):<20} {st.median(dc):>6.2f} {spread(dc):>6}")
PY
echo "=== done $(date +%H:%M:%S) ===" | tee -a "$OUT/progress.txt"
