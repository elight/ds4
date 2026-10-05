#!/usr/bin/env bash
# Strata Q2_0 head-to-head for qwen-hybrid.sh: same box, same story prompt, same
# context frontiers (8K and 32K), 128 generated tokens at temperature 0.
#
#   strata-bench.sh spec     one GPU window: start Strata, measure, kill it
#
# spec = Strata's shipped config (MTP speculative decoding, --spec 4), without
#        vision. There is no no-MTP variant: `strata serve` on a native pack
#        refuses to start without --spec and --mtp. Its log line "drafts
#        accepted A of D" gives tokens per verify pass = gen / (gen - A).
#
# Output: $OUT/strata-<variant>.result.json and .log (default ~/claude-tmp/ds4-qwen-hybrid).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
S=${S:-/srv/models/gguf/strata}
W=${W:-$HOME/claude-tmp/gpu-window.sh}
OUT=${OUT:-$HOME/claude-tmp/ds4-qwen-hybrid}
PORT=${PORT:-8090}
mkdir -p "$OUT"

if [ "${1:-}" != inner ]; then
  v=${1:-spec}
  case "$v" in spec) ;; *) sed -n 2,11p "$0"; exit 2 ;; esac
  python3 - "$S/strata-q2_0.json" "$OUT/strata-$v.json" "$v" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c["host"] = "127.0.0.1"
c.pop("vision", None)
a, out, skip = c["args"], [], 0
for x in a:
    if skip: skip -= 1; continue
    if x == "--vision": continue
    out.append(x)
c["args"] = out
json.dump(c, open(sys.argv[2], "w"), indent=1)
PY
  cat "$ROOT/tests/long_context_story_prompt.txt" "$ROOT/tests/long_context_story_prompt.txt" > "$OUT/story-x2.txt"
  exec "$W" --max 600 -- "$0" inner "$v"
fi

v=$2
SRV=""
cleanup() {
  [ -n "$SRV" ] && kill "$SRV" 2>/dev/null
  sleep 3
  pkill -f "serve/server.py.*--port $PORT" 2>/dev/null
  pkill -f "$S/engine/strata " 2>/dev/null
  sleep 2
}
trap cleanup EXIT
cd "$S" || exit 1
export TMPDIR="$S/Strata-data/tmp"
setsid .venv/bin/python serve/server.py --engine strata --config "$OUT/strata-$v.json" \
  --port "$PORT" > "$OUT/strata-$v.log" 2>&1 < /dev/null &
SRV=$!
for _ in $(seq 120); do
  curl -sf -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status": *"ok"' && break
  kill -0 "$SRV" 2>/dev/null || { echo "strata exited"; tail -20 "$OUT/strata-$v.log"; exit 1; }
  sleep 3
done
.venv/bin/python "$HERE/strata_bench.py" --port "$PORT" --story "$OUT/story-x2.txt" \
  --out "$OUT/strata-$v.result.json"
