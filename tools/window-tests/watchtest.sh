#!/usr/bin/env bash
# watchtest.sh — hermetic proof of the watchdog. No GPU, no gpud, no strata.
#
# Three things are stubbed (bench, gpu.sh, gpud) and the harness is killed the way
# a tool timeout does — SIGKILL, which runs no trap. Then it asserts what actually
# happened:
#   1. the bench (a stub that sleeps 300) is killed, not left holding the card
#   2. the wake call is made
#   3. the restore chat is issued
# Seconds to run, and it touches nothing real. The gpud stub listens on a port this
# script picks free at run time: 8099 is a live service (echo_owui_bridge.py).
set -uo pipefail
D=/tmp/watchtest

# Idempotent cleanup. Patterns are matched against other processes only: this
# script's own command line is "bash /tmp/watchtest.sh", so it cannot match itself
# (a pkill -f typed straight into a shell does match that shell — that bit me).
cleanup_strays() {
    local pat p
    for pat in 'watchtest/stub-bench.sh' 'watchtest/stub-server.py' 'watchtest/stub-gpu.sh'; do
        for p in $(pgrep -f "$pat" 2>/dev/null); do
            [ "$p" = "$$" ] && continue
            kill -9 "$p" 2>/dev/null
        done
    done
}
cleanup_strays
sleep 0.5
rm -rf "$D"; mkdir -p "$D/windows"

PORT=$(python3 -c "import socket;s=socket.socket();s.bind(('127.0.0.1',0));print(s.getsockname()[1]);s.close()")
echo "stub gpud port: $PORT"

cat > "$D/stub-bench.sh" <<'EOF'
#!/usr/bin/env bash
echo "stub bench up: $*"
sleep 300
EOF
chmod +x "$D/stub-bench.sh"

cat > "$D/stub-gpu.sh" <<EOF
#!/usr/bin/env bash
echo "\$(date +%H:%M:%S.%3N) gpu.sh \$*" >> $D/gpu-calls.log
case "\${1:-}" in
    json) echo '{"sleeping": true}' ;;
esac
exit 0
EOF
chmod +x "$D/stub-gpu.sh"

cat > "$D/stub-server.py" <<'EOF'
import http.server, sys
PORT = int(sys.argv[1])
OUT = sys.argv[2]
class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(n).decode('utf-8', 'replace')
        with open(OUT, 'a') as f:
            f.write(f"{self.path} {body[:80]}\n")
        out = b'{"choices":[{"message":{"content":"ok"}}]}'
        self.send_response(200); self.send_header('Content-Type','application/json')
        self.send_header('Content-Length', str(len(out))); self.end_headers()
        self.wfile.write(out)
    def log_message(self, *a): pass
http.server.HTTPServer(('127.0.0.1', PORT), H).serve_forever()
EOF

python3 "$D/stub-server.py" "$PORT" "$D/restore-requests.log" & SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
sleep 1

DS4_WINDOW_LOGDIR="$D/windows" GPU_SH="$D/stub-gpu.sh" DS4="$D/stub-bench.sh" \
GPUD_URL="http://127.0.0.1:$PORT" \
setsid bash -c 'exec /home/evan/src/ds4/tools/ds4-window.sh \
  --cuda -m "/srv/models/gguf/ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf" \
  --prompt-file /tmp/bench-prompt-32k.txt --ssd-streaming --ctx-max 2048 --gen-tokens 64' \
  > "$D/harness.log" 2>&1 &
HARNESS=$!

for _ in $(seq 1 40); do
    pgrep -f "$D/stub-bench.sh" >/dev/null 2>&1 && break
    sleep 0.5
done
if ! pgrep -f "$D/stub-bench.sh" >/dev/null 2>&1; then
    echo "FAIL: the stub bench never started. harness said:"; tail -5 "$D/harness.log"; exit 1
fi
echo "stub bench running: $(pgrep -f "$D/stub-bench.sh" | tr '\n' ' ')"
echo "watchdog present:   $(pgrep -f 'kill -0' | wc -l) process(es) polling liveness"

T0=$(date +%s.%N)
kill -9 "$HARNESS" 2>/dev/null
echo "KILL -9 harness=$HARNESS (no trap can run) -> stubs holding the card: $(pgrep -cf "$D/stub-bench.sh")"

for _ in $(seq 1 40); do
    [ "$(pgrep -cf "$D/stub-bench.sh")" = 0 ] && { T1=$(date +%s.%N); echo "bench KILLED by the watchdog after $(python3 -c "print(round($T1-$T0,2))")s"; break; }
    sleep 0.25
done
[ "$(pgrep -cf "$D/stub-bench.sh")" != 0 ] && { echo "FAIL: bench still running:"; pgrep -af "$D/stub-bench.sh" | head -2; }

for _ in $(seq 1 40); do
    grep -q "gpu.sh wake" "$D/gpu-calls.log" 2>/dev/null && { T2=$(date +%s.%N); echo "wake called after $(python3 -c "print(round($T2-$T0,2))")s"; break; }
    sleep 0.25
done
for _ in $(seq 1 40); do
    [ -s "$D/restore-requests.log" ] && { T3=$(date +%s.%N); echo "restore chat issued after $(python3 -c "print(round($T3-$T0,2))")s"; break; }
    sleep 0.25
done

echo "--- gpu.sh calls the window made ---"; cat "$D/gpu-calls.log" 2>/dev/null
echo "--- restore request the stub gpud received ---"; cat "$D/restore-requests.log" 2>/dev/null
echo "--- harness log (it ran no trap: SIGKILL) ---"; tail -3 "$D/harness.log"
echo "--- leftover stub processes (want none) ---"; pgrep -af "stub-bench|stub-gpu" 2>/dev/null | grep -v pgrep | head -3
echo "(end)"
