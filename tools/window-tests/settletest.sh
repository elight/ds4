#!/usr/bin/env bash
# settletest.sh — hermetic: no GPU, no gpud, no strata.
#  A. a window refuses to sleep the card while an earlier restore holds the lock
#  B. two restores at once: the second stands down, the first gets a 200
set -uo pipefail
D=/tmp/settletest
rm -rf "$D"; mkdir -p "$D/windows"
H=/home/evan/src/ds4/tools/ds4-window.sh
PORT=$(python3 -c "import socket;s=socket.socket();s.bind(('127.0.0.1',0));print(s.getsockname()[1]);s.close()")

cat > "$D/stub-gpu.sh" <<EOF
#!/usr/bin/env bash
echo "\$(date +%H:%M:%S) gpu.sh \$*" >> $D/gpu-calls.log
exit 0
EOF
chmod +x "$D/stub-gpu.sh"
printf '#!/usr/bin/env bash\nsleep 300\n' > "$D/stub-bench.sh"; chmod +x "$D/stub-bench.sh"

cat > "$D/srv.py" <<'EOF'
import http.server, sys, time
class H(http.server.BaseHTTPRequestHandler):
    def _send(self, code, body=b'{"ok":1}'):
        self.send_response(code); self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self): self._send(200)
    def do_POST(self):
        self.rfile.read(int(self.headers.get('Content-Length', 0)))
        with open(sys.argv[2], 'a') as f: f.write(time.strftime('%H:%M:%S') + " POST\n")
        time.sleep(2); self._send(200)
    def log_message(self, *a): pass
http.server.ThreadingHTTPServer(('127.0.0.1', int(sys.argv[1])), H).serve_forever()
EOF
python3 "$D/srv.py" "$PORT" "$D/posts.log" & SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
sleep 1
export DS4_WINDOW_LOGDIR="$D/windows" GPU_SH="$D/stub-gpu.sh" DS4="$D/stub-bench.sh" \
       GPUD_URL="http://127.0.0.1:$PORT" DS4_WINDOW_STRATA_HEALTH="http://127.0.0.1:$PORT/health" \
       STRATA_UNIT=no-such-unit.service DS4_WINDOW_SETTLE_TIMEOUT=6

echo "== A: an earlier restore holds the lock =="
setsid flock "$D/windows/.restore.lock" sleep 20 & HOLDER=$!
sleep 0.5
T0=$(date +%s)
"$H" --cuda -m /srv/models/gguf/ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
     --prompt-file /tmp/bench-prompt-32k.txt --ctx-max 2048 2>&1 | tail -3
echo "rc=${PIPESTATUS[0]} after $(( $(date +%s) - T0 ))s"
echo "gpu.sh calls during A (want none): $(cat "$D/gpu-calls.log" 2>/dev/null | wc -l)"
# flock's child inherits the locked fd, so killing flock alone leaves the lock held:
# kill the whole session.
kill -- -"$HOLDER" 2>/dev/null; wait $HOLDER 2>/dev/null; sleep 0.3

echo "== B: two restores at once =="
L1=$("$H" --restore); L2=$(DS4_WINDOW_STAMP=second "$H" --restore)
sleep 5
echo "-- first  ($L1)"; cat "$L1"
echo "-- second ($L2)"; cat "$L2"
echo "POSTs the stub gpud received (want 1): $(wc -l < "$D/posts.log")"
