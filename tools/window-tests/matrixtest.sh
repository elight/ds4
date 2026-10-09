#!/usr/bin/env bash
# matrixtest.sh — hermetic exit-path matrix. No GPU, no gpud, no strata.
# For each way a window can end: the harness exit code, whether gpu.sh wake was
# called and how long after the window opened, and whether any bench is left.
set -uo pipefail
D=/tmp/matrixtest
H=/home/evan/src/ds4/tools/ds4-window.sh
rm -rf "$D"; mkdir -p "$D"
PORT=$(python3 -c "import socket;s=socket.socket();s.bind(('127.0.0.1',0));print(s.getsockname()[1]);s.close()")
cat > "$D/srv.py" <<'EOF'
import http.server, sys
class H(http.server.BaseHTTPRequestHandler):
    def _ok(self):
        b=b'{"ok":1}'; self.send_response(200); self.send_header('Content-Length',str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self): self._ok()
    def do_POST(self): self.rfile.read(int(self.headers.get('Content-Length',0))); self._ok()
    def log_message(self,*a): pass
http.server.ThreadingHTTPServer(('127.0.0.1',int(sys.argv[1])),H).serve_forever()
EOF
python3 "$D/srv.py" "$PORT" & SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
sleep 1

# $1 case name, $2 sleep exit code, $3 bench body, $4 run timeout
run_case() {
    local name=$1 sleep_rc=$2 body=$3 tmo=$4 C="$D/$1"
    mkdir -p "$C/w"
    cat > "$C/gpu.sh" <<EOF
#!/usr/bin/env bash
echo "\$(date +%s.%N) \$1" >> $C/calls
[ "\$1" = sleep ] && exit $sleep_rc
exit 0
EOF
    chmod +x "$C/gpu.sh"
    printf '#!/usr/bin/env bash\n%s\n' "$body" > "$C/bench-$name.sh"; chmod +x "$C/bench-$name.sh"
    local t0 rc
    t0=$(date +%s.%N)
    DS4_WINDOW_LOGDIR="$C/w" GPU_SH="$C/gpu.sh" DS4="$C/bench-$name.sh" \
    GPUD_URL="http://127.0.0.1:$PORT" DS4_WINDOW_STRATA_HEALTH="http://127.0.0.1:$PORT/health" \
    STRATA_UNIT=no-such-unit.service DS4_WINDOW_TIMEOUT="$tmo" \
        timeout 60 "$H" --cuda -m /srv/models/gguf/ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
        --prompt-file /tmp/bench-prompt-32k.txt --ctx-max 2048 > "$C/out" 2>&1
    rc=$?
    local wake_at left
    wake_at=$(awk '$2=="wake"{print $1; exit}' "$C/calls" 2>/dev/null)
    left=$(pgrep -cf "$C/bench-$name.sh" 2>/dev/null)
    printf '%-14s rc=%-4s calls=%-28s wake=%-7s leftover-bench=%s\n' "$name" "$rc" \
        "$(awk '{printf "%s,", $2}' "$C/calls" 2>/dev/null)" \
        "$( [ -n "$wake_at" ] && python3 -c "print(f'{$wake_at-$t0:.2f}s')" || echo NONE)" "${left:-0}"
}

echo "case           exit  gpu.sh calls in order         wake after window opened"
run_case clean-exit    0 'echo "2048,2048,1,16,1,1,15,1,0"; exit 0'  60
run_case bench-crash   0 'kill -SEGV $$'                             60
run_case never-starts  0 'exec /no/such/binary'                      60
run_case bench-hang    0 'sleep 300'                                 3
run_case sleep-fails   1 'sleep 300'                                 60
echo "--- what the harness said when sleep failed ---"; grep -E "sleep failed|waking" "$D/sleep-fails/out" | head -3
echo "--- leftover stub benches anywhere (want none) ---"; pgrep -af "matrixtest/.*/bench-" | head -3
