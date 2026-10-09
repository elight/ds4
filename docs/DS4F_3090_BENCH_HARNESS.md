# The bench window harness

`tools/ds4-window.sh`. Recorded 2026-10-09. It exists because a bench that died
mid-window left the card asleep and strata down until somebody remembered to wake
them, and the box sat unusable for minutes while the agent thought about it.

## What it guarantees

The wake is a shell trap, so it runs on **every** exit path: clean finish, CUDA
OOM, segfault, `timeout`, Ctrl-C, SIGTERM. The window is never longer than the
bench and never outlives it.

```text
trap 'wake $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
```

`wake` is guarded by a flag so it runs exactly once.

## Who stops and starts what

Measured, not assumed. gpud is the single writer for VRAM tenancy and the only
thing on this box with the sudo to touch tenant units:

| Route | What it does | Source |
|---|---|---|
| `gpu.sh sleep` | gpud runs `sudo -n systemctl stop` on every running tenant — strata's 22.4 GB VRAM **and** 52.9 GiB RAM (`systemctl show -p MemoryCurrent --value llmbox-strata` = 56728317952) are released | `gpud.py:2884` |
| `gpu.sh wake` | gpud re-admits tenants and runs `sudo -n systemctl start` on them | `gpud.py:2944` |
| gpud itself | never stops: sleep/wake are its authenticated HTTP control routes | `gpud.py:4455`, `gpud.py:4481`, ADR-097 |

So the harness calls `gpu.sh sleep` and `gpu.sh wake` and **nothing else**. A
plain `systemctl start llmbox-strata` from an agent shell fails:

```text
Failed to start llmbox-strata.service: Interactive authentication required.
```

Verified on this box. An early draft of the harness called `systemctl start` in
the wake path and logged `WARN llmbox-strata.service start failed` on every
window — noise, and it made the restore look broken when gpud had already
admitted strata. gpud brings strata back; the harness only checks that it did.

## Prompt sizing, which is what killed the first runs

`ds4-bench` requires `prompt tokens >= ctx_max`, plus `gen_tokens` when
teacher-forced decode is on (`ds4_bench.c:741`), and `ctx_alloc` is
`ctx_max + gen_tokens + 1` (`ds4_bench.c:415`). Defaults are `ctx_max = 32768`,
`gen_tokens = 128`, `ctx_start = 2048` (`ds4_bench.c:220-223`). A 90820-byte
prompt tokenises to 16690 tokens and the bench exits 1 after CUDA init:

```text
ds4-bench: prompt has 16690 tokens, need at least 32768
```

That failure costs a whole window if you only discover it after sleeping the card.
The harness preflights it:

- **Build**: repeat a corpus whose bytes-per-token is measured on this tokenizer
  (90820 B / 16690 tok = **5.4427 B/tok**) to `target_tokens x bpt` bytes.
  Repeating the same corpus keeps the ratio stable, so one probe lands it.
- **Probe**: run the bench with `--ctx-max 100000000 --gen-tokens 0`, which makes
  the length check fail on purpose and prints the exact count before any work is
  done. Out of band → rebuild from the measured ratio and re-probe.

Measured: target 32800 tokens → 178520 bytes → **32797 tokens**, inside the
±120 band on the first probe.

## Measured windows

Crash path, the case the harness is for. `--ctx-max 200000` against a 32797-token
prompt, so the bench exits 1 right after CUDA init:

```text
ds4-bench: prompt has 32797 tokens, need at least 200000
ds4-window: bench rc=1 log=/srv/models/gguf/ds4/window-logs/20261009-031236.log
ds4-window: bench exited (rc=1) -> waking gpud now
ds4-window: restored: gpud awake, llmbox-strata.service active after 40s
real	0m42.745s
```

`gpu.sh wake` returned `{"sleeping": false, "admitted": []}` in the same second
the bench died; strata was active 40 s later. Whole window 43 s, start to
restored.

Probe path, same trap, `real 0m2.566s` end to end, gpud awake and strata
re-admitted (`"admitted": ["strata"]`) on exit.

## Usage

```bash
# baseline / any lever
~/claude-tmp/ds4-window.sh --cuda -m "$MODEL" --prompt-file /tmp/bench-prompt-32k.txt \
    --ssd-streaming --ctx-max 32768 --gen-tokens 128

# just the token count, no window spent
~/claude-tmp/ds4-window.sh --probe --cuda -m "$MODEL" --prompt-file /tmp/bench-prompt-32k.txt
```

Env overrides: `DS4_WINDOW_TOKENS` (32800), `DS4_WINDOW_BAND` (120),
`DS4_WINDOW_BPT` (5.4427), `DS4_WINDOW_CORPUS`, `DS4_WINDOW_TIMEOUT` (3600),
`DS4_WINDOW_VERIFY_SECS` (150), `DS4_WINDOW_LOGDIR`
(`/srv/models/gguf/ds4/window-logs`). Every window's full output is teed to a
timestamped log there.
