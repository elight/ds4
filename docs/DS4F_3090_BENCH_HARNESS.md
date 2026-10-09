# The bench window harness

`tools/ds4-window.sh`, rewritten 2026-10-09 after three failures in one evening.
This file is the failure matrix and the contract; the script is the fix.

## Why a window has to be short and sure

When pi runs the local model (`~/.pi/agent/settings.json`: `defaultProvider=gpud`,
`defaultModel=swift-1.5-iq3_xxs`), strata **is the agent's brain**. Sleeping gpud to
free the card stops the agent's own inference, so the agent cannot act again until
the card is back — it cannot notice the bench crashed, cannot wake anything, cannot
even report that it is stuck. Every choice below follows from that one fact.

## What went wrong, measured

| # | Failure | Mechanism | Evidence |
| --- | --- | --- | --- |
| 1 | A prompt below the floor was discovered **after** the card was slept | `ds4-bench` requires prompt tokens >= `ctx_max` (+ `gen_tokens` when teacher-forced), `ds4_bench.c:745`. A 90820-byte prompt is 16690 tokens | `ds4-bench: prompt has 16690 tokens, need at least 32768`, exit 1, after `gpu.sh sleep` had already stopped strata |
| 2 | `kill -9` on the harness skipped the trap, and an orphaned bench held the card for 59 s | SIGKILL runs no shell trap. The bench is a separate process and outlived the shell that started it | First kill test: gpud stayed asleep until the bench finished on its own, 59 s later |
| 3 | The exit path slept for 150 s and could never have succeeded | It polled `systemctl is-active llmbox-strata`. gpud does **not** start strata on wake — it starts it when a request arrives | `/gpu/wake` at 04:00:25 with no job waiting: strata still inactive 200 s later. Both observed starts (03:54:21, 03:55:54) followed an arriving job |
| 4 | The hygiene check matched **pgrep** | `pgrep -f "$DS4"` matches any process whose argv contains the path — including a concurrent pgrep, whose argv *is* the pattern | The harness refused a window because `824487 pgrep -f /tmp/watchtest/stub-bench.sh` was in the table. Chased as a phantom bench for a whole test cycle |
| 5 | The process-group kill did not land, and nothing recorded why | The first watchdog resolved the bench's group with `pgrep -P`. The group kill never took effect and the bench survived. That version logged nothing, so the cause is unproven | Stub bench and its `timeout` parent alive after the watchdog had already woken gpud; no log to say which kill ran |
| 6 | The restore's outcome log stayed empty | The chat blocks while the engine loads (40-70 s); a process killed in that window wrote nothing, so a window looked like it never asked | `restore-20261009-042012.log` absent while gpud's journal showed the request arriving at 04:21:35 |

## The design: two wake paths, deliberately overlapping

```text
trap on EXIT/INT/TERM   fires in the same second the bench exits
detached watchdog       survives SIGKILL; bench gone -> wake gpud;
                        harness gone -> kill the bench group, then wake gpud
```

Both also issue the restore, so a window that is killed still leaves strata coming
back. A double wake is harmless: gpud's routes are idempotent.

The four rules that came out of the failures:

1. **Nothing sleeps in the exit path.** gpud is woken, the restore is issued
   detached, and the shell exits. The window is never longer than the bench.
2. **The wake survives being killed.** A watchdog in its own session, because a
   SIGKILLed shell cannot run a trap and a trap is therefore not a guarantee.
3. **The bench's process group is known, not guessed.** The session leader writes
   its own PID; that PID *is* the group. `kill` decisions are logged with statuses.
4. **Strata comes back the way it actually comes back.** gpud is the single writer
   for VRAM tenancy (`gpud.py:2884` stops tenants, `gpud.py:2944` starts them) and
   it is the only thing on this box with the sudo to touch tenant units, so the
   harness calls `gpu.sh` and nothing else — a plain `systemctl start
   llmbox-strata` from an agent shell fails with `Interactive authentication
   required`. And because gpud starts strata on demand, the restore is a real
   request (a 1-token completion to `swift-1.5-iq3_xxs`), not a status poll.

## Measured proofs

**Watchdog, hermetic** — no GPU, no gpud, stubbed bench/gpu.sh/gpud, harness SIGKILLed
(`/tmp/watchtest.sh`). The watchdog's own log, verbatim:

```text
04:19:51.659 watchdog start: harness=834093 leader=834131 pgid=834131
04:19:52.661 harness 834093 gone -> TERM bench group -834131
04:19:52.662   TERM rc=0
04:19:52.663   bench gone after TERM
04:19:52.664 waking gpud
04:19:52.668   wake rc=0
04:19:52.674   restore rc=0
04:19:52.675 watchdog done
```

Bench killed **0.59 s** after the kill, wake at **0.62 s**, restore at **0.64 s**,
no leftover processes. This is the 59-second stranding from failure 2, now under a
second.

**A real window** (ctx 4096, gen 32, `--ssd-streaming`):

```text
04:21:30 ds4-window: bench rc=0 log=/srv/models/gguf/ds4/window-logs/20261009-042012.log
04:21:30 ds4-window: bench gone (rc=0) -> waking gpud
04:21:30 ds4-window: gpud awake. strata restore requested; check it with: --status
```

Same second, and gpud's own journal agrees: `gpud waking up` 04:21:30,
`starting strata (24126 MiB free)` 04:21:31, the completion POST at 04:21:35.
strata was active again ~5 s after the window ended. Whole window 1m20s: prefill
106.90 t/s at 2048 and 134.58 at 4096, decode 1.50 and 1.49 t/s.

**The floor abort** — the 16690-token file against a 32768 floor:

```text
ds4-window: prompt has 16690 tokens, the bench needs 32768 — the card was NOT touched
rc=2, and 0 /gpu/sleep or /gpu/wake calls on the journal
```

The claim is checked by counting gpud's own control calls, not by trusting the
script's message.

**Restore on demand** — gpud awake plus a 1-token chat: strata active at t+55 s
from a cold start (`restore-*.log`: `http=200 body=yes`).

## ds4-bench's contract, from the source

Each reference opened and verified, not recalled:

| Fact | Reference |
| --- | --- |
| Prompt tokens must be >= `ctx_max`, plus `gen_tokens` when teacher-forced | `ds4_bench.c:745` |
| `ctx_alloc = ctx_max + gen_tokens + 1` when not set | `ds4_bench.c:415` |
| Defaults: `ctx_start` 2048, `ctx_max` 32768, `gen_tokens` 128 | `ds4_bench.c:220`, `:221`, `:223` |
| `DS4_CUDA_STREAM_EXPERT_RESERVE_MB` lowers the streaming reserve; default 8 GiB | `ds4_cuda.cu:27198`, `:27199` |
| `DS4_CUDA_Q8_F16_CACHE_MB` bounds the Q8->F16 staging cache | `ds4_cuda.cu:1488` |
| The cache size the run actually chose prints as `CUDA SSD expert cache: N slots, X GiB` | `ds4_cuda.cu:27225` |
| On a 24 GB card the guard reserve scales to `base_gib / 8` (from `5a697a8`) | `ds4.c:44938` |

## Usage

```bash
# a window: the harness checks the floor, sleeps gpud, runs the bench, wakes gpud,
# issues the restore, and exits
~/claude-tmp/ds4-window.sh --cuda -m "$MODEL" --prompt-file /tmp/bench-prompt-32k.txt \
    --ssd-streaming --ctx-max 32768 --gen-tokens 128

# the floor without touching the card
~/claude-tmp/ds4-window.sh --floor /tmp/bench-prompt-32k.txt

# what the last window left behind
~/claude-tmp/ds4-window.sh --status
```

A verified token count is cached beside the prompt (`.floor`, keyed on size +
sha256), so the count is probed once and later windows need no card for it. An
unrecorded prompt is probed inside the window before the bench does any work.

Env overrides: `DS4_WINDOW_T0KENS` (`DS4_WINDOW_TOKENS`, default 32800),
`DS4_WINDOW_BAND` (120), `DS4_WINDOW_BPT` (5.4427), `DS4_WINDOW_CORPUS`,
`DS4_WINDOW_TIMEOUT` (3600), `DS4_WINDOW_RESTORE_TIMEOUT` (300),
`DS4_WINDOW_WATCH_INTERVAL` (1), `DS4_WINDOW_LOGDIR`
(`/srv/models/gguf/ds4/window-logs`), `GPU_SH`, `DS4`, `GPUD_URL`, `GPUD_MODEL`.

Each window writes three files: the bench log (`<stamp>.log`), the restore log
(`restore-<stamp>.log`, with the attempt and the outcome) and the watchdog's
decisions (`watchdog-<stamp>.log`).
