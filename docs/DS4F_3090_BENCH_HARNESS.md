# The bench window harness

`tools/ds4-window.sh`. Rewritten 2026-10-09 after the failures below, each
reproduced and then fixed. This file is the failure matrix, the proofs, and
ds4-bench's own contract; the script is the fix.

## Why a window has to be short and sure

When pi runs the local model (`~/.pi/agent/settings.json`: `defaultProvider=gpud`,
`defaultModel=swift-1.5-iq3_xxs`), strata **is the agent's brain**. Sleeping gpud to
free the card stops the agent's own inference, so for the length of the window the
agent cannot notice a crash, wake anything, or say that it is stuck. Whatever ends
the window has to give the card back by itself.

## Who stops and starts what

gpud is the single writer for VRAM tenancy and the only thing on this box with the
sudo to touch tenant units.

| Route | What it does | Source |
| --- | --- | --- |
| `gpu.sh sleep` | gpud stops every running tenant: strata's 22.4 GB of VRAM and 52.9 GiB of RAM are released | `gpud.py:2884` |
| `gpu.sh wake` | gpud reopens the card. It does **not** start strata | `gpud.py:4481` |
| a chat to `swift-1.5-iq3_xxs` | gpud starts strata on demand and holds the request until it is served | `gpud.py:2944`, `gpud.py:3136` (`ensure_up`) |

A plain `systemctl start llmbox-strata` from an agent shell fails with
`Interactive authentication required`, so the harness calls `gpu.sh` and sends a
chat, and nothing else.

## The failure matrix

Every way a window can end or go wrong, what used to happen, and what happens now.
"Proof" names the measurement in the next section.

| Way it ends | What went wrong before | Mechanism | What happens now | Proof |
| --- | --- | --- | --- | --- |
| Prompt below the floor | Found **after** the card was slept; window wasted | `ds4_bench.c:745` needs prompt tokens >= `ctx_max`; a 90820-byte prompt is 16690 tokens | Refused before the card is touched, from a token count cached beside the prompt (size + sha256) | P1 |
| Clean exit | Exit path polled `systemctl is-active` for 150 s and could never succeed | gpud starts strata on a request, not on wake (wake 04:00:25, strata inactive 200 s later) | Trap wakes gpud the same second; a detached restore sends the chat | P2 |
| Bench crash | Same 150 s tail | Same | Same trap; covered with fakes for SIGSEGV | P3 |
| Bench never starts | Same | Same | Same trap, exit 127 | P3 |
| Bench hangs | Nothing stopped it | No run limit | `timeout` ends it (default 3600 s); trap wakes gpud | P3 |
| Sleep call fails | Script died without waking | `die` with no trap | Trap is armed before the sleep, so a failed sleep still wakes | P3 |
| Harness `kill -9` | Trap skipped; orphaned bench held the card for its remaining 59 s | SIGKILL runs no trap; the bench outlived its shell | Watchdog in its own session sees the harness gone, kills the bench, wakes gpud, restores | P4 |
| Tool-timeout group kill | Same as `kill -9` | Same | Bench runs in its own session, so the group kill misses it; the watchdog catches it | P5 |
| Sleep lands while strata is loading | Strata down for minutes | gpud's `start()` waits out strata's 300 s start deadline (`vram-budget.json:997`) for an engine the sleep stopped; start 04:31:50, sleep 04:32:40, "never became healthy" 04:36:50 | Window waits for strata's `/health` to answer 200 before sleeping; refuses after 180 s without touching the card | P6, P7 |
| A restore from an earlier window still in flight | Restores piled up three deep, each re-asking for strata, and one fought the next window's sleep (503 "could not be started") | No coordination between windows | Restores run under a lock; one at a time; the next window waits for it | P6, P7 |
| A bench already running | Second window would fight the first | — | Refused, naming the process it found | P8 |
| The hygiene check matched itself | Windows refused over a phantom bench | `pgrep -f` matches any argv containing the path, pgrep's own included | Only real bench processes count | P8 |
| The restore counted "unit active" as done | Follow-up got 429 from an engine still loading | Active means launched, ~60 s before loaded | Success is a served completion (http=200); 429 counts only with `/health` at 200 | P2 |

## Measured proofs

Times in the "gpud" column are gpud's own journal lines, `journalctl -u
llmbox-gpud`.

**P1 — floor refused before the card.** The 16690-token prompt against a 32768
floor:

```text
ds4-window: prompt has 16690 tokens, the bench needs 32768 — the card was NOT touched
rc=2; 0 /gpu/sleep or /gpu/wake calls on gpud's journal since the attempt
```

**P2 — clean exit on the card** (ctx 2048, gen 16, `--ssd-streaming`):

| Event | Time | Source |
| --- | --- | --- |
| gpud going to sleep | 04:42:01 | gpud |
| bench rc=0 | 04:42:32 | harness |
| gpud waking up | 04:42:32 | gpud |
| starting strata | 04:42:32 | gpud |
| strata ready | 04:43:22 | gpud |
| restore served, http=200 | 04:43:24 | `restore-20261009-044201.log` |

Wake in the same second as the exit; the local model answering 52 s after it.

**P3 — the other exit paths, with fakes** (`tools/window-tests/matrixtest.sh`: stub bench, stub
`gpu.sh`, stub gpud; no card):

| Case | Harness exit | gpu.sh calls | Wake after the window opened | Bench left |
| --- | --- | --- | --- | --- |
| clean exit | 0 | sleep, wake | 0.06 s | 0 |
| bench crash (SIGSEGV) | 139 | sleep, wake | 1.15 s | 0 |
| bench never starts | 127 | sleep, wake | 0.05 s | 0 |
| bench hangs (limit 3 s) | 124 | sleep, wake | 3.05 s | 0 |
| sleep call fails | 2 | sleep, wake | 0.04 s | 0 |

**P4 — `kill -9` on the harness only, on the card** (`tools/window-tests/realkill.sh sock`; with fakes, `tools/window-tests/watchtest.sh`). The watchdog's own log:

```text
04:46:03.338 watchdog start: harness=912715 leader=914298 pgid=914298
04:46:12.346 harness 912715 gone -> TERM bench group -914298
04:46:12.850   bench gone after TERM
04:46:12.945   wake rc=0
```

gpud journal: `gpud waking up` 04:46:12. Restore served http=200 at 04:47:06. No
bench left.

**P5 — process-group kill, on the card** (`tools/window-tests/realkill.sh group`, what a tool timeout does): bench gone
0.56 s after the kill, gpud awake 0.97 s after it; gpud journal `gpud waking up`
04:45:06; restore served http=200 at 04:45:59.

**P6 — the settle wait, on the card.** The window after P5 started while P5's
restore was still loading strata:

```text
04:45:16 ds4-window: waiting for the card to settle before sleeping: a restore from an earlier window is still in flight
04:46:01 ds4-window: card settled after 45s
```

P5's restore was served at 04:45:59; the next sleep landed at 04:46:03. Nothing
was stranded.

**P7 — settle and the restore lock, with fakes** (`tools/window-tests/settletest.sh`): with the
restore lock held, a window refused after 10 s with **0** gpu.sh calls; two
restores started together sent **one** request, and the second logged that it
stood down.

**P8 — hygiene.** A bench already running is refused with the matching process
printed. The check that once matched its own `pgrep` now excludes it; the
phantom it produced (`824487 pgrep -f /tmp/watchtest/stub-bench.sh`) no longer
blocks a window.

## ds4-bench's contract, from the source

Each reference opened and checked:

| Fact | Reference |
| --- | --- |
| Prompt tokens must be >= `ctx_max`, plus `gen_tokens` when teacher-forced | `ds4_bench.c:745` |
| `ctx_alloc = ctx_max + gen_tokens + 1` when not set | `ds4_bench.c:415` |
| Defaults: `ctx_start` 2048, `ctx_max` 32768, `gen_tokens` 128 | `ds4_bench.c:220`, `:221`, `:223` |
| `DS4_CUDA_STREAM_EXPERT_RESERVE_MB` lowers the streaming reserve; default 8 GiB | `ds4_cuda.cu:27198`, `:27199` |
| `DS4_CUDA_Q8_F16_CACHE_MB` bounds the Q8->F16 staging cache | `ds4_cuda.cu:1488` |
| The cache the run chose prints as `CUDA SSD expert cache: N slots, X GiB` | `ds4_cuda.cu:27225` |
| On a 24 GB card the guard reserve scales to `base_gib / 8` (from `5a697a8`) | `ds4.c:44938` |

## Usage

```bash
# a window: check the floor, wait for the card to settle, sleep gpud, run the
# bench, wake gpud, send the restore chat, exit
~/claude-tmp/ds4-window.sh --cuda -m "$MODEL" --prompt-file /tmp/bench-prompt-32k.txt \
    --ssd-streaming --ctx-max 32768 --gen-tokens 128

~/claude-tmp/ds4-window.sh --floor /tmp/bench-prompt-32k.txt   # no card needed
~/claude-tmp/ds4-window.sh --status                            # strata, gpud, last restore
~/claude-tmp/ds4-window.sh --restore                           # ask gpud for strata now
```

A prompt with no recorded count is probed once inside the window, before the
bench does any work, and the count is cached for later windows.

Each window writes the bench log (`<stamp>.log`), the restore log
(`restore-<stamp>.log`) and the watchdog's decisions (`watchdog-<stamp>.log`) to
`/srv/models/gguf/ds4/window-logs`.

Overrides: `DS4_WINDOW_TOKENS` (32800), `DS4_WINDOW_BAND` (120),
`DS4_WINDOW_BPT` (5.4427), `DS4_WINDOW_CORPUS`, `DS4_WINDOW_TIMEOUT` (3600),
`DS4_WINDOW_RESTORE_TIMEOUT` (300), `DS4_WINDOW_RESTORE_TRIES` (4),
`DS4_WINDOW_SETTLE_TIMEOUT` (180), `DS4_WINDOW_WATCH_INTERVAL` (1),
`DS4_WINDOW_STRATA_HEALTH`, `DS4_WINDOW_LOGDIR`, `GPU_SH`, `DS4`, `GPUD_URL`,
`GPUD_MODEL`, `STRATA_UNIT`.
