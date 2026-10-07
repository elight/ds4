# Three-tier expert cache: VRAM, pinned RAM, SSD

Fork branch `ram-expert-tier` (github.com/elight/ds4). Target box: one RTX 3090
(24 GB, PCIe gen3 x16, about 12 GB/s host to device), i9-10900K (10 cores, AVX2,
no AVX-512), 64 GB DDR4, Samsung 980 PRO on a gen3 slot (about 3.5 GB/s).

Ideas are borrowed from Strata (github.com/Niko1221/Strata, MIT). Ported code
carries a header crediting it.

## The problem

Upstream CUDA SSD streaming keeps one tier: a VRAM slot cache filled by `pread`
from the GGUF, with the page cache dropped after each read. On this box:

- Qwen3.8 Flash Next is not streamable on CUDA at all
  (docs/QWEN38_FLASH_NEXT.md:157). Without streaming it either pins 42 GiB of
  host RAM and decodes every weight over PCIe, or fails to start.
- A miss costs an SSD read plus a PCIe copy, synchronously, on the decode path.

## Tiers

| Tier | Holds | Filled by | Served by |
|---|---|---|---|
| VRAM slots | hottest experts by routing profile, then learned heat | startup seed, async swaps | GPU kernels, by slot index |
| Pinned RAM arena | as many of the rest as the RAM budget allows | startup bulk read, SSD promotion | CPU kernels in place (AVX2), or a PCIe copy share |
| SSD | the remainder | — | O_DIRECT read into the RAM arena, ahead of need where predicted |

Non-routed weights (attention, GDN, routers, shared expert, head) and KV stay
in VRAM.

## Geometry this is tuned for

| Model | Routed expert | Experts | Routed total | Top-k per token |
|---|---|---|---|---|
| Qwen3.8 Flash Next Q2 | 1.42 MiB | 49 x 512 | 34.8 GiB | 10 (+1 shared) |
| MiMo V2.6 Flash Q2_K | about 8.3 MB | 47 x 256 | about 99 GB | 8 |

Qwen: about 14 GiB of VRAM slots hold about 40% of experts; all of the rest
fits pinned RAM, so SSD is only touched at startup. MiMo: VRAM about 15%,
RAM about 50%, SSD the rest. MiMo is where the SSD tier and prediction matter.

## Phases

1. **Qwen3.8 streams on CUDA.** Route `qwen4_graph_moe` through the slot cache
   (`begin_load`, slot-remapped `selected`), as DeepSeek and GLM already do.
2. **Pinned RAM tier.** Hugepage-backed arena, `cudaHostRegister`ed; misses
   copy from RAM, and from SSD only when RAM does not hold the expert.
3. **Profile seeding and learned heat.** Routing-frequency profile seeds VRAM
   and RAM; decayed counts drive async swaps off the decode path.
4. **CPU computes RAM-tier misses in place** while the GPU runs its hits, with
   a probed PCIe share (Strata's doorbell handoff). Port of Strata's AVX2
   IQ2_XXS and Q2_K kernels; core pinning, one thread per physical core.
5. **Router lookahead.** Apply layer l+1's router to layer l's input; promote
   predicted SSD experts into RAM (and RAM into VRAM when slots are free)
   before they are needed.
6. **MiMo V2.6 Flash** model support (`mimo2` GGUF, hybrid sliding-window
   attention, 256 experts top-8), then the same tiers.

Each phase lands with a measurement: decode and prefill tokens per second,
VRAM hit rate, RAM hit rate, SSD bytes per token. Results go in the table below.

## Results

| Date | Model | Phase | Decode t/s | Prefill t/s | VRAM hit | Notes |
|---|---|---|---:|---:|---:|---|
| 2026-10-05 | Qwen3.8 Q2 | 1: VRAM slots + SSD | 6.57 | 315.30 | 46.3% | `--ram-expert-cache 0`; 62.8 GiB read from SSD at 2.0 GB/s |
| 2026-10-05 | Qwen3.8 Q2 | 2: + pinned RAM tier | 29.34 | 717.35 | 59.2% | all 35.4 GiB of experts in RAM (16 s O_DIRECT fill); RAM→VRAM at 10.9 GB/s; no SSD reads |
| 2026-10-05 | MiMo V2.6 Flash Q2_K | 6, inclusive tiers | 3.41 | — | 65.8% | story: 34-token prompt, 154 decoded, ctx 4096, 36 GB RAM tier |
| 2026-10-05 | MiMo V2.6 Flash Q2_K | 6, exclusive tiers | 3.66 | — | 65.8% | same story run; SSD share 14.8% → 6.8%. 3.40 when other tenants leave 10 fewer slots |
| 2026-10-05 | MiMo V2.6 Flash Q2_K | llama.cpp `-ot exps=CPU` | 4.08 | — | — | same story run, 10 threads, page cache |
| 2026-10-05 | MiMo V2.6 Flash Q2_K | 6, ctx 8192, ubatch 1024 | 2.64 | 19.39 | 63.1% | 2204-token prompt (llama.cpp `docs/build.md`), 63 decoded |
| 2026-10-05 | MiMo V2.6 Flash Q2_K | 6, ctx 8192, ubatch 2304 | 2.52 | 27.25 | 62.0% | one prefill pass reads 65 GB from SSD instead of 164 GB; costs 84 VRAM slots |
| 2026-10-05 | MiMo V2.6 Flash Q2_K | 6, ctx 8192, 44 GB RAM | 2.87 | 20.66 | 63.1% | `--ram-gb 44`, ubatch 1024 |
| 2026-10-05 | MiMo V2.6 Flash Q2_K | llama.cpp, ctx 8192 | 3.66 | 17.59 | — | same 2204-token prompt, `-b 1024 -ub 1024` |
| 2026-10-05 | MiMo V2.6 Flash Q2_K | 4+6: RAM tier on CPU | 4.75 | — | 32.6% | story run; 59.5% of decode lookups computed on the CPU in place, SSD 7.9% |
| 2026-10-05 | MiMo V2.6 Flash Q2_K | 6, same window | 3.65 | — | 65.8% | `--cpu-experts 0`, back to back with the row above |
| 2026-10-07 | MiMo V2.6 Flash Q2_K | baseline, `--cpu-experts 1` | 5.18 | 2.59 | 41.5% | `mimo-bench.sh baseline` story step; 159 decoded, SSD wait 13.03 s of 30.69 s, 51.5% of lookups on the CPU |
| 2026-10-07 | MiMo V2.6 Flash Q2_K | baseline, `--cpu-experts 0` | 3.66 | 2.59 | 65.8% | same window, 154 decoded, SSD wait 12.19 s of 41.99 s |
| 2026-10-07 | MiMo V2.6 Flash Q2_K | llama.cpp, story | 4.70 | 1.67 | — | same story prompt, `-ngl 99 -ot exps=CPU -c 4096 -t 10` |
| 2026-10-07 | MiMo V2.6 Flash Q2_K | RAM tier 44 GB | 5.91 | — | 39.9% | story, `--ram-gb 44`; SSD wait 9.89 s of 26.91 s, SSD share 5.4% |
| 2026-10-07 | MiMo V2.6 Flash Q2_K | RAM tier 52 GB | 6.30 | — | 40.2% | story, `--ram-gb 52`; SSD wait 8.38 s of 25.25 s, SSD share 4.5% |
| 2026-10-07 | MiMo V2.6 Flash Q2_K | RAM tier default (45.2 GB pinned) | 5.98 | — | 40.3% | story, no `--ram-gb`; SSD wait 9.31 s of 25.42 s, SSD share 5.3%, against 13.01 s and 7.1% at the old 36 GB cap |

### Where a MiMo decode token goes

`ds4-mimo2` prints a phase breakdown per measured pass. On the 36 GB story run:
SSD 13.01 s, copy 0.00 s, CPU experts 0.10 s, routing 23.72 s, fetch 0.30 s,
attention 0.10 s, of 30.51 s total.

The routing column is not host cost. `m2g_download` is an async copy followed by a
stream synchronize, so the router download at the top of a layer drains the
compute stream: what is measured in it is the previous layer's GPU MoE, waited
on. The host's own share is the 256 sigmoids and the top-8 scan, about 2,300
flops per token-layer. A MiMo decode token is roughly 42% SSD wait and 34% GPU
MoE at the 36 GB tier; the SSD share falls to 4.5% at 52 GB.

Two things measured and rejected. Seeding the tiers from a routing profile built
by a decode-heavy pass made the story run marginally slower, 5.16 t/s against
5.21, SSD share unchanged at 7.1%: LRU over 12,032 experts with 1,793 slots
already places about as well as a static ranking. And the CPU expert split does
not collapse the VRAM hit rate — 41.5% against 65.8% is the split working as
designed, because a RAM-tier expert computed on the CPU is never promoted into a
VRAM slot, and the run is 42% faster for it.

`M2_STREAM_MIN` was tried at 32 and put back at 256. A 34-token prefill does not
stream at 256, and making it stream is much better for the prefill and worse for
the decode that follows it: prefill 3.89 t/s against 2.59, SSD share 49.5%
against 59.9%, 21.5 GB read against 26.0 GB — and decode 5.25 t/s against 5.98,
with the decode SSD share up from 5.3% to 7.1%. Streaming marks the slots it
used evict-first and never caches them, so decode starts on a cache that holds
none of the experts the prompt touched. Decode is the number the box is judged
on, so the threshold stays where a 34-token batch caches what it read.
| 2026-10-07 | MiMo V2.6 Flash Q2_K | baseline, 2203-token prompt | 3.25 | 17.89 | 36.0% | ctx 8192 ubatch 1024, `--cpu-experts 1`; prefill reads 65.5% of experts from SSD |
| 2026-10-07 | MiMo V2.6 Flash Q2_K | llama.cpp, 2203-token prompt | 3.87 | 17.54 | — | same prompt, `-b 1024 -ub 1024` |

The 2026-10-07 rows come from `misc/llmbox/mimo-bench.sh baseline`, whose
records are in `~/claude-tmp/mimo-bench/baseline/results.jsonl`. The prompt is a
2203-token cut of llama.cpp `docs/build.md` measured with `llama-tokenize`, so
the llama.cpp and ds4 rows score the same tokens. Two things the earlier rows
left open: decode on the long prompt is behind llama.cpp (3.25 against 3.87)
while prefill is level, and the story run's 13 s of SSD wait is 42% of the
time in the window.
| 2026-10-05 | MiMo V2.6 Flash Q2_K | llama.cpp, same window | 5.04 | — | — | story run; faster than the 4.08 row, likely a warmer page cache |
| 2026-10-05 | MiMo V2.6 Flash Q2_K | 4+6, MTP 3 drafts | 3.24 | — | — | 96-token story; 34% kept, 2.02 tokens/pass; a 4-token pass reads about 2.7x the experts of a one-token step |
| 2026-10-05 | MiMo V2.6 Flash Q2_K | 4+6, `--slots 1650` | 4.84 | — | — | 96-token story, MTP off; the rows below are the same window and slot count |
| 2026-10-05 | MiMo V2.6 Flash Q2_K | + MTP 1 draft | 4.66 | — | — | 66% kept, 1.66 tokens/pass |
| 2026-10-05 | MiMo V2.6 Flash Q2_K | + MTP 3, gate 0.70 | 4.55 | — | — | 64% kept, 1.83 tokens/pass |
| 2026-10-05 | MiMo V2.6 Flash Q2_K | + MTP 1, first gated 0.60 | 4.65 | — | — | 83% kept, 1.36 tokens/pass |
| 2026-10-05 | MiMo V2.6 Flash Q2_K | + MTP 1, first gated 0.85 | 4.56 | — | — | 89% kept, 1.20 tokens/pass |
| 2026-10-05 | MiMo V2.6 Flash Q2_K | + MTP 3, gates 0.90/0.85 | 4.44 | — | — | 100% kept, 1.14 tokens/pass |

RTX 3090, PCIe gen3, ctx 8192, 128 generated tokens, 5575 VRAM slots (7.7 GiB).
Command: `ds4-bench -m Qwen3.8-Flash-Next-Q2.gguf --cuda --ssd-streaming
--prompt-file tests/long_context_story_prompt.txt --ctx-start 8192 --ctx-max 8192
--gen-tokens 128`, plus `--ram-expert-cache 0` for the phase 1 row.

The 147 GB Qwen file is mostly a 95 GiB BF16 n-gram table that stays on disk;
the routed experts are 35.4 GiB and the rest of the weights are 6.3 GiB in VRAM.
Correctness: prompt logits from a 256-slot SSD-only cache and from the full
RAM tier are bit-identical.

### MiMo V2.6 Flash (`ds4-mimo2`)

A separate engine, `ds4_mimo2.c` + `ds4_mimo2_cuda.cu`, built with
`make ds4-mimo2`. It loads the ggml-org split GGUF directly (pass the
`-00001-of-` shard). Output matches llama.cpp: identical tokens on a short
prompt, and 95.5% top-1 agreement teacher-forcing llama.cpp's own 154-token
story (`--score`; every miss is a near-tie under 0.3 logits).

How the tiers behave:

- **Exclusive.** An expert lives in VRAM or in RAM, not both. Promoting one
  frees its RAM entry; the VRAM expert it displaces is copied back down to RAM
  on its own CUDA stream, overlapping the uploads.
- **Prefill streams.** A batch of 256 or more tokens touches nearly every
  expert, a scan larger than the cache. Those layers pass their misses through
  transient slots instead of flushing the cache.
- **Lookahead is off** (`--lookahead N` turns it on). Decode is bound by the
  SSD, so speculative reads slow the reads that are actually needed: 3.41 t/s
  off, 3.07 t/s at depth 2.

Where decode time goes on the 2204-token run, ~380 ms/token: ~180 ms waiting
on the SSD (16% of experts), ~65 ms copying RAM-tier experts over PCIe, the
rest compute.

**RAM-tier experts run on the CPU in decode** (`--cpu-experts 1`, the
default). `moe_cpu_split()` hands experts already resident in RAM to the CPU
expert pool, which reads them in place while the GPU runs its VRAM hits and
waits on SSD misses; `--cpu-pcie N` keeps N per layer on the copy path. The
down projection is MXFP4, so the pool has an MXFP4 x Q8_K AVX2 dot for it
(`tests/test_cpu_experts.c`). Parity: teacher-forcing llama.cpp's story one
token per step (`misc/llmbox/mimo2-score-decode.sh`) gives ppl 1.799 and 95.5%
top-1 agreement with the split, 1.802 and 95.5% without. Story decode goes
3.65 → 4.75 t/s; the host never waits on the CPU (0.09 s per run), so the
SSD (14.5 s of the 33.5 s) is what is left.

**MTP** (`--mtp mtp-MiMo-V2.6-Flash-RL-Q8_0.gguf`, `--mtp-draft N`, 1-3,
default 1) loads the nextn heads the way llama.cpp chains them: head i drafts
the token i+1 ahead from the previous head's output, and the trunk checks all
drafts in one batch. The heads take about 1.05 GB of VRAM; they reuse the
trunk's Q6_K output and embeddings instead of the file's Q8_0 copies.
`--mtp-gate P` stops the chain at a head whose top probability is below P, and
`--mtp-gate-first P` lets the step skip drafting altogether.

Correctness: with `--cpu-experts 0`, greedy output is byte-identical with and
without MTP (`misc/llmbox/mimo2-mtp.sh`). With the CPU split on, a plain run
is repeatable at a fixed `--slots` (two runs byte-identical), but MTP output
is not comparable byte for byte: a verify batch touches experts in a different
order, so later tokens find different experts in VRAM and on the CPU, whose
rounding differs from the GPU's. Without `--slots`, the slot count follows free
VRAM, so any change in VRAM use (the heads, a buffer) moves placement too.

MTP does not pay on this box, gated or not: every setting in the table is
slower than plain decode. Decode is bound by expert reads, a verify pass of
k+1 tokens reads the union of their experts, and the MTP runs also wait about
1 s longer on the SSD per 96 tokens. Drafting itself costs 4 ms per pass.
