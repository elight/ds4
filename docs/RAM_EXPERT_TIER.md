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
| 2026-10-05 | MiMo V2.6 Flash Q2_K | llama.cpp, same window | 5.04 | — | — | story run; faster than the 4.08 row, likely a warmer page cache |
| 2026-10-05 | MiMo V2.6 Flash Q2_K | 4+6, MTP 1 draft | 4.95 | — | — | 96-token story; 76% of drafts kept, 1.76 tokens per verify pass; 4.90 without MTP in the same window |
| 2026-10-05 | MiMo V2.6 Flash Q2_K | 4+6, MTP 3 drafts | 3.24 | — | — | same run; 34% kept, 2.02 tokens/pass; a 4-token pass reads about 2.7x the experts of a one-token step |

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
trunk's Q6_K output and embeddings instead of the file's Q8_0 copies. With
`--cpu-experts 0` greedy output is byte-identical with and without MTP
(`misc/llmbox/mimo2-mtp.sh`). With the CPU split on, plain greedy output
already differs from run to run, because which experts land on the CPU depends
on SSD timing, so the two cannot be compared byte for byte.

MTP barely pays here. Decode is bound by expert reads, and a verify pass of
k+1 tokens reads the union of their experts, so it costs nearly as much per
token as plain decode. One draft is a wash (+1%); three drafts lose a third,
because the second and third drafts are kept in only 23% and 9% of passes.
Drafting itself costs 3.6 ms per pass on the GPU against a ~355 ms pass.

