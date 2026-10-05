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
rest compute. The CPU-computes-RAM-experts split (branch `cpu-hybrid`) plugs in
at `moe_cpu_split()`, which already receives each job's tier.
