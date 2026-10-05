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
