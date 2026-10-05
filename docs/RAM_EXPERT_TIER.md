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

RTX 3090, PCIe gen3, ctx 8192, 128 generated tokens, 5575 VRAM slots (7.7 GiB).
Command: `ds4-bench -m Qwen3.8-Flash-Next-Q2.gguf --cuda --ssd-streaming
--prompt-file tests/long_context_story_prompt.txt --ctx-start 8192 --ctx-max 8192
--gen-tokens 128`, plus `--ram-expert-cache 0` for the phase 1 row.

### Phase 4: CPU computes RAM misses (hybrid decode)

Default on; `DS4_CPU_HYBRID=0` turns it off. Same command, the story prompt
repeated twice so a 32K frontier fits (`misc/llmbox/qwen-hybrid.sh bench`).

| ctx | Hybrid | Decode t/s | Prefill t/s | VRAM / PCIe / CPU share of lookups | Slots |
|---:|---|---:|---:|---|---:|
| 8192 | off | 21.53 | 727.2 | 44.2% / 55.8% / 0% | 5575 |
| 8192 | **on** | **34.47** | 721.6 | 43.7% / 31.6% / 24.7% | 5575 |
| 32768 | off | 28.04 | 705.9 | 32.1% / 67.9% / 0% | 4841 |
| 32768 | **on** | **34.70** | 705.2 | 22.7% / 61.9% / 15.4% | 4841 |

Decode is 60% faster at 8K and 24% faster at 32K; prefill is unchanged (it
batches on the GPU). The lookup shares include prefill, which is why PCIe
dominates them even when decode sends almost nothing over PCIe.

Quality, teacher-forced over 480 story tokens (`qwen-hybrid.sh ppl`):

| Mode | Perplexity |
|---|---:|
| hybrid off (all experts on the GPU) | 15.878 |
| hybrid on, `DS4_CPU_HYBRID_EXACT=1` (float activations, scalar) | 15.890 |
| hybrid on (8-bit activations, AVX2) | 15.535 |

The exact mode matches the GPU to 0.08%, so the CPU kernels are right. The
8-bit activation path moves perplexity by 2%, in the good direction on this
text, which is within what quantising activations does either way.

### Strata Q2_0 on the same box

Strata's shipped `strata-q2_0.json` (MTP drafts, `--spec 4`), same story text,
128 tokens, temperature 0 (`misc/llmbox/strata-bench.sh spec`). Strata will not
serve a native pack without MTP, so its draft acceptance gives the rate per
verify pass.

| ctx | Engine | Decode t/s | Tokens per pass | Passes/s | Prefill t/s | Expert slots |
|---:|---|---:|---:|---:|---:|---:|
| 8192 | ds4 hybrid | 34.47 | 1 | 34.5 | 721.6 | 5575 |
| 8192 | Strata | 107.9 | 2.78 | 38.8 | 2196.7 | 13398 |
| 32768 | ds4 hybrid | 34.70 | 1 | 34.7 | 705.2 | 4841 |
| 32768 | Strata | 102.5 | 2.98 | 34.4 | 2433.4 | 13398 |

Per forward pass, ds4 matches Strata at 32K and is 11% behind at 8K. The
3x decode gap is MTP drafting, and the 3x prefill gap is Strata's batched
prompt path. Strata also holds 2.8 times as many experts in VRAM: it budgets
the card with 32K of int8 KV resident, while ds4 reserves 7.7 GiB of prefill
and context buffers.

The 147 GB Qwen file is mostly a 95 GiB BF16 n-gram table that stays on disk;
the routed experts are 35.4 GiB and the rest of the weights are 6.3 GiB in VRAM.
Correctness: prompt logits from a 256-slot SSD-only cache and from the full
RAM tier are bit-identical.
