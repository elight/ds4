# ds4 levers for GLM 5.3 Flash on the 3090 (Step 5)

Every reference below was opened and read in the session that produced the
baseline in `docs/GLM53_QUANT_SURVEY.md`. Ranked by what they cost to get wrong,
worst first.

## 1. The memory guard has no rung for a 24 GB card — `ds4.c:44908`

`glm_graph_memory_guard_default_reserve_gib()` returns 24.0 GiB for a 480–640 GiB
base, 18.0 GiB for a 108–160 GiB base (the 128 GB Mac case), and **32.0 GiB as
the fallback for everything else**. `glm_graph_memory_guard_budget_bytes()`
(`ds4.c:44935`) then does `reserve_bytes >= budget_base ? 0 : …` at
`ds4.c:44941`. This box's base is 23.56 GiB, so the budget is 0.00 GiB and every
context is refused. Measured: `guard budget: 0.00 GiB (base 23.56 GiB, fraction
0.99, reserve 32.00 GiB)`.

## 2. Arena chunk and q8 reserve — `ds4_cuda.cu:2385`, `ds4_cuda.cu:1494`

`cuda_model_arena_chunk_bytes()` defaults to a **1792 MiB** chunk, overridable by
`DS4_CUDA_WEIGHT_ARENA_CHUNK_MB` (clamped 256–8192). The q8/fp16 cache reserve is
`DS4_CUDA_Q8_F16_CACHE_RESERVE_MB`, default 4096 (`ds4_cuda.cu:1494`), which is
the `reserve=4.00 GiB` in the failure line. **Proven on this box**: 1024/1024
turns `CUDA model arena alloc failed for Q4_K (1792.00 MiB)` into a working run
at 61.77 t/s prefill.

## 3. The expert cache is one global pool of 288 slots — `ds4_cuda.cu:27175`

The streaming cache is a single `std::vector<cuda_stream_expert_slot>
g_stream_expert_slots` (`ds4_cuda.cu:179`), sized at `ds4_cuda.cu:27213` by
`capacity = min(capacity, available / expert_bytes)` (`ds4_cuda.cu:27194`), where
`available = free_bytes − reserve` and **`reserve` is hardcoded to 8 GiB**
(`const uint64_t reserve = UINT64_C(8) << 30;`, `ds4_cuda.cu:27192`). The
`host_available` path at `ds4_cuda.cu:27190` only applies to `cudaDevAttrIntegrated`
devices, so a discrete 3090 never sees the 51.75 GB of free RAM.

Measured on this box: **288 slots, 1.90 GiB** — one layer's worth of experts,
shared across all 46 layers, i.e. 6 slots per layer of 288 experts = **2.2%
coverage**. Eviction is global LRU by a `used` counter
(`ds4_cuda.cu:27252-27261`), with no per-layer reservation, so a hot layer can
evict a cold layer's entire working set.

**This is the Strata lever that measures biggest.** Strata's
`--expert-cache-per-layer` curve is 2.97% → 21.4% → 70.4% hits at 1 / 8 / 64
slots per layer (`/srv/models/gguf/strata` engine `--help`). ds4's pool is
global, so it sits below the 1-slot-per-layer point in hit-rate terms.

## 4. Prefetch exists and has a kill switch — `ds4_cuda.cu:108`, `1985`, `185`

`g_model_prefetch_stream` (`ds4_cuda.cu:108`), `cuda_model_prefetch_range()`
(`ds4_cuda.cu:1985`), `cuda_stream_prefetch_before_load()`
(`ds4_cuda.cu:185`), and slots protected from eviction while prefetching via
`cuda_stream_prefetch_protects()` (`ds4_cuda.cu:27253`). `DS4_CUDA_DISABLE_SSD_PREFETCH`
turns it off, which makes it measurable as an ablation. Strata's read-ahead
measurement is 39 MB/s → 3.2 GB/s, 920 s → 70 s to first token.

## 5. Q4_K has no CUDA MoE route — `ds4.c:48573`, `ds4.c:46081`, `ds4_cuda.cu:31434`, `ds4_cuda.cu:32101`

Three routes, none accepts 12/12/12: generic routed MoE requires
`DS4_TENSOR_IQ2_XXS` (`ds4.c:46081`); the direct-scalar Q4 kernel body is a stub
that prints "CUDA stub called" and returns 0 (`ds4_cuda.cu:31434`); the batch
kernel requires type 10 in all three matrices (`ds4_cuda.cu:32101`). Prefill
picks route 2 only when `!use_grouped_moe` (`ds4.c:50399`), and
`glm_graph_indexed_prefill_grouped_moe_default()` is `g && !g->quality`
(`ds4.c:49920`); decode passes `direct_scalar_q4 = false` as a literal
(`ds4.c:51044`). Writing route 2 is the only way Q4_K runs.

## 6. Prefill chunk is fixed at 2048 — `ds4.c:39042`

`#define DS4_GLM53_PREFILL_CHUNK_TOKENS 2048u`, applied at `ds4.c:39321`.
Measured in the baseline log: `GLM compact indexed prefill chunk=2048`. Strata
sizes its prefill chunk by the cache-lend budget; ds4 has no such coupling.

## 7. KV is compact DSA only — `ds4.c:39336`

`glm_graph_…compact_cap` (`ds4.c:39336`) and the baseline log line `GLM graph
using compact DSA KV only; expanded full-attention KV cache is skipped`,
`allocating compact DSA cache: rows=4225 … f16 0.05 GiB`. KV is already small
(0.05 GiB at ctx 4225), so Strata's KV-streaming win (50.9 → 62.6 t/s with
`--kv-resident`) targets a resource that is not the bottleneck here.

## 8. Speculation is legacy-MTP only — `ds4.c:2961`

`DS4_SUPPORT_MTP_LEGACY` (`ds4.c:2961`, name string at `ds4.c:3253`). GLM's own
MTP block ships inside the main GGUF. Strata's lever is acceptance-rate costing
— spend speculation where a miss is cheaper than a stream.

## 9. The three-tier expert cache is a design, not code

`git diff --stat main..ram-expert-tier` → **`docs/RAM_EXPERT_TIER.md | 65 +++`,
1 file changed, 65 insertions**. No implementation. The design's own numbers
(`docs/RAM_EXPERT_TIER.md:3-5`): 24 GB, PCIe gen3 x16, ~12 GB/s host→device,
980 PRO ~3.5 GB/s. Its lever list (`:44-50`) is: route MoE through the slot
cache, add a pinned-RAM tier (`cudaHostRegister`), and compute RAM-tier misses
on the CPU while the GPU runs its hits. Measured on this box: RAM→VRAM 10.9 GB/s,
and the 8 GiB hardcoded reserve at `ds4_cuda.cu:27192` is exactly what keeps the
3090 from using the 51.75 GB.

## 10. Tuning surface

179 `DS4_CUDA_*` env knobs in `ds4_cuda.cu` and 892 `DS4_*` in `ds4.c`. Each is a
measurable ablation; `DS4_CUDA_DISABLE_SSD_PREFETCH` is the one that isolates the
largest single effect.

## Lever 1 fixed and measured: the guard rung

Added a sub-108 GiB rung to `glm_graph_memory_guard_default_reserve_gib()`
(`ds4.c:44908`): reserve = max(2 GiB, base/8), so this box's 23.56 GiB base gets a
2.95 GiB reserve instead of the 32 GiB fallback that collapsed the budget to zero.

Why it mattered more than it looked: with the budget at zero, the streaming
planner printed `GLM SSD streaming request adjusted to fit memory: … cache 920 -> 0
experts` — the expert cache was being **zeroed**, which is why cache size had no
effect on decode in the first sweep.

Measured, same prompt/ctx/gen-tokens, guard left ON (no `DS4_GLM_MEMORY_GUARD=0`):

| Run | prefill t/s | gen t/s | first token |
|---|---:|---:|---:|
| baseline, guard bypassed, cache default (288 slots, 1.90 GiB) | 61.77 | 1.15 | 873.8 ms |
| cache 288, guard rung, no bypass | 62.04 | 1.15 | 873.1 ms |
| **cache 920 (6.06 GiB), guard rung, no bypass** | **64.05** | **1.44** | **816.7 ms** |
| cache 1800, guard rung | — | — | `CUDA model arena alloc failed for q8_0 (1024.00 MiB): out of memory` at `blk.40.kda_v.weight` |

Decode 1.15 → **1.44 t/s (+25%)**, prefill 61.77 → 64.05 (+3.2%). The 1800-slot
run shows the ceiling: 1800 × 6.31 MiB = 11.4 GiB of cache plus 7.4 GiB of dense
weights plus 2.92 GiB of context buffers exceeds the 23.56 GiB base.
