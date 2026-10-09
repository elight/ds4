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

## The guard rung oversubscribed the card; the q8 staging cap is what makes it fit

The 920-slot row above is **not reproducible from a cold card**, and finding out
why is what produced the stable configuration. With the card verified free
(`nvidia-smi --query-gpu=memory.used` = 1 MiB before the run), cache 920 failed
with `CUDA model arena alloc failed for q8_0 (1024.00 MiB chunk): out of memory`,
and so did 300, 500, 700 and 288 at an arena chunk of 256 MiB.

The mechanism is the guard rung itself. With the budget at zero, the streaming
planner zeroed the expert cache and every allocation stayed small; giving the box
a real 20.3 GiB budget let ds4 plan a larger q8/fp16 staging cache, and the
staging cache plus the dense weights plus the expert cache exceeded the 23.56 GiB
base. Capping the staging cache — `DS4_CUDA_Q8_F16_CACHE_MB=1024`,
`DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=512` (`ds4_cuda.cu:1488`, `:1494`) — makes the
run reproducible from a cold card.

Re-measured on that platform, card verified free before the window, guard ON, no
`DS4_GLM_MEMORY_GUARD=0`, ctx 4096, 128 generated tokens,
`tests/long_context_story_prompt.txt`:

| cache experts | cache bytes | prefill t/s | gen t/s | first token |
|---:|---:|---:|---:|---:|
| 288 (default) | 1.90 GiB | 62.25 | 1.15 | 867.7 ms |
| 500 | 3.29 GiB | 63.72 | 1.38 | 863.7 ms |
| **920** | **6.06 GiB** | **63.70** | **1.43** | **826.0 ms** |

Against the baseline (61.77 prefill, 1.15 generation): **prefill +3.1%,
generation +24%**. Cache 1800 exceeds the base and fails in the dense arena.

## Rejected: per-layer expert-slot eviction (measured −12% on decode)

Strata's biggest measured win is per-layer expert slots
(`--expert-cache-per-layer`: 2.97% → 21.4% → 70.4% hits at 1/8/64 slots per
layer), so it was implemented in ds4's CUDA streaming cache: a slot's layer is
derived from its gate offset (`table->gate_offset` …
`+ n_total_expert × gate_expert_bytes`), and the victim scan prefers slots
holding the requesting layer's own experts, with global LRU kept as the fallback
so a layer that owns no slot can still load. An ablation switch
(`DS4_CUDA_DISABLE_EXPERT_PER_LAYER_EVICT`) made it measurable in one binary.

A/B in one window, card verified free, guard ON, cache 920, ctx 4096, 128
generated tokens (`~/claude-tmp/glm53-sweep-abl2.txt`):

| run | prefill t/s | gen t/s | steady t/s |
|---|---:|---:|---:|
| cache 288 (warm-up) | 62.22 | 1.15 | 1.15 |
| **cache 920, per-layer eviction OFF** | 63.78 | **1.63** | **1.64** |
| cache 920, per-layer eviction ON | 63.85 | 1.44 | 1.44 |

Per-layer reservation costs **12% of decode** (1.63 → 1.44). Prefill is
unchanged. The patch is reverted; global LRU is the better policy for this
cache. The likely reason is visible in the numbers: with 920 slots over 46
layers, a per-layer reservation is 20 slots, and a layer's top-8 selection
repeats heavily within a token burst — global LRU lets the hot experts of the
current layer keep slots that a strict per-layer quota would evict.

**Best accepted configuration so far** (guard rung + `DS4_CUDA_STREAM_EXPERT_
RESERVE_MB=2048` + `DS4_CUDA_Q8_F16_CACHE_MB=1024` +
`DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=512` + cache 920, global LRU):
**63.78 t/s prefill, 1.63 t/s generation** against the 61.77 / 1.15 baseline —
**prefill +3.3%, generation +42%**.

## Corrected: the rejected scan was still in the tree, and removing it is worth +20%

The paragraph above says the per-layer patch was reverted. It was not. The
ablation switch was removed but the victim scan itself stayed at
`ds4_cuda.cu:27255`, so every run measured after commit `aaa23f0` — including
the whole accepted table and the 1.43/1.44/1.43 decode rows — ran with the
rejected policy active. The notes drifted from the tree.

Removed in this commit (`ds4_cuda.cu`, one hunk, back to main's global-LRU
scan), and measured A/B across two separate binaries in one window, card
verified free at window start, guard on, ctx 4096, 128 generated tokens,
`~/claude-tmp/glm53-ab-plrev2.txt`:

| binary | cache experts | prefill t/s | gen t/s | first token |
|---|---:|---:|---:|---:|
| `ds4-bench-plon` (scan present) | 1200 | 63.69 | 1.47 | 798.0 ms |
| **`ds4-bench-ploff` (scan removed)** | **1200** | **63.48** | **1.76** | **796.0 ms** |
| `ds4-bench-ploff` (scan removed) | 920 | 63.39 | 1.64 | 816.5 ms |
| `ds4-bench-plon` (scan present) | 920 | 63.70 | 1.43 | 821.8 ms |

Decode **1.47 → 1.76 t/s at cache 1200 (+20%)** and **1.43 → 1.64 at cache 920
(+15%)**, prefill flat within 0.5%. This reproduces the ablation-switch A/B
(1.44 on vs 1.63 off at 920) with two independent binaries, which is what
settles it.

Why the removal cannot change output: eviction decides *which resident slot is
overwritten*, never *which experts are computed*. Every requested expert is
loaded before its kernel runs, whether from a slot hit or from the SSD tier. The
first-token times agree within 2 ms and prefill within 0.5%, which is what
output-identical work looks like here. The cache-size fixture evidence points
the same way: the Z.AI FP8 20-case score is identical at cache 288 and cache
1200 (`avg_nll=0.468912094`, 2455 tokens,
`~/claude-tmp/glm53-fixture-acc.txt`).

**New best accepted configuration**: guard rung + `DS4_CUDA_STREAM_EXPERT_RESERVE_
MB=2048` + `DS4_CUDA_Q8_F16_CACHE_MB=1024` + `DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=512`
+ cache 1200 + global LRU → **63.48 t/s prefill, 1.76 t/s generation** against
the 61.77 / 1.15 baseline: **prefill +2.8%, generation +53%**.

## Reproducibility: the first run after strata's teardown fails; follow-on runs do not

Every configuration above was tested again from a card verified free
(`memory.used` = 1 MiB before the run), and the result is a protocol fact, not a
config fact:

- A GLM run that is the **first** run in a window, immediately after
  `systemctl stop llmbox-strata`, fails with
  `CUDA model arena alloc failed for q8_0 (1024.00 MiB chunk): out of memory`
  at cache 288, 300, 500, 700, 920 and 1800, at arena chunks of 1024 and 256 MiB,
  with and without `DS4_CUDA_Q8_F16_CACHE_MB`/`_RESERVE_MB` caps.
- The **same binary and same flags** succeed as the second and third run in the
  same window.

Two default policy rungs were written and tested against this — a 1/24-of-card
cap on the Q8→F16 staging cache and a 1/48-of-card reserve for 20–32 GiB cards
(`cuda_q8_f16_cache_limit_bytes` / `cuda_q8_f16_cache_reserve_bytes`,
`ds4_cuda.cu:1486`, `:1492`). Neither made the cold first run succeed, so both
were reverted rather than shipped: an unmeasured policy change is not an
optimization.

**The accepted table, measured in a warm window** (a 288-expert run first, then
the measured runs; card verified free at window start; guard ON, no
`DS4_GLM_MEMORY_GUARD=0`; ctx 4096, 128 generated tokens;
`~/claude-tmp/glm53-sweep-warmtable.txt`):

| cache experts | prefill t/s | gen t/s | steady t/s | first token |
|---:|---:|---:|---:|---:|
| 288 (default) | 62.07 | 1.15 | 1.15 | 873.1 ms |
| 500 | 63.75 | 1.38 | 1.38 | 864.7 ms |
| **920** | **63.78** | **1.43** | **1.44** | **820.8 ms |

Generation at 920 measured 1.43, 1.44 and 1.43 in three separate windows, and
1.63 once (`~/claude-tmp/glm53-sweep-abl2.txt`); the spread is window state, not
cache size. Against the 61.77 / 1.15 baseline: **prefill +3.3%, generation
+24%**.

## Cache ceiling: 1200 experts is the largest that fits, and the fastest

Warm-window sweeps (`~/claude-tmp/glm53-sweep-ceiling.txt`,
`~/claude-tmp/glm53-sweep-ceiling2.txt`), same protocol as the accepted table:

| cache experts | cache bytes | prefill t/s | gen t/s | first token |
|---:|---:|---:|---:|---:|
| 288 (default) | 1.90 GiB | 62.03–62.15 | 1.15 | ~870 ms |
| 920 | 6.06 GiB | 63.78–63.87 | 1.43–1.44 | 818–821 ms |
| **1200** | **7.89 GiB** | **63.93** | **1.47** | **799.5 ms** |
| 1300 | 8.54 GiB | — | — | fails: dense `q8_0` arena |
| 1400 | 9.20 GiB | — | — | fails: dense `q8_0` arena |
| 1200 with `DS4_CUDA_Q8_F16_CACHE_MB=2048` | 7.89 GiB | — | — | fails: the bigger staging cache takes the room the cache needed |

The last row is the trade stated plainly: on this card the Q8→F16 staging cache
and the streaming expert cache draw from the same VRAM, and giving staging 2 GiB
instead of 1 GiB makes a 1200-expert cache stop fitting.

**Two neutral levers, measured and not adopted:**
- SSD prefetch (`DS4_CUDA_DISABLE_SSD_PREFETCH`, `ds4_cuda.cu:108/:1985`):
  1.43 t/s off vs 1.44 t/s on at cache 920 — no measurable contribution
  (`~/claude-tmp/glm53-sweep-prefetch.txt`). Strata's 39 MB/s → 3.2 GB/s
  read-ahead win does not show up here.
- Prefill chunk (`DS4_GLM53_PREFILL_CHUNK`, added at `ds4.c:39321`): 2048 →
  63.81 t/s, 4096 → 63.74 t/s, 1024 → fails. Prefill is not chunk-limited;
  it batches expert reuse across the 2048-token chunk, which is why prefill
  costs 15.7 ms/token while decode costs 870 ms/token for the same experts
  (`~/claude-tmp/glm53-sweep-chunk.txt`).
