# DeepSeek V4 Flash on the llmbox 3090 — prior art survey

Started 2026-10-09 on branch `ds4-flash-3090` (off `origin/main` @ 0aaea5a).
Purpose: record what other runtimes have already done to DeepSeek V4 Flash, and
specifically what has been done for Ampere (sm_86), before this branch writes a
single new kernel. Every claim below carries its source. Numbers are the source
authors' measurements on their own hardware, not llmbox measurements — they are
leads, not results.

## The box this survey is for

RTX 3090 (sm_86), 23.9 GiB usable VRAM, 64 GB DDR4-3200 (~50 GiB usable pinned
with strata stopped), 2.6–3.4 GB/s NVMe, no FP8 tensor cores, no AVX-512,
10 cores, PCIe gen3 x16 (~12 GB/s usable). Measured in
`docs/GLM53_3090_NOTES.md` on this box.

DS V4 Flash is 284B total / 13B active (`MODEL_CARD.md`), 43 layers, 256 routed
experts, top-6 + 1 shared, 1M context. ds4's `ds4f-q2` target is ~81 GB on disk
with IQ2_XXS gate/up and Q2_K down routed experts.

**The shape that matters:** 81 GB against 23.9 GiB VRAM + ~50 GiB pinned RAM
≈ 74 GiB. DS V4 Flash Q2 is the only one of the three Flash-class models on this
box that lands at the residency line — GLM 5.3 Flash Q2 is 89.9 GiB and MiMo
V2.6 Flash Q2_K is 117.5 GiB, and both need real streaming. So the path to a
fast DS V4 Flash here is to make it resident, not to stream it better.

---

## 1. alesha-pro/llama.cpp, branch `ds4-longctx` — the Ampere prior art

<https://github.com/alesha-pro/llama.cpp>
Docs read: `DS4HANDOFF.md`, `DS4_OPTIMIZATION_2026-07-27.md`, `README.md`.

DeepSeek-V4-Flash 284B, 2-bit IQ2_XXS, 87 GB, patched llama.cpp fork, on **4×
RTX 3090 — sm_86, no FP8, PCIe only, no NVLink**. This is the same GPU
architecture as llmbox, four of them, 96 GB VRAM. It is the closest published
prior art to this goal and the richest source of Ampere levers.

Their headline trajectory: 495 t/s prefill @97K, decode 17.61 → 31 → 41-43 →
44.5 t/s.

### Levers, with their measured deltas

| Lever | Flag | Delta | Mechanism |
|---|---|---|---|
| MMVQ `small_k` boundary | `DSV4_MMVQ_SMALLK=1` | **+5.45% decode** (36.009→37.971 pp512+tg256) | Upstream triggers `small_k` on strict `<`; both routed expert projections land *exactly* on the boundary and are excluded. IQ2_XXS @4096 cols = 16 blocks/row vs threshold 16; Q2_K @2048 = 8 vs 8. At equality the default layout pays a full cross-warp shared-memory reduction for one iteration of MAC work; `small_k` amortizes it over 4× the work. Decode-only — prefill goes through MMQ. |
| Fused decode Lightning Indexer | `DSV4_DECODE_FUSED_IDX=1` | **+16.52% decode** (28.74→33.49), −14.18% latency/token | Replaces the decomposed decode graph (`mul_mat→relu→weighted mul→permute→sum_rows`) with the fused `ggml_lightning_indexer` CUDA op, avoiding materializing and rereading a `[n_comp,1,64]` F32 score tensor in every ratio-4 layer. Long-context gain larger: ~1.79× vs the 100K legacy point. Greedy output byte-identical (SHA-256 recorded). |
| Device-resident MoE schedule | `DSV4_MOE_RESIDENT=1` | **+12.05% prefill** (442.14→495.43 t/s @97,450) | Replaces per-MoE host readback/synchronization with a device-resident schedule: GPU kernel builds the prefix sum of live token tiles per expert, persistent CTAs claim jobs through a device atomic queue. Bit-identical schedule vs deterministic MMQ; 12 production comparisons, 0 mismatches. |
| Radix top-k for prefill | `DSV4_PREFILL_RADIX_TOPK=1` | Removes a hard **90,112-token prefill OOM** | `ggml_argsort_top_k` sorts *all* rows then views k — 32,768 sorted to take 512, plus a 67 MiB i32 result. New one-block-per-row radix select, no context-scaled scratch. `test-backend-ops -o TOP_K` 455/455. |
| MoE MMQ tile sizing | `DSV4_MOE_TILE` | not stated | The ids path sized tiles for the worst-case column bound, wasting 87% of the tile. |
| Clock lock | `scripts/ds4-gpu-clocks.sh`, 1995 MHz @ 270 W | **+9% decode** (39.34→42.8) | 220 W power limit was throttling. |
| sinkhorn + fp8-KV quantize kernel rewrites | — | 42.8→**44.50** t/s | Register-based fast path for the head-compression split sinkhorn (shuffle reductions over the accumulator); two-pass bit-exact fp8 KV quantize kernel. |

### Negative results worth keeping

- **DSpark speculative decoding is a net loss.** Acceptance 27.5% with a Q4_K
  draft, yet end-to-end is slower than plain decode: the draft runs uncaptured
  and costs VRAM. Parked. Relevant here because ds4 also ships DSpark support.
- **Expert parallelism is prefill-only.** `DSV4_EXPERT_PARALLEL=1`: +20.86%
  prefill, **−34.44% decode** (36.34→23.82). Per-layer four-GPU
  broadcast/AllReduce dominates single-token decode. Kept as opt-in offline.
- **Upstream MMVQ fix (683f0c72e) is neutral for DSV4** on its own — the shapes
  run the `small_k = false` variant, which was not spilling. Kept because it
  removes the 32-byte spill the `small_k` variant carries, i.e. it is a
  prerequisite for the +5% lever, not a speedup itself. Do not report as a
  speedup.

### The pruning result, and why it is the residency answer

Same 90 GB budget, two ways to spend it (their table):

| checkpoint | params | weight bits | max context on 96 GB |
|---|---:|---:|---:|
| Unsloth UD-IQ2_M (unpruned) | 284B | 2.56 bpw | 131,072 |
| REAP K160 Q3_K/Q4_K | 180B | 3.99 bpw | 262,144 |

Quality A/B, temperature 0, 150 scenarios: **118/150 (REAP K160) vs 117/150
(2-bit full)** — they read it as a tie, with differing error profiles (REAP wins
investigation and multi-step; 2-bit wins exact-format text). K144/162B is also
validated on the fork (500.16 t/s prefill + 31.24 t/s decode at 129,960).

**Why this matters for llmbox:** REAP K160 keeps 160 of 256 routed experts per
MoE scope, 180.4B params. At 2-bit that is roughly half the bytes of the 284B at
2-bit — i.e. ~40-45 GB, which fits *resident* in 23.9 GiB VRAM + 50 GiB RAM with
room for context. Pruning is the lever that turns this box from streaming to
resident, and the published quality evidence says it costs little.

---

## 2. Lasimeri/vllm-dsv4-ampere — Ampere kernel replacements under vLLM

<https://github.com/Lasimeri/vllm-dsv4-ampere>

Upstream vLLM does not support FP8 / sparse-MLA / DeepGEMM on SM < 90 and
declined SM80-class patches ("SM80 support better lives in a fork", PR #40906).
This repo replaces every Hopper-only kernel on the V4-Flash path (cutedsl,
DeepGEMM, FlashMLA-sparse, TileLang MHC, native fp8 casts) with SM86-compatible
Triton kernels or PyTorch references.

Verified on 8× RTX 3080 20 GB, TP=8, full FP4+FP8 checkpoint, 80/20 CPU/GPU
weight offload, 32k context. Decode:

| configuration | decode tok/s |
|---|---:|
| eager, host syncs in decode path | 2.3–2.7 |
| eager, **sync-free** decode | 4.0 |
| breakable CUDA graphs (default) | **4.3–4.4** |

**The transferable finding is the sync removal: 2.3–2.7 → 4.0 t/s, +~60%, from
removing host syncs and per-token Python loops from the decode path.** Same
shape as the ds4 concern — a decode step that waits on the host is a decode step
that pays latency for nothing. Also `int4-autoround-support` branch: Marlin int4
experts, 5.89 t/s FULL capture.

Not a runtime we can use here (vLLM, multi-GPU, Python stack), but its kernel
inventory is a map of exactly which V4 ops have no Ampere path.

---

## 3. daystar7777/MoE-MADV — the streaming result, on a 64 GB machine

<https://github.com/daystar7777/MoE-MADV>

284B DS V4 Flash, 150 GB MXFP4_MOE GGUF, on a **64 GB M1 Max** — the same RAM
figure as llmbox. mmap-backed, no CPU repack, no static prewarm in steady state,
and `MADV_WILLNEED` on the selected expert byte ranges after routing, before
worker threads enter the dot-product loop.

| mode | expert page hint | decode | wall time |
|---|---|---:|---:|
| baseline | off | 0.98 tok/s | 115.5 s |
| optimized | `MADV_WILLNEED` | **1.23 tok/s** | 103.1 s |

**+25.4% decode throughput, −10.7% wall, no model change.** Supporting
measurements: first traced profile 97.6% I/O-active; decode touched ~3.08 GiB of
expert byte ranges per round; **adjacent decode expert-set overlap Jaccard 0.22**
(low — the expert set genuinely changes token to token); static top-16 prewarm
helped cold start only and did not win the steady-state 5-hour run.

Directly applicable to ds4's `--ssd-streaming` path: ds4 already owns an expert
cache, and the cheap version of this is a read-ahead on the just-routed expert
ranges. Note the GLM branch measured SSD prefetch on this box as a **no-op**
(`DS4_CUDA_DISABLE_SSD_PREFETCH`: 1.43 off vs 1.44 on, `docs/GLM53_DS4_LEVERS.md`)
— so the win here is expected to be about *page-cache admission*, not about
issuing reads earlier. That distinction is a hypothesis to test, not a result.

---

## 4. antirez/llama.cpp-deepseek-v4-flash — the ds4 author's own llama.cpp fork

<https://github.com/antirez/llama.cpp-deepseek-v4-flash>
Quants: <https://huggingface.co/antirez/deepseek-v4-gguf>

The upstream ds4 author's llama.cpp implementation of DSv4, targeting 128 GB
Macs with 2-bit routed experts: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat.gguf`.
CPU and Metal backends, Metal faster. Author's own caveat, quoted: "behaves very
very well in the chat, frontier-model vibes, but it was not extensively tested."

Relevant to this branch because it is the same quant recipe ds4 ships
(IQ2_XXS gate/up + Q2_K down, Q8_0 attention/shared/output), so the quant
choices are already agreed between the two implementations, and the llama.cpp
side is where the MMVQ/MMQ kernel levers above live.

---

## 5. KTransformers — CPU/GPU expert split, and its quant

<https://ktransformers.net/en/docs/inference/long-context-deployment>,
<https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/deepseek-v2-injection.md>

DeepSeek V4 Flash deployment: `--kt-method MXFP4` (routed experts as I8 +
ue8m0), `--kt-num-gpu-experts 10` — 10 experts on GPU, the rest on CPU.
Placement strategies `uniform | frequency | front-loading | random`, plus
`--kt-enable-dynamic-expert-update` to re-place from runtime routing statistics.
Historical result on DeepSeek-R1: 286.55 t/s prefill vs 10.31 t/s in llama.cpp
on the same 2×32-core box (AMX MoE kernel + selective 6-expert GPU use).

**Caveat for llmbox, stated plainly:** KTransformers' CPU side leans on
AVX512-BF16 (Zen4+) and AVX-VNNI (Alder Lake+). llmbox is an i9-10900K — AVX2
only, no AVX-512 (`docs/GLM53_3090_NOTES.md`). The CPU-expert-computation half of
this design is not available here; the *placement policy* (which experts are
hot) is, and ds4 already has a heat-based cache to compare it against.

---

## 6. Quant artifacts worth knowing about

- **`teamblobfish/DeepSeek-V4-Flash-GGUF`** — quants for the V4-aware llama.cpp
  fork `cchuter/llama.cpp @ feat/v4-port-cuda`. README states these do **not**
  load on upstream llama.cpp; V4 architecture support (compressor decode,
  hyperconnection, lightning indexer, FP8 KV simulation, NextN heads) lives only
  in the fork. Backends: Metal, CUDA (Ada/Blackwell), CPU.
- **`servantofares/DeepSeek-V4-Flash-FP4-FP8-SSD`** — dense GGUF + routed-expert
  sidecar layout: FP8 (`F8_E4M3_B128`) dense, native MXFP4 routed experts in a
  sidecar manifest. A layout experiment for SSD-backed MoE.
- **`xik94/DeepSeek-V4-Flash-162B-REAP-GGUF`** — REAP-pruned 162B, multi-GPU
  quants, tested on 2× RTX 4090 at sm_89. Requires a patched llama.cpp: stock
  crashes after 2–3 prompts on non-unique expert IDs in REAP's `tid2eid` routing
  tensors (upstream issues #24591, #25598, labeled wontfix). Patch 2 excludes i32
  routing tensors from quantization. **Read before building any REAP quant here.**
- **`anonymousmaharaj/DeepSeek-V4-Flash-0731-REAP-K160-GGUF`** — the K160 GGUF
  from §1, 89.9 GB, 3.99 bpw, four shards.
- **`lovedheart/DeepSeek-V4-Flash-GGUF`** — MXFP4_MOE, 139.91 GiB, the file
  MoE-MADV measured.
- **`deepseek-ai/DeepSeek-V4-Flash-DSpark`** — same checkpoint with the
  speculative-decoding module attached (`deepseek-ai/DeepSpec`). Given the §1
  negative result, treat DSpark as a measured-loss path until proven otherwise
  on this card.

---

## What this survey says to do on this box, in order

1. **Port the MMVQ `small_k` boundary fix.** It is +5.45% decode on the same
   architecture with the *same two quant types* ds4 uses (IQ2_XXS @4096, Q2_K
   @2048). ds4 has its own CUDA MoE kernels, so the equivalent is a dispatch
   condition, not a new kernel. Cheapest lever on the list.
2. **Check the power limit and lock clocks.** llmbox reports a 250 W limit at
   1950 MHz (`docs/GLM53_3090_NOTES.md`); the fork gained +9% decode by locking
   clocks on a card that was throttling at 220 W. Measure, do not assume.
3. **Audit ds4's decode path for host syncs.** The vLLM Ampere fork gained ~60%
   from removing them. This is a measurement task, not a kernel task.
4. **Fuse the decode indexer if ds4 has a decomposed one.** +16.5% on the same
   model, byte-identical output.
5. **Try REAP K160/K144 at 2-bit as the residency route.** Published quality
   parity at the same byte budget, and it is the only idea here that makes the
   model fit in RAM+VRAM outright.
6. **Try `MADV_WILLNEED`-style admission on ds4's streaming reads** — but expect
   it to interact with the fact that SSD prefetch measured as a no-op for GLM on
   this card.

Explicitly not on the list: expert parallelism (−34% decode), DSpark
speculation (net loss at 27.5% acceptance), KTransformers' CPU expert compute
(no AVX-512 on this box).
