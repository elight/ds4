# GLM 5.3 Flash on llmbox — hardware and driver notes (Step 0)

Every figure below was measured on this box on 2026-10-08, with the command that
produced it. Later steps (quant choice, Strata ports, per-optimization
benchmarks) are tuned against these numbers, not against assumptions.

Window scripts used: `~/claude-tmp/glm53-hw-window.sh`,
`~/claude-tmp/glm53-idle-window.sh`, `~/claude-tmp/memprobe.cu`.
Raw captures: `~/claude-tmp/glm53-idle-window-idle.txt`,
`~/claude-tmp/glm53-hw-window-baseline.txt`.

## The box

| Item | Measured | Command |
|---|---|---|
| GPU | NVIDIA GeForce RTX 3090 (GA102), compute cap 8.6 → **sm_86** | `nvidia-smi --query-gpu=name,compute_cap --format=csv` |
| SMs / L2 / regs | **82 SMs**, **6.0 MB L2** (persisting max **4.1 MB**), 65536 regs/SM | `~/claude-tmp/memprobe` (cudaGetDeviceProperties) |
| Clocks / power | SM 1950 MHz, mem 9501 MHz, power limit **250 W** | `nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.limit --format=csv` |
| VRAM | 24576 MiB nominal; **24126 MiB** as CUDA reports it → **450 MiB driver reserve** | `nvidia-smi --query-gpu=memory.total` vs `memprobe` `cudaMemGetInfo` |
| CUDA context cost | **265 MiB** (free 23861 of 24126 with a live context, card otherwise empty) | `memprobe` at idle |
| Driver | 595.84, open kernel module; nvidia-smi reports CUDA 13.2 | `cat /proc/driver/nvidia/version` |
| Toolkit | **CUDA 13.0.88** at `/usr/local/cuda-13.0` (this is what strata is built against) | `/usr/local/cuda-13.0/bin/nvcc --version` |
| GPU PCIe link | **8.0 GT/s × 16** (gen3 x16) — card is capable of 16.0 GT/s, the CPU root port is not | `cat /sys/bus/pci/devices/0000:01:00.0/current_link_{speed,width}` |
| CPU | i9-10900K, **10 cores / 20 threads**, max 5.3 GHz, **AVX2, no AVX-512** | `lscpu`; `grep -o avx512 /proc/cpuinfo` → empty |
| RAM | **64 GB DDR4-3200**, 4 × 16 GB, dual channel (≈51.2 GB/s theoretical) | `sudo dmidecode -t memory` |
| THP | `always [madvise] never` — strata is using 24 GB of anon hugepages | `cat /sys/kernel/mm/transparent_hugepage/enabled`; `grep AnonHugePages /proc/meminfo` |
| swappiness | 60 | `cat /proc/sys/vm/swappiness` |
| OS / kernel | Ubuntu 24.04.5, kernel 6.8.0-142-generic | `uname -r` |
| Boot/root NVMe | WD WDS100T3XHC (SN750), gen3 x4 | `cat /sys/class/nvme/nvme0/model` |
| Model NVMe | **Samsung 980 PRO 1TB** on `/srv/models` (ext4, 915.8 GB, **386 GB free**), gen3 x4, scheduler `none` | `cat /sys/class/nvme/nvme1/model`; `findmnt /srv/models`; `lspci` |

**PCIe gen3 x16 ≈ 15.75 GB/s raw, ~12 GB/s usable.** The `ram-expert-tier` branch
measured RAM→VRAM at **10.9 GB/s** on this box (`docs/RAM_EXPERT_TIER.md`), which
is the number to plan host→device transfers against.

## Storage bandwidth (measured, not spec)

`/srv/models/gguf/Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf` as the source:

| Pattern | Throughput | Command |
|---|---|---|
| 4 GiB, single stream, `iflag=direct` | **2.6 GB/s** | `dd bs=1M count=4096 iflag=direct` |
| 4 × 1 GiB, parallel, `iflag=direct` | **3.4 GB/s** aggregate | four backgrounded `dd … iflag=direct skip=…` |

So the SSD tier is worth **2.6 GB/s single-stream, 3.4 GB/s with 4 concurrent
readers**. Expert streaming must read concurrently, not serially.

## Memory and VRAM budget, with and without strata

Measured in one window (`glm53-idle-window.sh idle`), strata stopped and restarted:

| State | RAM `MemAvailable` | VRAM used | Card empties in |
|---|---|---|---|
| strata running (swift-1.5-iq3_xxs) | **2.5 GB** | 23841 MiB (strata 22396 + vision 1330) | — |
| strata stopped, card empty | **51.75 GB** | **1 MiB** (no display on the 3090) | 2 s |
| strata back up | 7.4 GB | 23771 MiB | **69 s** to first served request |

Two hard consequences:

1. **GLM 5.3 Flash cannot run beside strata.** With strata up there is 2.5 GB of
   RAM and 286 MiB of VRAM. The pinned-RAM expert tier needs tens of GiB. A
   benchmark window means strata stopped, and it costs 69 s to bring strata back.
2. **Swap is 100 % full on this box at all times** (`SwapFree: 152 kB` at idle,
   8191/8191 MB used in every sample). Any pinned-RAM tier must be sized against
   `MemAvailable`, and must not rely on swap.

Budget for the streamed GLM run, from these numbers: ~**23.9 GiB** usable VRAM
(CUDA view, minus the 265 MiB context), ~**50 GiB** usable pinned RAM with
strata stopped, ~**3.4 GB/s** of SSD behind that.

## Tenancy: how the card is actually obtained

`services/llmbox-strata.service` runs `tools/gpu-claim.sh strata` as
`ExecStartPre`: gpud decides whether the tenant may have the card, evicts what
was in the way, and fails the unit with the reason if it says no. ADR-123 makes
LLM and strata **one category with one slot**; ADR-131 makes strata the default
resident LLM. Practical rule for this work: stop `llmbox-strata`, benchmark,
start it, and confirm it is serving again before the window is called finished.

## Strata baseline (first measured row, to be extended in Step 6a)

From the live completion issued in the idle window (58-token prompt, 24 tokens
generated, `swift-1.5-iq3_xxs`, `--kv q4_0`, `--spec 4`, `--suffix-draft 5`,
`--mtp rt`, 262144 ctx, `--batch 2`):

- prefill **56.1 t/s**, decode **55.4 t/s** (`predicted_per_second` in the response's `timings`)

This is a short-prompt number and is not the comparison baseline. Step 6a must
measure strata on the same long prompt file the GLM benchmarks use.

## What this architecture is good and bad at, for a streamed MoE

- **Good:** 936 GB/s of GDDR6X and 82 SMs mean resident experts decode fast;
  6 MB L2 with a 4.1 MB persisting window is a real lever for hot expert tiles.
- **Bad:** the whole model does not fit. Q4 (178 GiB) and Q2 (90 GiB) both exceed
  24 GiB VRAM + 50 GiB pinned RAM, so the SSD tier is load-bearing, and it is
  2.6–3.4 GB/s against a 12 GB/s PCIe link and a 51 GB/s host.
- **Sharp edge:** no AVX-512, so any CPU-side compute of RAM-tier experts is
  AVX2-only, and only 10 physical cores exist to share with everything else.
- **Sharp edge:** Ampere has no FP8 tensor cores, so an FP8-native quant buys
  nothing on this card; bits-per-weight density is what matters, because it
  decides how many experts fit in VRAM and RAM.

---

# Step 1 — What the literature says about running a model bigger than the card

Each entry is: the source, what it establishes, and the optimization it points to
for GLM 5.3 Flash on this 24 GB card. Nothing here is unsourced.

## 1. KLD is the metric to rank quants with, and it needs a reference logit file

- **Source:** llama.cpp `tools/perplexity/README.md`
  (github.com/ggml-org/llama.cpp@a678916 `tools/perplexity/README.md`) — perplexity
  measures next-token prediction; `--kl-divergence-base path/to/logits.kld` records
  the reference logits from the full-precision model and then computes KL
  divergence of the quantized model against them.
- **Source:** llama.cpp PR #4739 (Q4_K/Q5_K introduction) — KLD computed with
  reference logits from the fp16 model on `wiki.test.raw`.
- **What it means here:** KLD is a *distribution* comparison against the
  full-precision model, so it detects the damage that perplexity on a small
  corpus hides. ds4 has no KLD tool (`grep -i kld` over the repo returns
  nothing), so Step 2 must build the measurement, not borrow it.
- **Optimization:** measure KLD of each candidate GLM 5.3 Flash quant against the
  official FP8 reference on the repo's own Z.AI FP8 fixture corpus
  (`gguf-tools/quality-testing/data/glm53-flash-openrouter-zai-fp8-100/`), because
  that corpus is the model's own distribution (chat + tool prompts), not generic
  wiki text.

## 2. Bits-per-weight is the lever that decides how many experts fit

- **Source:** arXiv 2601.14277, *Which Quantization Should I Use? A Unified
  Evaluation of llama.cpp Quantization on Llama-3.1-8B-Instruct* — effective bpw:
  Q3_K_S ≈ 3.44, Q3_K_M ≈ 3.53, Q3_K_L ≈ 3.6, Q4_K_S ≈ 4.2, Q4_K_M ≈ 4.5,
  Q5_K_S ≈ 5.2, Q5_K_M ≈ 5.5, Q6_K ≈ 6.5. Pareto-efficient set is compact:
  Q5_0 (accuracy-favouring), Q4_K_S (balanced default), Q3_K_L, Q3_K_M, Q3_K_S
  (max reduction). Q6_K/Q8_0 are dominated — more bits, no seat at the frontier.
- **Source:** same paper, §2.1 — increasing bits by one reduces modeled quantization
  variance by ≈4×; finer block grouping improves quality at fixed nominal bits.
- **What it means here:** on a card where the model does not fit, bytes are the
  scarce resource, so the ranking is by *quality per GiB*, not by quality.
- **Optimization:** candidate layouts for the expert tier are the low-bpw K-quants
  and I-quants, and the dense/non-routed weights are where extra bits should go
  (they are read by every token and stay in VRAM, so they cost transfer, not bits).
  ds4 already ships GLM routed paths in IQ2_XXS, Q2_K and Q4_K
  (`docs/MODELS.md`: "The routed paths include IQ2_XXS, Q2_K, and Q4_K").

## 3. imatrix is what makes a 2–3 bit expert quant usable

- **Source:** gguf-docs `importance-matrix.md` (github.com/iuliaturc/gguf-docs) —
  imatrix is orthogonal to the quant format; low-bit I-quants (IQ2) *require* an
  imatrix for decent quality; the core insight is that not all weights matter
  equally, and importance is measured by the effect of perturbing a weight.
- **Source:** llama.cpp PR #4861 (imatrix introduction) and llama.cpp discussion
  #5006 — imatrix computed on CPU from a calibration corpus; near-random token
  selection calibrated the matrix better than more of the same domain data
  (Mistral 7B q3_K_L: 8.3157 ppl with a near-random-calibrated matrix vs 8.4068
  with none, q8_0 reference 8.1901).
- **Source:** llama.cpp `tools/imatrix/README.md` — `llama-imatrix -m model -f corpus`.
- **What it means here:** ds4 already does this for GLM — the Q2 file "uses
  imatrix-guided IQ2_XXS gate/up and Q2_K down experts" (`docs/MODELS.md`), and
  `gguf-tools/imatrix/` holds a built dataset for DeepSeek
  (`gguf-tools/imatrix/dataset/build_ds4_imatrix_dataset.py`).
- **Optimization:** build a GLM 5.3 Flash imatrix from the repo's Z.AI FP8 prompt
  corpus before quantizing any custom layout. Density without imatrix is a
  quality cliff, and a density win bought with imatrix is the only way to buy
  more experts per GiB without paying for it in KLD.

## 4. Treat SSD, RAM and VRAM as one pipeline, and decide what to KEEP

- **Source:** arXiv 2609.18110, *SSD-LLaMA: SSD-Native Inference for
  Trillion-Parameter MoE at 1+ Token/s on a Consumer PC*, §2.3.1–2.3.2, §4.2 —
  three named gaps in existing systems: (a) expert tensors scattered across
  shards cause small fragmented reads and mmap page-fault I/O that cannot use
  SSD bandwidth; (b) systems optimize one transfer path, so SSD reads, RAM
  retention, H2D copies and VRAM compute barely overlap; (c) prediction answers
  *what to fetch* but not *what to keep*. Design: one aligned directly-accessible
  block per expert, concurrent direct reads, each completed read goes through
  pinned RAM into its VRAM slot without waiting for the others; RAM and VRAM are
  bounded caches with retention by observed frequency **and recency**; pinned
  host buffers in a fixed-size pool, reused.
- **Source:** same paper, §2.3.1 third gap — binding RAM-resident experts to CPU
  execution leaves the GPU idle; residency and execution placement must be
  decoupled.
- **Optimization for this box:** ds4's CUDA streaming path should (i) read whole
  expert blocks with concurrent O_DIRECT reads (measured 2.6 GB/s single-stream,
  3.4 GB/s at four readers — concurrency is where the bandwidth is), (ii) keep a
  fixed pinned pool rather than registering per transfer, and (iii) add a
  recency term to the eviction policy, not just a heat count. The `ram-expert-tier`
  branch already has tiers 1–2 and learned heat; recency and the decoupled
  execution placement are the pieces it does not have.

## 5. Prefetch with a two-layer lookahead, and budget PCIe against on-demand loads

- **Source:** arXiv 2509.23638, *PreScope*, §4.3–4.4, §6.5 — expert activation
  patterns shift sharply across layers, so gating-statistics predictors mispredict
  and waste PCIe; prefetch and on-demand loads compete for the same PCIe
  bandwidth and layer-by-layer execution always prioritizes on-demand, degrading
  prefetch into on-demand for the next layer; a cross-layer cost model decides
  per expert whether to load on-demand, prefetch, or run on CPU; prefetch buffers
  are split into current-layer and next-layer groups so prefetch stays one layer
  ahead; a two-layer lookahead takes prefetch off the critical path (CPU/GPU
  per-layer overhead gap within 8 ms for 50% of layers, peak 14 ms ≈ one expert
  prefetch).
- **Optimization for this box:** ds4's router lookahead (phase 5 of
  `docs/RAM_EXPERT_TIER.md`) is the right idea at one layer of depth; the paper's
  result says two layers, and that the lookahead must be *costed* — a prefetch
  that steals PCIe from an on-demand load on a 12 GB/s gen3 link can be a net
  loss. So: two-layer lookahead, with a cap on prefetch bytes per layer.

## 6. Speculative decoding doubles as a memory sensor

- **Source:** arXiv 2603.09983, *MoE-SpAc* — repurposes speculative decoding as
  an informative lookahead sensor for memory management, not only as a compute
  accelerator.
- **Source:** arXiv 2603.19289 — expert prefetching that uses currently computed
  internal representations to speculate future experts.
- **Source:** arXiv 2508.06978 — SSD offload energy analysis: SSD read energy per
  bit is substantially higher than DRAM, so a RAM hit is worth more than an
  equally fast SSD read.
- **What it means here:** GLM 5.3 Flash has a built-in MTP draft block
  (`docs/SPECULATIVE_DECODING.md`: "GLM's draft block is already in its main
  GGUF", commits up to two tokens). The drafted tokens are a free predictor of
  which experts the *next* token will route to.
- **Optimization:** run MTP speculation ahead of the expert cache and use the
  draft's routing as the prefetch request. This is the cheapest lookahead
  available on this box: no extra model file, no extra VRAM tier, and it is
  already in the binary.

## 7. The published reference number for exactly this workload

- **Source:** `docs/SSD_STREAMING.md` (this repo) — GLM 5.3 Flash Q4_K, 177.77 GiB,
  128 GB M5 Max, automatic cache, no speculative decoding: **121 t/s initial
  prefill, 104 t/s continued prefill, 11.9 / 14.9 t/s generation** (three-run
  median, 2K prompt + 1K append, 128 generated tokens per frontier).
- **What it means here:** this is the number the 3090 fork is trying to beat on a
  card with a fifth of the memory. It is also the number that says streaming GLM
  at 12–15 t/s is achievable, so the target is not a research gamble.
- **Source:** `docs/PERFORMANCE.md` (this repo) — the benchmark contract this goal
  must follow: same checkpoint, quantization, context and sampling; record the
  commit and whether weights were resident, streamed or distributed; keep other
  GPU workloads idle; repeat in alternating order; "one favorable run is not a
  speed result".

## 8. What the literature says NOT to do on this box

- **FP8 weights:** Ampere (sm_86) has no FP8 tensor cores; ds4 lists `glm53-fp8`
  as "Packaged native weights only; inference not implemented"
  (`docs/MODELS.md`). An FP8 artifact buys bytes and no speed here.
- **Page-cache/mmap streaming:** arXiv 2609.18110 §2.3.1 — page-level mmap
  cannot generate requests large enough to use SSD bandwidth. ds4's CUDA path
  already uses `pread` with the page cache dropped (`docs/RAM_EXPERT_TIER.md`),
  and the measured 2.6 GB/s direct figure is what to beat, not match.
- **Paging non-routed weights:** `docs/SSD_STREAMING.md` — non-routed weights are
  needed by every token and paging them delays generation; an oversized expert
  cache that displaces them is a regression, not a win.
