# Strata tricks, enumerated and ranked for the ds4 GLM 5.3 Flash path (Step 4)

Enumerated from the Strata tree at `/srv/models/gguf/strata`: the engine's own
`--help` (`./engine/strata --help`, 262 lines) and `docs/`, not from its README.
Every row's numbers are Strata's own published measurements, with the file they
come from. "Lands in" names the ds4 file the change belongs to.

Verdict legend: **PORT** = ds4 lacks it and it applies to a streamed GLM run on
one 3090; **HAVE** = ds4 already does this; **N/A** = needs hardware this box
does not have; **SKIP** = measured not to pay, by Strata itself.

## Tier 1 — port first: measured wins on a card this size

| # | Trick | Strata source | Measured | Lands in | Verdict |
|---|---|---|---|---|---|
| 1 | **Read-ahead on every bulk load**: `madvise`/`posix_fadvise WILLNEED` in 128 KiB steps for weights, the RAM copy, the cache fill and the MTP files | `docs/DETAILS.md` §Speed | Fill **39 MB/s → 3.2 GB/s**; ready in **70 s instead of ~920 s** on a Gen3 NVMe | `ds4_ssd.c` (startup fill), `ds4_cuda.cu` | **PORT** — this box is Gen3 NVMe at 2.6 GB/s single-stream measured; a 178 GiB Q4_K fill is exactly the 920-second case |
| 2 | **Per-layer expert slots, not one shared counter**: the default arrival-order policy lets the first positions eat all slots | `engine --help` `--expert-cache-per-layer` | Same routing: **2.97% → 21.4%** at 8 slots/layer, **70.4%** at 64 slots/layer | `ds4_cuda.cu` slot cache, `ds4_ssd.c` | **PORT** — cheapest known win in the whole list; it is a policy change, not a kernel |
| 3 | **KV streaming**: keep only the attention window in VRAM, whole K/V in pinned RAM | `engine --help` `--kv-resident N`; `docs/DETAILS.md` | Q2_0 at 262K: **50.9 → 62.6 t/s** (1,589 → 3,872 experts in VRAM); at 128K **+6%**; costs ~13.7 KB RAM per context token | `ds4_kvstore.c` | **PORT** — on this box VRAM is the binding tier and RAM is the loose one (51.75 GB with strata off) |
| 4 | **4-bit KV after Hadamard rotation** (`--kv q4_0`), and the hybrid `--kv k8v4`: INT8 K (exact scores) + rotated Q4_0 V | `engine --help` `--kv q4_0`, `--kv k8v4` | q4_0 = **half of int8's** memory; k8v4 = **816 B/cell vs 1,056** | `ds4_kvstore.c` | **PORT** — 24 GB card, long context; every KV byte is an expert slot not spent |
| 5 | **Exchange-buffer rotation**: the evicted expert's temporary buffer *becomes* the resident slot, so the copy back into RAM disappears | `docs/EXCHANGE_ROTATION.md` (`STRATA_EXCHANGE_ROTATE`) | **104.90 → 113.44 t/s (+8.1%)**, 3,802 exchanges, **5,255,884,800 bytes** of host copies avoided, **output IDs identical** (1,024 tokens, token-identical A/B) | `ds4_cuda.cu` (RAM↔VRAM exchange) | **PORT** — measured, bit-identical, and it removes work from the decode path |
| 6 | **Profile-seeded cache + learned profile persistence**: seed the VRAM tier from a routing profile instead of admitting on first use, and save what the tier learned | `engine --help` `--expert-profile`, `--expert-profile-save` | Strata's shipped profile is the default start; learned profile is written on clean exit and every 10 min | `ds4_streaming_hotlist*.inc` (ds4's existing hotlist), `ds4_ssd.c` | **PORT** — ds4 already ships a GLM hotlist (`ds4_streaming_hotlist_glm52.inc`, 6509 lines) but has no way to *learn and persist* one on this box |
| 7 | **Doorbell poke after the launch** so the GPU starts while the CPU pool runs | `engine --help` `--no-hit-poke` (the A/B arm) | Without the poke the work "waits for the next driver entry and **does not overlap at all**" | `ds4_cuda.cu` | **PORT** — ds4's RAM-tier CPU compute (phase 4 of `docs/RAM_EXPERT_TIER.md`) needs exactly this |
| 8 | **Host-thread core placement**: put the host loop on the *last* physical core, workers on the others | `engine --help` `--host-core last`, `--pool-workers`, `--pool-affinity` | Strata: the driver sends a GPU's interrupts to one logical processor, usually the first, "and a spinning host there waits for them" | `ds4_cuda.cu` / CPU pool | **PORT** — this box has 10 physical cores and AVX2 only; core choice is a real lever, and the host thread is currently unpinned |
| 9 | **Speculation as the decode accelerator with a measured acceptance rate**: MTP draft layer + suffix/prompt-lookup drafts, window grows where a repeat is likely | `engine --help` `--spec`, `--spec-min-p`, `--suffix-draft`, `--lookup-chain`; `docs/INTEL.md` | Suffix drafter only **20.2–20.8 t/s** → with the MTP layer **42.7 t/s** (80% accepted, 2.9 tokens/round); on that card `--spec 4` (16.9) was *worse* than `--spec 2` (20.8) because a round costs ~53 ms whatever its size | ds4 GLM MTP path (`--mtp`, `docs/SPECULATIVE_DECODING.md`) | **PORT** — GLM's MTP block is already in the GGUF; the ported part is *measuring acceptance and costing the window*, not enabling MTP |
| 10 | **Prefill chunk sized by what the expert cache can lend** (`--prefill auto` = largest chunk whose buffers the cache can lend) | `engine --help` `--prefill auto` | Strata's prefill path streams missing experts per chunk; the Arc port measured **790 tok/s** prompt reading after fixing its streaming ring | `ds4.c` prefill, `ds4_cuda.cu` | **PORT** — ds4's GLM path "chooses its own chunks and rejects `--prefill-chunk`" (`docs/PERFORMANCE.md`), so the lever is the *cache-lend* budget, not the chunk flag |

## Tier 2 — port after the first benchmark round

| # | Trick | Strata source | Measured | Lands in | Verdict |
|---|---|---|---|---|---|
| 11 | **Resident RAM complement of the GPU cache**, page-locked, with adaptive swaps that never re-read the file | `engine --help` `--resident-cpu-experts`, `--resident-experts`, `--shared-expert-arena` (on `/dev/shm`) | AMD doc: the complement of the static GPU cache is copied to RAM; swaps exchange experts "without reading the file" | `ds4_cuda.cu`, `docs/RAM_EXPERT_TIER.md` phase 2 | **HAVE (partial)** — the `ram-expert-tier` branch has the pinned tier; the *swap-without-reread* and `/dev/shm` backing are the missing halves |
| 12 | **Adaptive tier advances between verify windows** (`--adapt-async 1`) | `engine --help` | "the swaps advance between verify windows instead of a window waiting for a whole round. Not bit-exact run to run" | `ds4_cuda.cu` | **PORT, gated** — it trades determinism for speed, so it must pass the accuracy gate, not just the speed gate |
| 13 | **Direct I/O for the n-gram/PLE table with a bounded row cache and deep read queue** (`--ple-io direct`, `--ple-row-cache`, `--ple-inflight 256`, `--no-ple-prefetch` A/B) | `engine --help` | direct = "the table never enters RAM or the file cache"; 256 outstanding reads by default | `ds4_engram.c` | **PORT** — ds4's engram reader is the same shape; the Qwen file's 95 GiB n-gram table is why that file is 147 GB (`docs/RAM_EXPERT_TIER.md`) |
| 14 | **Prompt/conversation cache between requests** (`--prompt-cache N`, `--conversation-cache-mib`, `--conversation-cache-slots`, `--prompt-cache-tail`, `--short-read`) | `engine --help` | ~118 MB of RAM per checkpoint, 6 kept by default; llmbox runs it at 8192 MiB / 4 slots | `ds4_prompt_prefix.c`, `ds4_server.c` | **PORT** — this is the trick that makes llmbox's chat pattern (same long system prompt every turn) cheap |
| 15 | **Message-boundary checkpoint for edited history** (`STRATA_CACHE_MESSAGE_BOUNDARY`) | `docs/MESSAGE_BOUNDARY_CACHE.md` | First 100-record edit **17.236 → 11.527 s (-33.1%)**; reused prefix 22,016 → 32,831 tokens; mixed suite **-4.4%**; cold 8K requests **+0.2 s** | `ds4_prompt_prefix.c` | **PORT, conditional** — pays only for edit-heavy sessions; the +0.2 s cold cost is a real loss to record |
| 16 | **Elastic expert cache in segments** so VRAM can be handed back between requests (`--vram-elastic`, `--vram-segment-mib 512`) | `docs/VRAM_ELASTIC.md` | Only the expert cache moves; dense weights, K/V, MTP head, context and prompt buffers stay | `ds4_cuda.cu` | **PORT** — this is how a GLM tenant coexists with tts/vision on one 24 GB card |
| 17 | **Batch slots with a measured cost** (`--batch N`, `--batch-mtp`) | `docs/BATCHING.md` | On a 24 GB card, Q2_0 at 32K = **3 slots**; on 12–16 GB cards a batch costs **10–25% per request**; setup recommends parallel only where experts mostly fit VRAM | `ds4_server.c` | **PORT, gated** — for a streamed GLM the experts mostly do *not* fit, so Strata's own rule says this is a latency win and a per-request loss |
| 18 | **Idle unload / load-gate / pre-load hook** (`--idle-unload`, `--min-free-vram-mib`, `--before-load`) | `docs/DETAILS.md` | Unload ~0.3 s; answers 503 "the GPU is in use by another program" instead of starting into a card someone else is using | `ds4_server.c` | **PORT** — this is the gpud-facing behaviour a tenant on this box needs |

## Tier 3 — measurement and correctness machinery (adopt for Step 6a)

| # | Trick | Strata source | What it gives | Lands in | Verdict |
|---|---|---|---|---|---|
| 19 | **Window hashes**: per verify window, index, position, size, a 64-bit hash of the final residual rows and argmaxes — "two builds that do the same arithmetic write the same file" | `engine --help` `--window-hashes`, `--spec-follow` | A bit-level equality test between builds on the *same* generated text | `ds4_bench.c`, `ds4_eval.c` | **PORT** — this is the accuracy half of the accept/reject gate |
| 20 | **True GPU floor measurement**: `--gpu-only-full` (pre+post graphs for all layers plus LM head, no CPU pool) and `--no-pool`; `--graph-only` explicitly labelled NOT the floor | `engine --help` | Separates "the card is the limit" from "the host is the limit" | `ds4_bench.c` | **PORT** — without it, a benchmark cannot say which tier to attack next |
| 21 | **Per-tier accounting in the log**: `expert tiers` (blobs from the RAM copy, blobs and MB from files, time reading them), `routing prefetch`, `ram_blobs`/`file_blobs`/`file_mb` per request, `drafts_offered`/`drafts_accepted` | `docs/DETAILS.md` §Speed | The hit-rate numbers this goal's contract demands, reported per request | `ds4_server.c`, `ds4_bench.c` | **PORT** — the contract asks for VRAM/RAM hit % per optimization; this is how to produce it |
| 22 | **The hit-rate trap, stated plainly**: `hit_rate` is the VRAM share of experts looked up, and experts fetched over PCIe are not in it, "so a higher `--pcie-frac` raises it even when decoding gets slower"; `pcie_share` is the honest companion metric | `docs/DETAILS.md` §Speed | A named way for a benchmark to lie to you | benchmark notes | **ADOPT** — record `pcie_share` alongside hit rate, never hit rate alone |
| 23 | **Slot verification against the file**: read every filled cache slot back and compare with the GGUF (`STRATA_VERIFY_ALL_SLOTS`), plus an arena aliasing check | `docs/INTEL.md` | Found a real driver bug (two 2 MiB pages mapped onto the same memory) that produced NaN at layer 4 | `ds4_cuda.cu` startup | **PORT** — cheap at startup (Strata: under 0.5 s), catches a class of silent corruption |
| 24 | **Determinism switch with its cost published**: `STRATA_IQ_MT_MIN=1` makes the answer independent of drafting, at **-1..-3%** decode | `docs/DETAILS.md` | The price of reproducibility, stated as a number | benchmark notes | **ADOPT** — the accept/reject gate needs to know which arm is deterministic |

## Tier 4 — measured not to pay, by Strata itself (record as rejected)

| # | Trick | Strata source | Measured | Verdict |
|---|---|---|---|---|
| 25 | **Batched DMA submission** (`cudaMemcpyBatchAsync`, `STRATA_DMA_BATCH`) | `docs/BATCHED_DMA.md` | Submission latency **-82.7%**, total transfer **-1.3%**; full-model screen "did not establish a generation-speed gain" (30.3 vs 30.0 t/s); Strata's own conclusion: "does not justify changing the default" | **SKIP** — and it is the model for how to write a rejected row: the number, the reason, and the source |
| 26 | **mmap-based expert reads** (`--mmap-experts`) | `engine --help` | **71.97 vs 34.78 ms/token cold vs warm** — the rate depends on the OS page cache holding 34 GB | **SKIP** — on a 64 GB box with strata's 49 GB resident, the warm case does not exist |
| 27 | **Sampled drafting** (coupled draft, rejection sampling) | `docs/DETAILS.md` | 42.4/43.2/43.3 default vs 40.6/42.4/43.3 coupled vs 41.0/42.6/42.6 rejection, with up to 5% run-to-run spread: "Neither is faster in a way that holds up" | **SKIP** |

## N/A on this box (needs hardware llmbox does not have)

- `--peer-device` / `--peer-reserve-mib` / `--peer-prefill-rows` — a second GPU as
  an adaptive expert tier over P2P (`docs/SECOND_GPU.md`). One 3090.
- `--expert-cache-device1..3`, `--layer-split`, `--batch-groups`,
  `--pipeline-windows`, `--trim-stage-weights` — multi-GPU layer split.
- `STRATA_DISJOINT_ADAPT` — avoiding duplicate slots across helper and primary
  GPUs (`docs/DISJOINT_EXPERT_CACHE.md`).
- `--native-*` CUDA-oracle pinned paths — Strata's own labels are
  "experimental"; they are per-port A/B arms, not shipped wins, and the measured
  numbers quoted for them are on other cards.

## What this list says about the shape of the work

The top of the list is not kernels. Two of the ten Tier-1 items are cache
*policy* (per-layer slots, profile seeding), two are *I/O scheduling*
(read-ahead, deep read queues), two are *memory placement* (KV streaming, 4-bit
KV), and one is removing a copy (exchange rotation). All of them are
measurable with `ds4-bench` before any new kernel is written.
