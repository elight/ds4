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
