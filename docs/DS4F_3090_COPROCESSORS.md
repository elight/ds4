# Coprocessors: the M4 Max (24 GB) and the iPad Air on llmbox

Report-only study, 2026-10-09. Nothing here gates the DS V4 Flash goal. Every
number is a measurement with its command; the two things I could not measure are
named, with what stopped them.

## What is actually on the wire — measured

| Fact | Value | Command |
|---|---|---|
| llmbox's only live LAN link | `enp3s0`, **1000 Mb**, driver `r8169` | `cat /sys/class/net/enp3s0/speed` |
| Second NIC | `enp5s0`, driver `igc` (Intel 2.5G part), **state down** — no cable | `cat /sys/class/net/enp5s0/operstate` |
| Thunderbolt on this box | **None.** Only a Comet Lake USB 3.1 xHCI is present | `lspci \| grep -i thunderbolt` → empty |
| M4 Max identity | `am=Mac16,8` = M4 Max MacBook Pro, `Evans-MacBook-Pro.local`, **192.168.1.39** | `avahi-browse -r -t _raop._tcp` |
| RTT to the M4 Max | min/avg/max = **3.082 / 5.322 / 10.890 ms**, 0% loss, n=10 | `ping -c 10 -i 0.3 192.168.1.39` |
| iPad Air on the LAN | **Not discovered.** LAN Apple devices are the iPhone (192.168.1.26), a MacBook Air M1 (`MacBookAir10,1`, 192.168.1.52), and the M4 Max | `avahi-browse -a -p` |
| ds4 build targets for the iPad | **No iOS/iPadOS target exists** — no `ios`/`ipad`/`android` in `README.md`, `Makefile`, `docs/METAL.md` | grep over those files |

Two measurements I could not take, and what stopped them:

- **Mac-side TCP throughput.** No iperf3 server on `192.168.1.39:5201` (connect
  failed) and **SSH port 22 is refused** (`ssh: connect to host 192.168.1.39
  port 22: Connection refused`). Measuring it would mean installing something on
  the Mac, which the goal puts out of bounds.
- **iperf3 is not installed on llmbox** either, so I have no local wire-rate
  measurement to fall back on.

The conclusion below does not depend on the missing throughput number, because
the NIC reports a **1000 Mb link rate**, which is a physical ceiling: ~116 MB/s
(0.116 GB/s) of TCP at best.

## The arithmetic that decides it

llmbox's bottleneck is the storage tier: **2.6 GB/s single-stream, 3.4 GB/s with
four concurrent readers**, measured on this box (`docs/GLM53_3090_NOTES.md`).

| Path | Throughput | vs local NVMe |
|---|---:|---:|
| Local NVMe, 4 readers | 3.4 GB/s | 1× |
| 1 GbE TCP ceiling (measured link rate) | ~0.116 GB/s | **~29× slower** |
| 2.5 GbE, if `enp5s0` were cabled | ~0.29 GB/s | ~11× slower |
| Thunderbolt 4 RDMA | ~5 GB/s | 1.5× faster — **not available, no Thunderbolt on this box** |

**Rule 1 — anything that moves weights is a no-go.** A remote RAM tier is only
worth having if it beats the local SSD. At 1 GbE it is 29× worse, and cabling
the 2.5G port would still leave it 11× worse. There is no configuration of this
LAN that makes a remote expert tier better than the NVMe it would replace.

**Rule 2 — the roles that move activations and tokens are a different question,
and they are bandwidth-free.** A drafted token is a token id (4 bytes); a
pipeline stage passes a hidden-state row (4096 × 4 B = 16 KB per token per
layer). Both are trivial against 116 MB/s. Those roles are judged on **latency**,
and the measured RTT is 5.3 ms average.

## M4 Max (24 GB) — verdict

**As a host: no.** DS V4 Flash Q2 is ~81 GB; 24 GB unified memory cannot hold it,
and ds4's documented Mac targets for this model are 96/128 GB machines.

**As a remote expert/RAM tier: no.** Decided by Rule 1 — 29× slower than the
local NVMe, and 24 GB minus macOS headroom is ~16 GB of usable tier anyway.

**As a 50/50 tensor-parallel peer: no.** `docs/DISTRIBUTED.md` documents tensor
parallelism as Mac+Mac or Spark+Spark, and it needs RDMA over Thunderbolt, which
this box does not have.

**As a pipeline-parallel stage: possible, and not worth it.** Pipeline
parallelism is the one cross-architecture mode ds4 documents ("Activations
travel from one stage to the next over TCP"), so it is technically reachable.
It is a bad trade here: a stage boundary serialises the two machines, llmbox
would idle behind a peer that is 29× slower at moving anything bulky, and the
activation traffic is the only part that is cheap.

**As a speculative drafter host: this is the one real idea, and it is the
strongest coprocessor case the prior art allows.** The argument is specific:

`docs/DS4F_3090_PRIOR_ART.md` records that DSpark speculation on the 4×3090 rig
was a **net loss** — 27.5% acceptance with a Q4_K draft, "e2e is slower than
plain decode because the draft runs uncaptured and costs VRAM". Both of those
costs are properties of running the drafter *on the inference card*. Moving the
drafter to the Mac removes exactly those two costs: it costs llmbox no VRAM, and
the draft is no longer an uncaptured CUDA graph on the verifier's card.

The latency side is favourable because the verifier is slow. At the GLM-measured
1.76 t/s a token costs 568 ms, so a draft round costing ~45 ms (5.3 ms RTT plus
draft compute) is ~8% of a single token. With k=4 at 27.5% acceptance, expected
accepted tokens ≈ 1.1, so gross speedup ≈ 2.1× and net ≈ 1.9× — 1.76 → **~3.4
t/s** if ds4 verifies drafted tokens in parallel.

**Assumption (b) is now answered, and the answer is no.** ds4 has no path for an
externally-supplied draft:

- `--mtp` and `--dspark` are parsed at `ds4_cli.c:2019` and `ds4_cli.c:2030`, and
  both select a drafter that runs **in the same process** as the verifier.
- DSpark is loaded as a support artifact and described by model metadata —
  `DS4_SUPPORT_DSPARK` and the `deepseek4.dspark.*` keys read in
  `model_dspark_summary()` at `ds4.c:3016` — not received over the wire.
- `ds4_server.c` has no endpoint that takes proposed token ids. Grepping it for
  draft/speculation surfaces only `decode_speculative` (an internal flag) and
  unrelated uses of the word "spec".
- `./ds4 --help` (77 lines) exposes no flag accepting foreign drafts.

So the drafter-coprocessor idea is a **no-go on the current code**, and the
reason is architectural, not the network: the verifier has no entry point for
proposals from another machine. The favourable latency arithmetic above is
therefore never spent, and assumption (a) — the drafter's own tokens/second on
the M4 Max — does not need measuring.

The cost to change that, stated so it can be declined knowingly: a draft-ingestion
path is a server endpoint taking proposed token ids plus the acceptance and
rollback handling in `ds4.c` around the existing speculative path. That is
implementation work in ds4, not a configuration change, and it is out of scope
for a report-only study. If it is ever built, the network is not the binding
constraint — 5.3 ms RTT against a 568 ms verifier token is 1%. The binding
constraint is the work itself.

## iPad Air M4 (12 GB) — verdict

**No, and the reason is not the network.** ds4 has no iOS/iPadOS build target,
so there is nothing to run on it. Even granting a port, 12 GB is below any
drafter worth verifying against a 13B-active model, and it is Wi-Fi-only, so its
RTT is worse than the Mac's measured 5.3 ms.

The honest summary: the iPad is a client, not a coprocessor. It is a fine
consumer of a DS V4 Flash endpoint served by llmbox, and that is the whole of it.

## What would change the verdict

Ranked by what it costs to get:

1. **Cable `enp5s0`** (the igc 2.5G port that is present and down) → ~0.29 GB/s.
   Still 11× worse than local NVMe, so it changes no weight-moving verdict. It
   does make the drafter case slightly worse in latency terms and is not worth
   doing for this purpose.
2. **Add a Thunderbolt PCIe card to llmbox** → TB4 RDMA ~5 GB/s, 1.5× the local
   NVMe. This is the only change that makes a remote tier beat local storage,
   and it is the prerequisite for ds4's documented RDMA modes. It costs a card,
   Linux Thunderbolt bring-up, and it still does not give the Mac more than
   ~16 GB of usable tier at 24 GB.
3. **A bigger Mac** (128 GB) → then it is not a coprocessor at all; it is a
   complete DS V4 Flash host, which is the published 96/128 GB target. Worth
   naming because it is the cheapest path to *running this model well* — it is
   just not a coprocessor, and it is not this box.

## Status: closed

The study's one open question is answered and the answer is no, so no Mac-side
work is proposed and nothing further is spent on it.

| Machine | Host | Remote expert tier | TP peer | Pipeline stage | Drafter host |
|---|---|---|---|---|---|
| M4 Max, 24 GB | no — 81 GB model | no — 29× slower than local NVMe | no — Mac+Mac, needs TB RDMA | possible, not worth it | **no — verifier has no draft ingestion path** |
| iPad Air, 12 GB | no | no | no | no | **no — ds4 has no iOS target** |

Both coprocessor roles that survive the bandwidth test die on software: the
verifier cannot accept foreign drafts, and ds4 does not build for iPadOS. The
M4 Max's real value to this goal is as the published 96/128 GB host for a
*fatter* quant — a machine that runs DS V4 Flash well on its own, which is a
different question from coprocessing and is not this box.
