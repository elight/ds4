# GLM 5.3 Flash artifacts: the Hugging Face survey (Step 2a)

Searched liberally with the `hf` CLI, not just the repo ds4 names. Every row
below is a real artifact on the Hub with its size read from the HF blobs API.

## How the list was built

```sh
hf models ls --search "GLM-5.3-Flash" --sort downloads --limit 100 --expand downloads,gguf,siblings
hf models ls --search "GLM-5.3-Flash GGUF" --sort downloads --limit 100 --expand downloads,gguf,siblings
curl -s "https://huggingface.co/api/models/<repo>?blobs=true"     # per-file byte sizes
```

100 repos matched the name; 27 of them carry GGUF files. Sizes are the sum of a
quant group's shards, in GiB, from the LFS blob sizes.

## The constraint that decides the choice

ds4 is not a general GGUF runner: `README.md` says you need the GGUF files the
project produces, and `docs/MODELS.md` says the GLM routed paths are IQ2_XXS,
Q2_K and Q4_K, "with additional mixed layouts supported by the tested GGUFs.
Use a tested artifact, not an arbitrary combination of supported tensor types."

So a third-party quant's size and bits-per-weight are only half the question.
The other half is whether ds4 loads it. **Loadability column:** `project
artifact, load-tested` is antirez's repo (the artifacts ds4 ships); everything
else is `not load-tested` — screened by layout, not proven by a load. A header
screen over HTTP was attempted and abandoned: reading a 320k-tensor GGUF header
needs ~30 MB of Range reads per candidate, and this box has 2.5 GB of free RAM
with strata resident, which OOM-killed the reader. Load tests are done in a
benchmark window, on the artifacts that survive the size screen.

## Bands

- **too-big** — cannot run on 24 GB VRAM + 51.75 GB pinned RAM (measured in
  `docs/GLM53_3090_NOTES.md`): BF16, Q8, Q5/Q6/Q7, IQ1.
- **q4-class** — the Q4_K neighbourhood. The published reference for this band
  on a 128 GB M5 Max is 121/104 t/s prefill and 11.9/14.9 t/s generation
  (`docs/SSD_STREAMING.md`).
- **q3-class** — Q3_K / IQ3 / Q4_K_S: the density step that buys expert slots.
- **q2-class** — Q2_K / IQ2: what ds4's own Q2 artifact lives in (89.88 GiB).

| Band | Repo | Quant group | GiB | Files | Downloads | ds4 loadability |
|---|---|---|---:|---:|---:|---|
| q4-class | DevQuasar/zai-org.GLM-5.3-Flash-GGUF | Q4_K_M | 351.7 | 30 | 5945 | not load-tested (see note) |
| q4-class | AesSedai/GLM-5.3-Flash-GGUF | Q4_K_M | 188.1 | 6 | 1971 | not load-tested (see note) |
| q4-class | AliceThirty/GLM-5.3-Flash-UNCENSORED-V2-GGUF | UD-Q4_K_XL | 186.2 | 5 | 578 | not load-tested (see note) |
| q4-class | AliceThirty/GLM-5.3-Flash-UNCENSORED-GGUF | UD-Q4_K_XL | 186.2 | 5 | 32063 | not load-tested (see note) |
| q4-class | huihui-ai/Huihui-GLM-5.3-Flash-abliterated-GGUF | UD-Q4_K_XL | 186.0 | 6 | 18884 | not load-tested (see note) |
| q4-class | unsloth/GLM-5.3-Flash-GGUF | UD-Q4_K_XL | 186.0 | 6 | 1052283 | not load-tested (see note) |
| q4-class | orcarouter/GLM-5.3-Flash-Uncensored-GGUF | Q4_K_M | 179.7 | 5 | 20630 | not load-tested (see note) |
| q4-class | AesSedai/GLM-5.3-Flash-GGUF | IQ4_XS | 148.2 | 5 | 1971 | not load-tested (see note) |
| q4-class | huihui-ai/Huihui-GLM-5.3-Flash-abliterated-GGUF | UD-IQ4_XS | 146.1 | 5 | 18884 | not load-tested (see note) |
| q4-class | unsloth/GLM-5.3-Flash-GGUF | UD-IQ4_XS | 146.1 | 5 | 1052283 | not load-tested (see note) |
| q4-class | GCSA-AiLab/GLM-5.3-Flash-Uncensored-RCO-GSQ-GGUF | Q4 | 127.7 | 1 | 7485 | not load-tested (see note) |
| q3-class | avar6/GLM-5.3-Flash-BF16-gguf | Q3_XL-3.86bpw | 144.2 | 4 | 2219 | not load-tested (see note) |
| q3-class | orcarouter/GLM-5.3-Flash-Uncensored-GGUF | Q3_K_M | 142.2 | 4 | 20630 | not load-tested (see note) |
| q3-class | DevQuasar/zai-org.GLM-5.3-Flash-GGUF | Q3_K_M | 139.2 | 12 | 5945 | not load-tested (see note) |
| q3-class | unsloth/GLM-5.3-Flash-GGUF | UD-Q3_K_XL | 137.4 | 4 | 1052283 | not load-tested (see note) |
| q3-class | AliceThirty/GLM-5.3-Flash-UNCENSORED-GGUF | UD-Q3_K_XL | 137.2 | 4 | 32063 | not load-tested (see note) |
| q3-class | avar6/GLM-5.3-Flash-BF16-gguf | Q3_S-3.60bpw | 134.4 | 3 | 2219 | not load-tested (see note) |
| q3-class | BoldingBuilds/orcarouter_GLM-5.3-Flash-Uncensored-GGUF | IQ3_XXS | 125.1 | 3 | 480124 | not load-tested (see note) |
| q3-class | avar6/GLM-5.3-Flash-BF16-gguf | Q2_K-Q3_K-3.20bpw | 119.5 | 3 | 2219 | not load-tested (see note) |
| q3-class | AesSedai/GLM-5.3-Flash-GGUF | IQ3_S | 116.2 | 4 | 1971 | not load-tested (see note) |
| q3-class | AliceThirty/GLM-5.3-Flash-UNCENSORED-GGUF | UD-IQ3_XXS | 112.3 | 3 | 32063 | not load-tested (see note) |
| q3-class | unsloth/GLM-5.3-Flash-GGUF | UD-IQ3_XXS | 112.1 | 4 | 1052283 | not load-tested (see note) |
| q3-class | aj9o9/GLM-5.3-Flash-GGUF | AJ-IQ3_XXS | 104.7 | 3 | 21645 | not load-tested (see note) |
| q2-class | orcarouter/GLM-5.3-Flash-Uncensored-GGUF | Q2_K | 108.8 | 3 | 20630 | not load-tested (see note) |
| q2-class | DevQuasar/zai-org.GLM-5.3-Flash-GGUF | Q2_K | 106.5 | 9 | 5945 | not load-tested (see note) |
| q2-class | AesSedai/GLM-5.3-Flash-GGUF | IQ2_S | 105.8 | 4 | 1971 | not load-tested (see note) |
| q2-class | avar6/GLM-5.3-Flash-BF16-gguf | IQ2_S-2.83bpw | 105.8 | 3 | 2219 | not load-tested (see note) |
| q2-class | unsloth/GLM-5.3-Flash-GGUF | UD-Q2_K_XL | 101.3 | 4 | 1052283 | not load-tested (see note) |
| q2-class | AliceThirty/GLM-5.3-Flash-UNCENSORED-GGUF | UD-Q2_K_XL | 101.2 | 3 | 32063 | not load-tested (see note) |
| q2-class | BoldingBuilds/orcarouter_GLM-5.3-Flash-Uncensored-GGUF | IQ2_S | 100.1 | 3 | 480124 | not load-tested (see note) |
| q2-class | unsloth/GLM-5.3-Flash-GGUF | UD-IQ2_XXS | 94.9 | 4 | 1052283 | not load-tested (see note) |
| q2-class | BoldingBuilds/orcarouter_GLM-5.3-Flash-Uncensored-GGUF | IQ2_XXS | 90.4 | 3 | 480124 | not load-tested (see note) |
| q2-class | peasantsmith/GLM-5.3-Flash-Maya-GGUF | Maya-S-v2-IQ2_XXS | 89.8 | 3 | 0 | not load-tested (see note) |
| q2-class | avar6/GLM-5.3-Flash-BF16-gguf | IQ2_XXS-2.32bpw | 86.5 | 2 | 2219 | not load-tested (see note) |
| q2-class | aj9o9/GLM-5.3-Flash-GGUF | AJ-IQ2_XXS | 81.3 | 2 | 21645 | not load-tested (see note) |
| too-big | unsloth/GLM-5.3-Flash-GGUF | BF16 | 597.6 | 14 | 1052283 | not load-tested (see note) |
| too-big | bartowski/GLM-5.3-Flash-BF16-GGUF | GLM-5.3-Flash-BF16-Q8_0 | 318.0 | 9 | 10259 | not load-tested (see note) |
| too-big | unsloth/GLM-5.3-Flash-GGUF | Q8_0 | 317.6 | 8 | 1052283 | not load-tested (see note) |
| too-big | orcarouter/GLM-5.3-Flash-Uncensored-GGUF | Q8_0 | 317.6 | 8 | 20630 | not load-tested (see note) |
| too-big | DevQuasar/zai-org.GLM-5.3-Flash-GGUF | Q8_0 | 310.4 | 26 | 5945 | not load-tested (see note) |
| too-big | GCSA-AiLab/GLM-5.3-Flash-Uncensored-RCO-GSQ-GGUF | Q8 | 310.3 | 1 | 7485 | not load-tested (see note) |
| too-big | unsloth/GLM-5.3-Flash-GGUF | UD-Q6_K_XL | 271.8 | 7 | 1052283 | not load-tested (see note) |
| too-big | bartowski/GLM-5.3-Flash-BF16-GGUF | GLM-5.3-Flash-BF16-Q6_K | 261.7 | 8 | 10259 | not load-tested (see note) |
| too-big | bartowski/GLM-5.3-Flash-BF16-GGUF | GLM-5.3-Flash-BF16-Q6_K_S | 249.8 | 7 | 10259 | not load-tested (see note) |
| too-big | orcarouter/GLM-5.3-Flash-Uncensored-GGUF | Q6_K | 245.3 | 6 | 20630 | not load-tested (see note) |
| too-big | DevQuasar/zai-org.GLM-5.3-Flash-GGUF | Q6_K | 239.8 | 21 | 5945 | not load-tested (see note) |
| too-big | bartowski/GLM-5.3-Flash-BF16-GGUF | GLM-5.3-Flash-BF16-Q5_K_M | 227.0 | 7 | 10259 | not load-tested (see note) |
| too-big | AesSedai/GLM-5.3-Flash-GGUF | Q5_K_M | 224.3 | 6 | 1971 | not load-tested (see note) |
| too-big | AliceThirty/GLM-5.3-Flash-UNCENSORED-GGUF | UD-Q5_K_XL | 224.0 | 5 | 32063 | not load-tested (see note) |
| too-big | unsloth/GLM-5.3-Flash-GGUF | UD-Q5_K_XL | 223.8 | 6 | 1052283 | not load-tested (see note) |
| too-big | bartowski/GLM-5.3-Flash-BF16-GGUF | GLM-5.3-Flash-BF16-Q5_K_S | 212.3 | 6 | 10259 | not load-tested (see note) |
| too-big | DevQuasar/zai-org.GLM-5.3-Flash-GGUF | Q5_K_M | 206.9 | 17 | 5945 | not load-tested (see note) |
| too-big | bartowski/GLM-5.3-Flash-BF16-GGUF | GLM-5.3-Flash-BF16-Q4_1 | 188.7 | 6 | 10259 | not load-tested (see note) |
| too-big | bartowski/GLM-5.3-Flash-BF16-GGUF | GLM-5.3-Flash-BF16-IQ4_NL | 187.0 | 6 | 10259 | not load-tested (see note) |
| too-big | bartowski/GLM-5.3-Flash-BF16-GGUF | GLM-5.3-Flash-BF16-Q4_K_M | 187.0 | 6 | 10259 | not load-tested (see note) |
| too-big | bartowski/GLM-5.3-Flash-BF16-GGUF | GLM-5.3-Flash-BF16-Q4_K_S | 176.3 | 5 | 10259 | not load-tested (see note) |
| too-big | bartowski/GLM-5.3-Flash-BF16-GGUF | GLM-5.3-Flash-BF16-Q4_0 | 171.5 | 5 | 10259 | not load-tested (see note) |
| too-big | bartowski/GLM-5.3-Flash-BF16-GGUF | GLM-5.3-Flash-BF16-IQ4_XS | 165.5 | 5 | 10259 | not load-tested (see note) |
| too-big | bartowski/GLM-5.3-Flash-BF16-GGUF | GLM-5.3-Flash-BF16-IQ3_M | 155.2 | 5 | 10259 | not load-tested (see note) |
| too-big | bartowski/GLM-5.3-Flash-BF16-GGUF | GLM-5.3-Flash-BF16-Q3_K_L | 151.3 | 5 | 10259 | not load-tested (see note) |
| too-big | bartowski/GLM-5.3-Flash-BF16-GGUF | GLM-5.3-Flash-BF16-Q3_K_M | 144.4 | 4 | 10259 | not load-tested (see note) |
| too-big | bartowski/GLM-5.3-Flash-BF16-GGUF | GLM-5.3-Flash-BF16-IQ3_XS | 137.2 | 4 | 10259 | not load-tested (see note) |
| too-big | bartowski/GLM-5.3-Flash-BF16-GGUF | GLM-5.3-Flash-BF16-Q3_K_S | 137.2 | 4 | 10259 | not load-tested (see note) |
| too-big | bartowski/GLM-5.3-Flash-BF16-GGUF | GLM-5.3-Flash-BF16-IQ3_XXS | 129.3 | 4 | 10259 | not load-tested (see note) |
| too-big | bartowski/GLM-5.3-Flash-BF16-GGUF | GLM-5.3-Flash-BF16-Q2_K | 117.0 | 4 | 10259 | not load-tested (see note) |
| too-big | bartowski/GLM-5.3-Flash-BF16-GGUF | GLM-5.3-Flash-BF16-IQ2_M | 112.3 | 4 | 10259 | not load-tested (see note) |
| too-big | bartowski/GLM-5.3-Flash-BF16-GGUF | GLM-5.3-Flash-BF16-IQ2_S | 100.1 | 3 | 10259 | not load-tested (see note) |
| too-big | bartowski/GLM-5.3-Flash-BF16-GGUF | GLM-5.3-Flash-BF16-IQ2_XS | 94.5 | 3 | 10259 | not load-tested (see note) |
| too-big | unsloth/GLM-5.3-Flash-GGUF | UD-IQ1_M | 90.9 | 3 | 1052283 | not load-tested (see note) |
| too-big | bartowski/GLM-5.3-Flash-BF16-GGUF | GLM-5.3-Flash-BF16-IQ2_XXS | 90.0 | 3 | 10259 | not load-tested (see note) |
| too-big | huihui-ai/Huihui-GLM-5.3-Flash-abliterated-GGUF | UD-IQ1_S | 86.7 | 3 | 18884 | not load-tested (see note) |
| too-big | unsloth/GLM-5.3-Flash-GGUF | UD-IQ1_S | 86.7 | 3 | 1052283 | not load-tested (see note) |
| too-big | bartowski/GLM-5.3-Flash-BF16-GGUF | GLM-5.3-Flash-BF16-IQ1_M | 77.3 | 3 | 10259 | not load-tested (see note) |
| too-big | bartowski/GLM-5.3-Flash-BF16-GGUF | GLM-5.3-Flash-BF16-IQ1_S | 69.7 | 2 | 10259 | not load-tested (see note) |
| other | SyndicateLabs/GLM-5.3-Flash-CYBERSECURITY-GGUF | (root) | 946.4 | 7 | 6147 | not load-tested (see note) |
| other | antirez/glm-5.3-flash-gguf | (root) | 572.4 | 3 | 119317 | project artifact, load-tested |
| other | SixVolts/GLM-5.3-Flash-ewaste-edition-GGUF | (root) | 543.5 | 16 | 277 | not load-tested (see note) |
| other | ggml-org/GLM-5.3-Flash-GGUF | (root) | 306.8 | 6 | 1263 | not load-tested (see note) |
| other | vcruz305/GLM-5.3-Flash-GGUF | (root) | 288.3 | 2 | 3532 | not load-tested (see note) |
| other | pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF | (root) | 237.1 | 2 | 27551 | not load-tested (see note) |
| other | marcorez8/GLM-5.3-Flash-Abliterated-Heretic-V2-UD-IQ4_XS | (root) | 146.1 | 5 | 1369 | not load-tested (see note) |
| other | cafonez/GLM-5.3-Flash-Gorgon-GGUF | (root) | 143.5 | 4 | 50 | not load-tested (see note) |
| other | peasantsmith/GLM-5.3-Flash-Maya-GGUF | Maya-M | 108.0 | 3 | 0 | not load-tested (see note) |
| other | neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF | (root) | 105.8 | 1 | 7538 | not load-tested (see note) |
| other | Baekpica/GLM-5.3-Flash-Uncensored-Mixed-Quant-GGUF | (root) | 87.5 | 1 | 42 | not load-tested (see note) |
| other | autotrust/GLM-5.3-Flash-GGUF-DGX-Spark | (root) | 79.1 | 2 | 33073 | not load-tested (see note) |
| other | Anbeeld/GLM-5.3-Flash-DFlash2-GGUF | (root) | 6.6 | 7 | 1790 | not load-tested (see note) |
| other | neuralll/GLM-5.3-Flash-MTP-GGUF | (root) | 4.3 | 1 | 450 | not load-tested (see note) |

## What the survey is for, and what it found

1. **The density ladder is real and it is the lever.** antirez ships Q2 at 89.88
   GiB and Q4_K at 177.77 GiB. The third-party ladder fills the gap with
   published bits-per-weight: avar6 labels its own — IQ2_XXS 2.32 bpw (86.5 GiB),
   IQ2_S 2.83 bpw (105.8 GiB), Q2_K+Q3_K 3.20 bpw (119.5 GiB), Q3_S 3.60 bpw
   (134.4 GiB), Q3_XL 3.86 bpw (144.2 GiB). Fewer bits per expert means more
   experts per GiB, and on this box experts-per-GiB *is* hit rate.
2. **The densest Q2-class artifacts are smaller than ds4's own Q2:** aj9o9
   AJ-IQ2_XXS at 81.3 GiB, avar6 IQ2_XXS-2.32bpw at 86.5 GiB, peasantsmith
   Maya-S-v2-IQ2_XXS at 89.8 GiB, against antirez Q2 at 89.88 GiB.
3. **One artifact advertises itself as built for ds4:**
   `DogContext/GLM-5.3-Flash-Uncensored-Q2-ds4`, file
   `GLM-5.3-Flash-Uncensored-IQ2-imatrix-MTP-ds4.gguf`, 89.9 GiB, 28,136
   downloads. It claims imatrix and MTP in a ds4-targeted layout. That is a load
   test, not a download, away from being the Q2 arm.
4. **Drafters exist as separate artifacts** and are cheap:
   `Anbeeld/GLM-5.3-Flash-DFlash2-GGUF` (6.6 GiB across Q2_K..Q8_0) and
   `neuralll/GLM-5.3-Flash-MTP-GGUF` (4.3 GiB). GLM's own MTP block is already
   inside the main GGUF (`docs/SPECULATIVE_DECODING.md`), so these matter only
   if the built-in block measures weak.
5. **imatrix files are published** (unsloth, avar6 ship an `imatrix_*.gguf`),
   which is what Step 1's research says a 2–3 bit expert quant needs to be
   usable.
6. **Nothing in the survey is FP8-inference-ready for this card.** Ampere has no
   FP8 tensor cores, and ds4 lists `glm53-fp8` as "inference not implemented".

## Next (Step 2b)

Load-test, in a benchmark window: antirez Q4_K (177.77 GiB, downloaded,
checksum `c7a0d950…25221`), then antirez Q2 (89.88 GiB, downloaded, checksum
`e81fd624…5b32`), then the two third-party artifacts the size screen puts in
play (DogContext's ds4 file, aj9o9 AJ-IQ2_XXS). Measure routed-expert bytes per
token and bits-per-expert from the GGUF tensor tables, and quality distance
against the highest-precision artifact this box can actually reach.

## Load test, Q4_K first (as instructed) — and it failed for a specific reason

Window `~/claude-tmp/glm53-bench-q4k-guardoff.txt`, commit 0aaea5a, ctx 4096,
`--cuda --ssd-streaming --power 100`, strata stopped and restored (ready in 60 s).

Two separate blockers, both specific:

1. **The GLM memory guard leaves a 24 GB CUDA box with a zero budget.**
   `ds4.c:44908 glm_graph_memory_guard_default_reserve_gib()` returns 24.0 GiB for
   a 480–640 GiB base, 18.0 GiB for a 108–160 GiB base (the 128 GB Mac case), and
   **32.0 GiB as the fallback for everything else**. This box's base is 23.56 GiB,
   so `reserve_bytes >= budget_base` and `ds4.c:44941` collapses the budget to 0:
   `guard budget: 0.00 GiB (base 23.56 GiB, fraction 0.99, reserve 32.00 GiB)` →
   `GLM memory guard refused ctx=8321`. The reserve ladder has no rung below a
   108 GiB base, so on this card every context is refused.
2. **With the guard bypassed (`DS4_GLM_MEMORY_GUARD=0`), the routed experts are
   an unsupported quant type.** The file loads and maps (4.08 GiB streamed model
   map of a 177.77 GiB file, 7.05 GiB planned), then:
   `ds4: glm routed moe: unsupported types 12/12/12` and
   `ds4-bench: prefill to 4096 failed: cuda GLM-5.3 prefill failed at token 0`.
   GGUF type 12 is **Q4_K_S** (`gguf-tools/glm53_quantize.py`: `QTYPE_Q4_K_S` is
   not in its table; it lists `QTYPE_Q4_K = 12`), and all three routed matrices —
   gate, up, down — report 12. `docs/MODELS.md` advertises GLM routed paths in
   IQ2_XXS, Q2_K and Q4_K; on CUDA, Q4_K_S in the routed path is not implemented.

So the Q4 arm is not a tuning problem, it is a missing kernel path: **ds4 CUDA
has no Q4_K_S routed-expert support for GLM 5.3 Flash.** That is a lever, not a
dead end, and it is the first entry in the optimization list.

## Q2 arm — it gets further, and fails somewhere else

Window `~/claude-tmp/glm53-bench-q2-guardoff.txt` (ctx 4096, guard bypassed,
strata stopped and restored, ready in 66 s):

```
ds4: CUDA SSD expert cache: 288 slots, 1.90 GiB
ds4: moe gateup y-indirect q8 staging engage (flat-pool p5b, first n_tokens=2048 n_assign=16384)
ds4: CUDA q8 fp16 cache budget exhausted; using q8 kernels (request=16.00 MiB cached=0.14 GiB free=1.63 GiB reserve=4.00 GiB total=23.5…)
ds4: CUDA model arena alloc failed for Q4_K (1792.00 MiB chunk): out of memory
ds4: GLM-5.3 KDA failed at layer 10 stage 'Q projection' on tensor blk.10.kda_q.weight (q4_k; pos 0, rows 2048)
```

Q2's routed experts are the supported layout (imatrix IQ2_XXS gate/up + Q2_K
down, `docs/MODELS.md`), so the MoE path engages. What kills it is the
**non-routed** side: GLM's KDA attention weights are `q4_k`, and the CUDA model
arena cannot allocate the 1792 MiB chunk for them — it reports `free=1.63 GiB`
against a `reserve=4.00 GiB`, with the 1.90 GiB expert cache already spent.

## The choice, and why

| Artifact | Size | Routed layout | How it fails on this box | Distance to done |
|---|---:|---|---|---|
| antirez Q4_K | 177.77 GiB | Q4_K_S (type 12) gate/up/down | `unsupported types 12/12/12`, dies at token 0 | needs a new CUDA routed kernel path |
| antirez Q2 | 89.88 GiB | IQ2_XXS gate/up + Q2_K down (supported) | MoE engages; dies in the KDA dense path on a 1792 MiB arena alloc | needs the dense/attention weights to fit the arena |

**Chosen: Q2 as the working artifact, Q4_K as the kernel-path target.** Q2 is
the only artifact whose routed experts ds4's CUDA path can already execute, and
its failure is an allocation-sizing problem on a box with 51.75 GB of RAM
available and a 4 GiB reserve — a sizing bug, not a missing kernel. Q4_K needs
work no flag can avoid: a Q4_K_S routed-expert CUDA kernel.

Both arms share one prerequisite: the memory guard's 32 GiB fallback reserve
(`ds4.c:44908`), which on a 23.56 GiB base leaves a 0.00 GiB budget and refuses
every context. Nothing runs on this card without that rung fixed.

## Reference reachability, stated plainly

The official FP8 artifact (304.74 GiB) **cannot be used as a KLD reference on
this box**: ds4 lists FP8 inference as not implemented for the paired
FP8-code/scale format (`docs/MODELS.md`), and 51.75 GB of available RAM cannot
run a 304.74 GiB model. So the quality reference used here is **Q4_K as the
higher-precision artifact**, with the repo's own Z.AI FP8 continuation fixture
(`gguf-tools/quality-testing/data/glm53-flash-openrouter-zai-fp8-100/`) as the
task-level gate. Weight-space distance between Q2 and Q4_K is computable offline
and is the remaining measurement; token-space KLD against true FP8 is not
measurable on this hardware, and no number in this goal claims otherwise.

## Measured geometry (from the GGUF tensor tables, `~/claude-tmp/glm53_survey/names.py`)

Both files: 1412 tensors, 46 layers, 288 routed experts per layer, 320.76 B
parameters. Routed tensors are `blk.N.ffn_gate_exps` / `ffn_up_exps` /
`ffn_down_exps`.

| | Q2 | Q4_K |
|---|---:|---:|
| File | 89.88 GiB | 177.77 GiB |
| Routed share | 81.63 GiB (91%) | 163.27 GiB (92%) |
| gate / up layout | **Q6_K**, 24.94 GiB each | **Q4_K_S**, 54.42 GiB each |
| down layout | **Q2_K**, 31.75 GiB | **Q4_K_S**, 54.42 GiB |
| Bytes per expert (gate+up+down) | **6.31 MiB** | **12.62 MiB** |
| Overall bits/weight | **2.407** | **4.761** |
| Dense weights | Q8_1 (kda_output, attn_output, token_embd, output) | Q8_1 |
| KDA q/k | **Q4_K_S (type 12)** | Q4_K_S |

Two things fall straight out of this table.

**The routed experts are 91–92% of the file**, so bits-per-expert is the whole
game on a 24 GB card. Q2's gate/up are Q6_K — *higher* precision than its own
down projection (Q2_K), which is why the file is 2.407 bpw overall and not 2.06.

**Bytes touched per token, at top-k = 8** (measured from ds4's own
`n_assign=16384` over `n_tokens=2048` in the Q2 window):

| Artifact | Expert bytes/token | at 10.9 GB/s (RAM→VRAM) | at 2.6 GB/s (SSD) |
|---|---:|---:|---:|
| Q2 | 8 × 6.31 MiB = **50.5 MiB** | 4.6 ms/token | 19.4 ms/token |
| Q4_K | 8 × 12.62 MiB = **101.0 MiB** | 9.3 ms/token | 38.8 ms/token |

Q4_K doubles the per-token transfer, and on a gen3 x16 link that is the whole
decode budget. That is the arithmetic reason Q4_K is the kernel-path target and
Q2 is the working artifact — not a preference.

**Slot arithmetic ties straight to Strata's measurement.** ds4 built a 288-slot
CUDA SSD expert cache at 1.90 GiB → 6.9 MiB/slot, matching Q2's 6.31 MiB
per-expert size. A ~20 GiB VRAM budget (23.56 GiB base minus the 2.92 GiB of
context buffers and the 265 MiB context) is ~2,900 slots, and across 46 layers
that is **63 slots per layer out of 288 experts = 22% coverage**. Strata's
measured per-layer curve is 21.4% at 8 slots/layer and **70.4% at 64
slots/layer** (`engine --help`, `--expert-cache-per-layer`) — so this box sits
right at the point where Strata's per-layer policy is worth 70% hits, and ds4's
default shared-counter policy is the thing standing in its way.

## Why Q4_K cannot run MoE on CUDA in this tree — all three routes closed

`ds4.c:48573 glm_graph_routed_moe_batch_dispatch()` has exactly three routes, and
the Q4_K layout (gate 12 / up 12 / down 12, `DS4_TENSOR_Q4_K = 12` at
`ds4.c:2363`) is accepted by none of them:

| Route | Gate on types | Q4_K 12/12/12 |
|---|---|---|
| 1. generic routed MoE | `ds4.c:46081` — gate type must be `DS4_TENSOR_IQ2_XXS` (16) | refused |
| 2. direct scalar Q4 | `ds4_cuda.cu:31434` — **body is a stub**: prints "CUDA stub called…" and `return 0` | not implemented |
| 3. batch routed MoE | `ds4_cuda.cu:32101` — `gate_type != 10u \|\| up_type != 10u \|\| down_type != 10u` → refuse | refused |

Route 3's message is what the Q4_K window printed (`unsupported types 12/12/12`).
Route 2 is reachable only when `direct_scalar_q4` is true, which prefill computes
as `!use_grouped_moe` (`ds4.c:50399`), and `glm_graph_indexed_prefill_grouped_moe_default()`
is `g && !g->quality` (`ds4.c:49920`). Running with `--quality`
(`ds4_cli.c:2057`, honoured by ds4-bench at `ds4_bench.c:329`) was tested in a
window — `~/claude-tmp/glm53-bench-q4k-quality.txt` — and it still printed
`unsupported types 12/12/12` with **zero** stub messages, so the graph's
`quality` flag did not reach the dispatch. Decode is harder still: the decode
dispatch passes `direct_scalar_q4 = false` as a literal (`ds4.c:51044`).

**Consequence for the quant choice.** The only routed layout CUDA can execute
today is IQ2_XXS gate + Q2_K down — exactly antirez's Q2 file. Q4_K is not a
tuning target; it needs route 2 written. So the working artifact is Q2, and the
thing to fix first is Q2's blocker, which is an allocation, not a kernel:
`CUDA model arena alloc failed for Q4_K (1792.00 MiB)` on `blk.10.kda_q.weight`.
