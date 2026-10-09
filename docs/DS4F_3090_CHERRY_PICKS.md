# Cherry-picks from glm53-flash-3090 onto ds4-flash-3090

Recorded 2026-10-09. Each candidate is picked or rejected with the evidence that
decided it. The GLM branch's numbers are its own measurements on this card; the
DeepSeek-path deltas are measured on this branch and land in the lever-sweep
notes, not here.

## Picked

| Commit | What | Why it applies to the DeepSeek path |
|---|---|---|
| `77043cb` ← `eb39e8f` | `DS4_CUDA_STREAM_EXPERT_RESERVE_MB` makes the streaming expert cache reserve tunable | Applied clean to `origin/main`. It is the knob the lever sweep needs — without it the reserve is fixed and the cache ceiling cannot be swept. |
| `0a6c067` ← `640a4f0` | `ldmatrix.sync.aligned.m8n8.x4.b16` → `.x4.shared.b16` (and the x2 / x2.trans forms) | Applied clean. This is an Ampere PTX correctness fix, not a model change: the `.shared` qualifier is the spelled form the ptxas on this card wants. Verified present after the pick. |
| `9dd326f` ← `5a697a8` | Sub-108 GiB reserve rung: scale the reserve to the base (`base_gib / 8`, floor 2.0) | Code-only pick; the GLM notes file is not carried on this branch, so provenance is recorded in the comment at `ds4.c:44928`. The condition is the card's size, not the model: on a 24 GB card the 32 GiB fallback reserve exceeds the reported base, collapses the budget to zero, zeroes the streaming expert cache, and leaves decode streaming every expert. Measured on the GLM path as 1.15 → 1.44 t/s. |

Rebuilt clean after all three: `make cuda CUDA_ARCH=sm_86`.

## Rejected, with the evidence

**`b9b527f` — "drop the rejected per-layer expert eviction scan": nothing to
drop.** `git log -S "per_layer"` is empty on `origin/main` *and* on
`glm53-flash-3090`, and `grep -n "per_layer" ds4_cuda.cu` is empty in this tree.
The scan main would be shedding never existed on main. The pick conflicts in
`ds4_cuda.cu` for exactly that reason — it deletes lines main does not have.
No-op, correctly skipped.

**`8fa5b25` — `DS4_GLM53_PREFILL_CHUNK`: does not apply, DeepSeek already has
its own tunable.** The GLM constant is used only inside GLM-gated code:
`ds4.c:52133` is `(g->glm53 && n_tokens > DS4_GLM53_PREFILL_CHUNK_TOKENS)`
inside a function whose other guard is `glm_graph_span_fits_context()`. The
DeepSeek path goes through `ds4_effective_prefill_chunk()` (`ds4.c:14301`) and
`ds4_prefill_cap_for_prompt()` (`ds4.c:14310`), which already expose
`DS4_METAL_PREFILL_CHUNK` and default to 4096 (8192 for PRO). So the DeepSeek
equivalent the contract asked for **already exists**; picking the GLM override
would add a second knob over a path that has one.

Worth keeping from that commit's notes, because it constrains the sweep: on the
GLM path the prefill chunk was measured **not** to be the limit (2048 → 63.81
t/s, 4096 → 63.74 t/s, 1024 → fails). Prefill batches expert reuse across the
chunk, which is why prefill costs 15.7 ms/token while decode costs 870 ms/token
for the same experts. So prefill chunk is not a decode lever, and the sweep
should not spend a rung on it.

## Not candidates, recorded so they are not re-litigated

- `d94870c`, `aaa23f0`, `ea4b843`, `9d29c1c`, `1e2f36d`, `fea29a1`, `2eccbf8`,
  `4cc42b0`, `02f56a4`, `e24c2ec` — notes-only commits on the GLM branch. Their
  content is GLM-specific and lives in `docs/GLM53_*.md` on that branch; the
  DeepSeek-relevant conclusions are carried into `docs/DS4F_3090_PRIOR_ART.md`
  and this file rather than duplicated.
- `d94870c`'s finding is the exception worth naming: `DS4_CUDA_Q8_F16_CACHE_MB=1024`
  is what makes a large expert cache fit, because the Q8→F16 staging cache and
  the streaming expert cache draw from the same VRAM. That is a sweep rung here,
  not a cherry-pick — the env var already exists on main at `ds4_cuda.cu:1488`.
