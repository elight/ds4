/* CPU side of the hybrid MoE decode: routed experts that live in the pinned
 * RAM tier are computed here, in place, while the GPU runs the experts that
 * live in its slot cache.  See docs/RAM_EXPERT_TIER.md, phase 4. */
#ifndef DS4_CPU_EXPERTS_H
#define DS4_CPU_EXPERTS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* GGUF tensor types the kernels handle. */
#define DS4_CPU_EXPERT_Q2_K    10u
#define DS4_CPU_EXPERT_IQ2_XXS 16u
#define DS4_CPU_EXPERT_MXFP4   39u  /* down projection only */

/* One routed expert applied to one token: out = down(silu(gate x) * up x). */
typedef struct {
    const void *gate, *up, *down; /* this expert's matrices, row-major blocks */
    const float *x;               /* K activations */
    float *out;                   /* K outputs */
} ds4_cpu_expert_job;

/* Geometry and types shared by every job in a batch.  K is the model width,
 * M the expert's hidden width; down's rows are M padded up to 256. */
typedef struct {
    uint32_t K, M, gate_type, down_type;
} ds4_cpu_expert_shape;

/* True when the kernels support this shape on this CPU. */
int ds4_cpu_experts_supported(const ds4_cpu_expert_shape *shape);

/* Starts the pool on first use: one worker per physical core except the
 * calling thread's, which joins in ds4_cpu_experts_wait.  Returns the number
 * of threads that compute, the caller included. */
uint32_t ds4_cpu_experts_start(void);

/* Queues a batch and returns at once; workers start on it immediately.
 * The jobs array and every buffer it names must stay live until wait. */
int ds4_cpu_experts_submit(const ds4_cpu_expert_shape *shape,
                           const ds4_cpu_expert_job *jobs, uint32_t n_jobs);

/* The caller works on the batch too, then blocks until it is finished.
 * Returns seconds the caller spent waiting with nothing left to take. */
double ds4_cpu_experts_wait(void);

/* Scalar reference for tests: dequantizes and computes in float, with the
 * activations quantized the way the AVX2 path quantizes them. */
void ds4_cpu_expert_ref(const ds4_cpu_expert_shape *shape, const ds4_cpu_expert_job *job,
                        int quantize_activations);

/* Bytes of one row of a matrix of this type and width (K padded to 256). */
uint64_t ds4_cpu_expert_row_bytes(uint32_t type, uint32_t K);

#ifdef __cplusplus
}
#endif

#endif
