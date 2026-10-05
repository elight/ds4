/* CPU routed-expert kernels and pool against a scalar dequantize-and-dot
 * reference.  No GPU, no model file: weights are random blocks with sane
 * scales.  Also reports the pool's expert throughput. */
#include "../ds4_cpu_experts.h"

#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint32_t rnd(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (uint32_t)rng;
}
static float frnd(void) { return (float)(rnd() & 0xffffff) / 16777216.0f * 2.0f - 1.0f; }

/* Random rows of the given type; the fp16 scales sit at fixed offsets. */
static void *make_matrix(uint32_t type, uint32_t K, uint32_t rows, float scale) {
    const uint64_t rb = ds4_cpu_expert_row_bytes(type, K), n = rb * rows;
    uint8_t *m = aligned_alloc(64, (n + 63) / 64 * 64);
    for (uint64_t i = 0; i < n; i++) m[i] = (uint8_t)rnd();
    if (type == DS4_CPU_EXPERT_MXFP4) {   /* E8M0 scales near scale / 8 */
        const int e0 = 128 + (int)lrintf(log2f(scale)) - 3;
        for (uint32_t r = 0; r < rows; r++)
            for (uint32_t b = 0; b < K / 32; b++) m[r * rb + b * 17] = (uint8_t)(e0 + (int)(rnd() % 3) - 1);
        return m;
    }
    const uint32_t nb = (K + 255) / 256;
    for (uint32_t r = 0; r < rows; r++)
        for (uint32_t b = 0; b < nb; b++) {
            if (type == DS4_CPU_EXPERT_Q2_K) {
                uint16_t *d = (uint16_t *)(m + r * rb + b * 84 + 80);
                d[0] = _cvtss_sh(scale * (0.5f + 0.5f * frnd()), 0);
                d[1] = _cvtss_sh(scale * 0.5f * (0.5f + 0.5f * frnd()), 0);
            } else {
                uint16_t *d = (uint16_t *)(m + r * rb + b * 66);
                d[0] = _cvtss_sh(scale * (0.5f + 0.5f * frnd()), 0);
            }
        }
    return m;
}

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* max |a-b| / max |b| and cosine similarity */
static void compare(const float *a, const float *b, uint32_t n, double *rel, double *cos) {
    double maxd = 0, maxb = 0, ab = 0, aa = 0, bb = 0;
    for (uint32_t i = 0; i < n; i++) {
        const double d = fabs((double)a[i] - b[i]);
        if (d > maxd) maxd = d;
        if (fabs(b[i]) > maxb) maxb = fabs(b[i]);
        ab += (double)a[i] * b[i]; aa += (double)a[i] * a[i]; bb += (double)b[i] * b[i];
    }
    *rel = maxb ? maxd / maxb : maxd;
    *cos = ab / sqrt(aa * bb + 1e-30);
}

static int run_shape(const char *name, ds4_cpu_expert_shape s, uint32_t n_experts, uint32_t n_tokens) {
    const uint32_t K = s.K, M = s.M, Mp = (M + 255) / 256 * 256;
    void **gate = malloc(n_experts * sizeof(void *)), **up = malloc(n_experts * sizeof(void *));
    void **down = malloc(n_experts * sizeof(void *));
    for (uint32_t e = 0; e < n_experts; e++) {
        gate[e] = make_matrix(s.gate_type, K, M, 0.02f);
        up[e] = make_matrix(s.gate_type, K, M, 0.02f);
        down[e] = make_matrix(s.down_type, Mp, K, 0.05f);
    }
    float *x = malloc((size_t)n_tokens * K * sizeof(float));
    for (uint32_t i = 0; i < n_tokens * K; i++) x[i] = frnd();
    const uint32_t n = n_experts;
    ds4_cpu_expert_job *jobs = calloc(n, sizeof(*jobs)), ref;
    float *out = calloc((size_t)n * K, sizeof(float)), *want = calloc(K, sizeof(float));
    float *want_f = calloc(K, sizeof(float));
    for (uint32_t j = 0; j < n; j++)
        jobs[j] = (ds4_cpu_expert_job){gate[j], up[j], down[j], x + (size_t)(j % n_tokens) * K,
                                       out + (size_t)j * K};
    int fail = 0;
    /* Many back-to-back batches: the pool must not mix one into the next. */
    for (int rep = 0; rep < 50; rep++) {
        memset(out, 0, (size_t)n * K * sizeof(float));
        if (!ds4_cpu_experts_submit(&s, jobs, n)) { fprintf(stderr, "submit failed\n"); return 1; }
        ds4_cpu_experts_wait();
    }
    double worst_q = 0, worst_cos = 1;
    for (uint32_t j = 0; j < n; j++) {
        ref = jobs[j];
        ref.out = want;
        ds4_cpu_expert_ref(&s, &ref, 1);
        ref.out = want_f;
        ds4_cpu_expert_ref(&s, &ref, 0);
        double rel, cos, relf, cosf;
        compare(out + (size_t)j * K, want, K, &rel, &cos);
        compare(out + (size_t)j * K, want_f, K, &relf, &cosf);
        if (rel > worst_q) worst_q = rel;
        if (cosf < worst_cos) worst_cos = cosf;
    }
    /* Same activation quantization as the reference: only summation order
     * differs.  Against float activations (what the GPU computes) the Q8_K
     * rounding shows, but the direction must hold. */
    if (worst_q > 2e-3 || worst_cos < 0.995) fail = 1;
    const int reps = 200;
    const double t0 = now();
    for (int r = 0; r < reps; r++) {
        ds4_cpu_experts_submit(&s, jobs, n);
        ds4_cpu_experts_wait();
    }
    const double dt = (now() - t0) / reps;
    const double bytes = (double)n * (2.0 * ds4_cpu_expert_row_bytes(s.gate_type, K) * M +
                                      (double)ds4_cpu_expert_row_bytes(s.down_type, Mp) * K);
    printf("%-26s %s  vs-q8-ref rel %.2e  vs-float cos %.6f  %u experts %.3f ms  %.1f GB/s\n",
           name, fail ? "FAIL" : "ok  ", worst_q, worst_cos, n, dt * 1e3, bytes / dt / 1e9);
    for (uint32_t e = 0; e < n_experts; e++) { free(gate[e]); free(up[e]); free(down[e]); }
    free(gate); free(up); free(down); free(x); free(jobs); free(out); free(want); free(want_f);
    return fail;
}

int main(void) {
    const uint32_t threads = ds4_cpu_experts_start();
    printf("pool threads: %u\n", threads);
    int fail = 0;
    /* Qwen3.8 Flash Next Q2: IQ2_XXS gate/up, Q2_K down, 2560 x 640. */
    const ds4_cpu_expert_shape qwen = {2560, 640, DS4_CPU_EXPERT_IQ2_XXS, DS4_CPU_EXPERT_Q2_K};
    fail |= run_shape("qwen38 1 token x 1", qwen, 1, 1);
    fail |= run_shape("qwen38 1 token x 6", qwen, 6, 1);
    fail |= run_shape("qwen38 2 tokens x 12", qwen, 12, 2);
    /* All-Q2_K experts, the MiMo layout, at a smaller width for speed. */
    const ds4_cpu_expert_shape q2k = {1024, 512, DS4_CPU_EXPERT_Q2_K, DS4_CPU_EXPERT_Q2_K};
    fail |= run_shape("q2_K/q2_K 1 token x 4", q2k, 4, 1);
    /* IQ2_XXS down too. */
    const ds4_cpu_expert_shape iq = {512, 256, DS4_CPU_EXPERT_IQ2_XXS, DS4_CPU_EXPERT_IQ2_XXS};
    fail |= run_shape("iq2_xxs/iq2_xxs 1 x 3", iq, 3, 1);
    /* MiMo V2.6 Flash: Q2_K gate/up, MXFP4 down, 4096 x 2048. */
    const ds4_cpu_expert_shape mimo = {4096, 2048, DS4_CPU_EXPERT_Q2_K, DS4_CPU_EXPERT_MXFP4};
    fail |= run_shape("mimo2 1 token x 1", mimo, 1, 1);
    fail |= run_shape("mimo2 1 token x 8", mimo, 8, 1);
    printf(fail ? "FAILED\n" : "all passed\n");
    return fail;
}
