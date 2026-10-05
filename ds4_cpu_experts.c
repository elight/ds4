/* CPU routed-expert kernels and worker pool for the hybrid MoE decode.
 *
 * The AVX2 dot products (IQ2_XXS x Q8_K, Q2_K x Q8_K) and the Q8_K
 * activation format are ported from llama.cpp's ggml-cpu
 * (ggml/src/ggml-cpu/arch/x86/quants.c), MIT License, Copyright (c) 2023-2026
 * The ggml authors.  The pool's shape - one worker per physical core, the
 * host thread pinned and joining the work, a spin-then-sleep doorbell - and
 * the idea of computing RAM-resident experts in place while the GPU runs its
 * resident ones come from Strata (github.com/Niko1221/Strata,
 * src/kernels/cpu/pool.cpp and expert.cpp), MIT License, Copyright (c) 2025
 * Niko1221.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions: The above
 * copyright notice and this permission notice shall be included in all copies
 * or substantial portions of the Software.  THE SOFTWARE IS PROVIDED "AS IS",
 * WITHOUT WARRANTY OF ANY KIND.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "ds4_cpu_experts.h"

#include <linux/futex.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#if defined(__AVX2__) && defined(__F16C__) && defined(__FMA__)
#include <immintrin.h>
#define DS4_CPU_EXPERTS_SIMD 1
#else
#define DS4_CPU_EXPERTS_SIMD 0
#endif

#define QK_K 256

typedef struct {
    uint8_t  scales[QK_K / 16];
    uint8_t  qs[QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} cpu_block_q2_K;

typedef struct {
    uint16_t d;
    uint16_t qs[QK_K / 8];
} cpu_block_iq2_xxs;

typedef struct {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K / 16];
} cpu_block_q8_K;

_Static_assert(sizeof(cpu_block_q2_K) == 84, "q2_K block");
_Static_assert(sizeof(cpu_block_iq2_xxs) == 66, "iq2_xxs block");

/* MXFP4: 32 weights per 17-byte block, an E8M0 scale then 16 bytes of 4-bit
 * E2M1 codes, weight j in the low nibble of byte j and weight j+16 in the
 * high one (ggml block_mxfp4). */
#define QK_MXFP4 32
#define MXFP4_BLOCK 17
static const int8_t kvalues_mxfp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

/* ggml's GGML_E8M0_TO_FP32_HALF: 2^(e-128), the half folded into the codes. */
static inline float e8m0_half(uint8_t e) {
    const uint32_t bits = e < 2 ? 0x00200000u << e : (uint32_t)(e - 1) << 23;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

#include "ds4_cpu_experts_grid.inc"

/* Sign patterns: seven stored bits, the eighth makes the count of minus signs
 * even.  keven_signs[i] holds the eight signs of pattern i as +1/-1 bytes. */
static int8_t keven_signs[128][8];
static pthread_once_t tables_once = PTHREAD_ONCE_INIT;

static void tables_init(void) {
    for (int i = 0; i < 128; i++) {
        const int parity = __builtin_popcount(i) & 1;
        for (int j = 0; j < 7; j++) keven_signs[i][j] = (i >> j) & 1 ? -1 : 1;
        keven_signs[i][7] = parity ? -1 : 1;
    }
}

static inline float fp16(uint16_t h) {
#if DS4_CPU_EXPERTS_SIMD
    return _cvtsh_ss(h);
#else
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    int32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff, bits;
    if (exp == 0) {
        if (!mant) bits = sign;
        else {
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3ff;
            bits = sign | (uint32_t)(exp + 112) << 23 | mant << 13;
        }
    } else if (exp == 31) bits = sign | 0x7f800000u | mant << 13;
    else bits = sign | (uint32_t)(exp + 112) << 23 | mant << 13;
    float f;
    memcpy(&f, &bits, 4);
    return f;
#endif
}

uint64_t ds4_cpu_expert_row_bytes(uint32_t type, uint32_t K) {
    const uint64_t blocks = (K + QK_K - 1) / QK_K;
    switch (type) {
    case DS4_CPU_EXPERT_Q2_K: return blocks * sizeof(cpu_block_q2_K);
    case DS4_CPU_EXPERT_IQ2_XXS: return blocks * sizeof(cpu_block_iq2_xxs);
    case DS4_CPU_EXPERT_MXFP4: return (uint64_t)(K + QK_MXFP4 - 1) / QK_MXFP4 * MXFP4_BLOCK;
    default: return 0;
    }
}

/* llama.cpp quantize_row_q8_K_ref. */
static void quantize_q8_K(const float *x, cpu_block_q8_K *y, uint32_t k) {
    for (uint32_t b = 0; b < k / QK_K; b++, x += QK_K) {
        float max = 0, amax = 0;
        for (int j = 0; j < QK_K; j++) {
            const float ax = fabsf(x[j]);
            if (ax > amax) { amax = ax; max = x[j]; }
        }
        if (!amax) {
            memset(&y[b], 0, sizeof(y[b]));
            continue;
        }
        const float iscale = -127.f / max;
        for (int j = 0; j < QK_K; j++) {
            int v = (int)lrintf(iscale * x[j]);
            y[b].qs[j] = (int8_t)(v > 127 ? 127 : v < -128 ? -128 : v);
        }
        for (int j = 0; j < QK_K / 16; j++) {
            int sum = 0;
            for (int i = 0; i < 16; i++) sum += y[b].qs[j * 16 + i];
            y[b].bsums[j] = (int16_t)sum;
        }
        y[b].d = 1 / iscale;
    }
}

/* ---- dequantization, for the reference path ---- */

static void dequant_q2_K(const cpu_block_q2_K *x, float *y, uint32_t k) {
    for (uint32_t i = 0; i < k / QK_K; i++) {
        const float d = fp16(x[i].d), dmin = fp16(x[i].dmin);
        const uint8_t *q = x[i].qs;
        int is = 0;
        for (int n = 0; n < QK_K; n += 128) {
            for (int shift = 0; shift < 8; shift += 2) {
                uint8_t sc = x[i].scales[is++];
                float dl = d * (sc & 0xF), ml = dmin * (sc >> 4);
                for (int l = 0; l < 16; l++) *y++ = dl * ((q[l] >> shift) & 3) - ml;
                sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = dmin * (sc >> 4);
                for (int l = 0; l < 16; l++) *y++ = dl * ((q[l + 16] >> shift) & 3) - ml;
            }
            q += 32;
        }
    }
}

static void dequant_iq2_xxs(const cpu_block_iq2_xxs *x, float *y, uint32_t k) {
    for (uint32_t i = 0; i < k / QK_K; i++) {
        const float d = fp16(x[i].d);
        for (int ib = 0; ib < QK_K / 32; ib++) {
            uint32_t aux[2];
            memcpy(aux, x[i].qs + 4 * ib, 8);
            const uint8_t *idx = (const uint8_t *)aux;
            const float db = d * (0.5f + (float)(aux[1] >> 28)) * 0.25f;
            for (int l = 0; l < 4; l++) {
                const uint8_t *grid = (const uint8_t *)(iq2xxs_grid + idx[l]);
                const int8_t *sg = keven_signs[(aux[1] >> (7 * l)) & 127];
                for (int j = 0; j < 8; j++) *y++ = db * grid[j] * sg[j];
            }
        }
    }
}

static void dequant_mxfp4(const uint8_t *x, float *y, uint32_t k) {
    for (uint32_t i = 0; i < k / QK_MXFP4; i++, x += MXFP4_BLOCK, y += QK_MXFP4) {
        const float d = e8m0_half(x[0]);
        for (int j = 0; j < 16; j++) {
            y[j] = d * kvalues_mxfp4[x[1 + j] & 15];
            y[j + 16] = d * kvalues_mxfp4[x[1 + j] >> 4];
        }
    }
}

static void dequant_row(uint32_t type, const void *row, float *y, uint32_t k) {
    if (type == DS4_CPU_EXPERT_Q2_K) dequant_q2_K(row, y, k);
    else if (type == DS4_CPU_EXPERT_MXFP4) dequant_mxfp4(row, y, k);
    else dequant_iq2_xxs(row, y, k);
}

/* ---- dot products against Q8_K activations ---- */

#if DS4_CPU_EXPERTS_SIMD
static inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

static const uint8_t k_shuffle_q3k[128] = {
     0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
     4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
     8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
    12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
};

static float dot_q2_K(const cpu_block_q2_K *x, const cpu_block_q8_K *y, uint32_t nb) {
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m256i *shuf = (const __m256i *)k_shuffle_q3k;
    __m256 acc = _mm256_setzero_ps();
    for (uint32_t i = 0; i < nb; ++i) {
        const float d = y[i].d * fp16(x[i].d);
        const float dmin = -y[i].d * fp16(x[i].dmin);
        const uint8_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        const __m128i mins_and_scales = _mm_loadu_si128((const __m128i *)x[i].scales);
        const __m128i scales8 = _mm_and_si128(mins_and_scales, m4);
        const __m128i mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        const __m256i mins = _mm256_cvtepi8_epi16(mins8);
        const __m256i prod = _mm256_madd_epi16(mins, _mm256_loadu_si256((const __m256i *)y[i].bsums));
        acc = _mm256_fmadd_ps(_mm256_set1_ps(dmin), _mm256_cvtepi32_ps(prod), acc);
        const __m256i all_scales = _mm256_cvtepi8_epi16(scales8);
        const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
        const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
        const __m256i scales[2] = {_mm256_set_m128i(l_scales, l_scales), _mm256_set_m128i(h_scales, h_scales)};
        __m256i sumi = _mm256_setzero_si256();
        for (int j = 0; j < QK_K / 128; ++j) {
            const __m256i q2bits = _mm256_loadu_si256((const __m256i *)q2); q2 += 32;
            const __m256i q8_0 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q2_0 = _mm256_and_si256(q2bits, m3);
            const __m256i q2_1 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), m3);
            const __m256i q2_2 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), m3);
            const __m256i q2_3 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), m3);
            __m256i p0 = _mm256_maddubs_epi16(q2_0, q8_0);
            __m256i p1 = _mm256_maddubs_epi16(q2_1, q8_1);
            __m256i p2 = _mm256_maddubs_epi16(q2_2, q8_2);
            __m256i p3 = _mm256_maddubs_epi16(q2_3, q8_3);
            p0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], _mm256_loadu_si256(shuf + 0)), p0);
            p1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], _mm256_loadu_si256(shuf + 1)), p1);
            p2 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], _mm256_loadu_si256(shuf + 2)), p2);
            p3 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], _mm256_loadu_si256(shuf + 3)), p3);
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(_mm256_add_epi32(p0, p1), _mm256_add_epi32(p2, p3)));
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }
    return hsum_float_8(acc);
}

static float dot_iq2_xxs(const cpu_block_iq2_xxs *x, const cpu_block_q8_K *y, uint32_t nb) {
    const uint64_t *signs64 = (const uint64_t *)keven_signs;
    uint32_t aux32[4];
    const uint8_t *aux8 = (const uint8_t *)aux32;
    __m256 accumf = _mm256_setzero_ps();
    for (uint32_t i = 0; i < nb; ++i) {
        const float d = fp16(x[i].d) * y[i].d;
        const uint16_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        __m256i sumi1 = _mm256_setzero_si256(), sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            memcpy(aux32, q2, 4 * sizeof(uint32_t)); q2 += 8;
            const __m256i q2_1 = _mm256_set_epi64x(iq2xxs_grid[aux8[3]], iq2xxs_grid[aux8[2]],
                                                   iq2xxs_grid[aux8[1]], iq2xxs_grid[aux8[0]]);
            const __m256i q2_2 = _mm256_set_epi64x(iq2xxs_grid[aux8[11]], iq2xxs_grid[aux8[10]],
                                                   iq2xxs_grid[aux8[9]], iq2xxs_grid[aux8[8]]);
            const __m256i s2_1 = _mm256_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                                   signs64[(aux32[1] >> 7) & 127], signs64[(aux32[1] >> 0) & 127]);
            const __m256i s2_2 = _mm256_set_epi64x(signs64[(aux32[3] >> 21) & 127], signs64[(aux32[3] >> 14) & 127],
                                                   signs64[(aux32[3] >> 7) & 127], signs64[(aux32[3] >> 0) & 127]);
            const __m256i q8s_1 = _mm256_sign_epi8(q8_1, s2_1);
            const __m256i q8s_2 = _mm256_sign_epi8(q8_2, s2_2);
            const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
            const uint16_t ls1 = aux32[1] >> 28, ls2 = aux32[3] >> 28;
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot1, _mm256_set1_epi16(2 * ls1 + 1)));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot2, _mm256_set1_epi16(2 * ls2 + 1)));
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    return 0.125f * hsum_float_8(accumf);
}

/* After llama.cpp's ggml_vec_dot_mxfp4_q8_0 (AVX2), against Q8_K: eight
 * MXFP4 blocks share each Q8_K block's scale. */
static float dot_mxfp4(const uint8_t *x, const cpu_block_q8_K *y, uint32_t nb) {
    const __m128i lut = _mm_loadu_si128((const __m128i *)kvalues_mxfp4);
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    for (uint32_t i = 0; i < nb; ++i) {
        const int8_t *q8 = y[i].qs;
        for (int b = 0; b < QK_K / QK_MXFP4; b++, x += MXFP4_BLOCK, q8 += 32) {
            const __m128i q4 = _mm_loadu_si128((const __m128i *)(x + 1));
            const __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(q4, m4));
            const __m128i hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(q4, 4), m4));
            const __m256i w = _mm256_set_m128i(hi, lo);
            const __m256i a = _mm256_loadu_si256((const __m256i *)q8);
            const __m256i p = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(a, w));
            const __m256i s = _mm256_madd_epi16(p, ones);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(y[i].d * e8m0_half(x[0])), _mm256_cvtepi32_ps(s), acc);
        }
    }
    return hsum_float_8(acc);
}

static inline float dot_row(uint32_t type, const void *row, const cpu_block_q8_K *y, uint32_t nb) {
    return type == DS4_CPU_EXPERT_Q2_K ? dot_q2_K(row, y, nb)
         : type == DS4_CPU_EXPERT_MXFP4 ? dot_mxfp4(row, y, nb) : dot_iq2_xxs(row, y, nb);
}
#endif

static inline float silu(float x) { return x / (1.f + expf(-x)); }

int ds4_cpu_experts_supported(const ds4_cpu_expert_shape *s) {
    if (!DS4_CPU_EXPERTS_SIMD || !s || !s->K || !s->M || s->K % QK_K) return 0;
    const int g = s->gate_type == DS4_CPU_EXPERT_Q2_K || s->gate_type == DS4_CPU_EXPERT_IQ2_XXS;
    const int d = s->down_type == DS4_CPU_EXPERT_Q2_K || s->down_type == DS4_CPU_EXPERT_IQ2_XXS ||
                  s->down_type == DS4_CPU_EXPERT_MXFP4;
    return g && d && s->M <= 16384;
}

void ds4_cpu_expert_ref(const ds4_cpu_expert_shape *s, const ds4_cpu_expert_job *job, int quantize) {
    pthread_once(&tables_once, tables_init);
    const uint32_t K = s->K, M = s->M, Mp = (M + QK_K - 1) / QK_K * QK_K;
    const uint64_t grb = ds4_cpu_expert_row_bytes(s->gate_type, K);
    const uint64_t drb = ds4_cpu_expert_row_bytes(s->down_type, Mp);
    float *x = malloc(K * sizeof(float)), *w = malloc((K > Mp ? K : Mp) * sizeof(float));
    float *w2 = malloc(K * sizeof(float)), *mid = calloc(Mp, sizeof(float));
    cpu_block_q8_K *q = malloc((K > Mp ? K : Mp) / QK_K * sizeof(cpu_block_q8_K));
    memcpy(x, job->x, K * sizeof(float));
    if (quantize) {
        quantize_q8_K(x, q, K);
        for (uint32_t i = 0; i < K; i++) x[i] = q[i / QK_K].d * q[i / QK_K].qs[i % QK_K];
    }
    for (uint32_t r = 0; r < M; r++) {
        dequant_row(s->gate_type, (const char *)job->gate + r * grb, w, K);
        dequant_row(s->gate_type, (const char *)job->up + r * grb, w2, K);
        double a = 0, b = 0;
        for (uint32_t i = 0; i < K; i++) { a += (double)w[i] * x[i]; b += (double)w2[i] * x[i]; }
        mid[r] = silu((float)a) * (float)b;
    }
    if (quantize) {
        quantize_q8_K(mid, q, Mp);
        for (uint32_t i = 0; i < Mp; i++) mid[i] = q[i / QK_K].d * q[i / QK_K].qs[i % QK_K];
    }
    for (uint32_t r = 0; r < K; r++) {
        dequant_row(s->down_type, (const char *)job->down + r * drb, w, Mp);
        double a = 0;
        for (uint32_t i = 0; i < M; i++) a += (double)w[i] * mid[i];
        job->out[r] = (float)a;
    }
    free(x); free(w); free(w2); free(mid); free(q);
}

/* ---- the pool ---- */

#define ROWS_A 64u   /* gate/up rows per task */
#define ROWS_B 256u  /* down rows per task */

typedef struct {
    float *mid;                 /* Mp floats */
    cpu_block_q8_K *midq;       /* Mp / QK_K blocks */
    const cpu_block_q8_K *xq;
    _Atomic uint32_t a_done;
    _Atomic uint32_t ready;     /* holds the batch generation once midq is valid */
} job_state;

static struct {
    pthread_mutex_t start_lock;
    int started;
    uint32_t n_workers;
    pthread_t threads[256];
    /* doorbell */
    _Atomic uint32_t gen;
    _Atomic uint32_t sleepers;
    /* (generation << 32) | next task; index UINT32_MAX means closed */
    _Atomic uint64_t ticket;
    _Atomic uint32_t b_done;
    /* the batch, written by the host only while the ticket is closed */
    ds4_cpu_expert_shape shape;
    const ds4_cpu_expert_job *jobs;
    uint32_t n_jobs, a_per_job, b_per_job, n_a, n_total, Mp;
    job_state *state;
    uint32_t cap_jobs, cap_x;
    cpu_block_q8_K *xq;          /* one Q8_K row per distinct x */
    float *mid_store;
    cpu_block_q8_K *midq_store;
    uint32_t mid_store_mp;
} P = {.start_lock = PTHREAD_MUTEX_INITIALIZER, .ticket = UINT32_MAX};

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void futex_wait(_Atomic uint32_t *addr, uint32_t val) {
    syscall(SYS_futex, (uint32_t *)addr, FUTEX_WAIT_PRIVATE, val, NULL, NULL, 0);
}

static void futex_wake_all(_Atomic uint32_t *addr) {
    syscall(SYS_futex, (uint32_t *)addr, FUTEX_WAKE_PRIVATE, 0x7fffffff, NULL, NULL, 0);
}

static inline void cpu_relax(void) {
#if DS4_CPU_EXPERTS_SIMD
    _mm_pause();
#endif
}

/* Claims one task of generation g, or returns UINT32_MAX. */
static uint32_t claim(uint32_t g) {
    uint64_t t = atomic_load_explicit(&P.ticket, memory_order_acquire);
    for (;;) {
        const uint32_t idx = (uint32_t)t;
        if ((uint32_t)(t >> 32) != g || idx == UINT32_MAX || idx >= P.n_total) return UINT32_MAX;
        if (atomic_compare_exchange_weak_explicit(&P.ticket, &t, t + 1,
                memory_order_acq_rel, memory_order_acquire)) return idx;
    }
}

static void run_task(uint32_t g, uint32_t idx) {
#if DS4_CPU_EXPERTS_SIMD
    const ds4_cpu_expert_shape *s = &P.shape;
    if (idx < P.n_a) {
        const uint32_t j = idx / P.a_per_job, r0 = idx % P.a_per_job * ROWS_A;
        const uint32_t r1 = r0 + ROWS_A < s->M ? r0 + ROWS_A : s->M;
        const ds4_cpu_expert_job *job = &P.jobs[j];
        job_state *st = &P.state[j];
        const uint64_t rb = ds4_cpu_expert_row_bytes(s->gate_type, s->K);
        const uint32_t nb = s->K / QK_K;
        for (uint32_t r = r0; r < r1; r++) {
            const float a = dot_row(s->gate_type, (const char *)job->gate + r * rb, st->xq, nb);
            const float b = dot_row(s->gate_type, (const char *)job->up + r * rb, st->xq, nb);
            st->mid[r] = silu(a) * b;
        }
        if (atomic_fetch_add_explicit(&st->a_done, 1, memory_order_acq_rel) + 1 == P.a_per_job) {
            for (uint32_t r = s->M; r < P.Mp; r++) st->mid[r] = 0;
            quantize_q8_K(st->mid, st->midq, P.Mp);
            atomic_store_explicit(&st->ready, g, memory_order_release);
        }
    } else {
        const uint32_t b = idx - P.n_a, j = b / P.b_per_job, r0 = b % P.b_per_job * ROWS_B;
        const uint32_t r1 = r0 + ROWS_B < s->K ? r0 + ROWS_B : s->K;
        const ds4_cpu_expert_job *job = &P.jobs[j];
        job_state *st = &P.state[j];
        /* Every gate/up task sorts before this one and was claimed already, so
         * this waits only on threads that are running. */
        while (atomic_load_explicit(&st->ready, memory_order_acquire) != g) cpu_relax();
        const uint64_t rb = ds4_cpu_expert_row_bytes(s->down_type, P.Mp);
        const uint32_t nb = P.Mp / QK_K;
        for (uint32_t r = r0; r < r1; r++)
            job->out[r] = dot_row(s->down_type, (const char *)job->down + r * rb, st->midq, nb);
        atomic_fetch_add_explicit(&P.b_done, 1, memory_order_acq_rel);
    }
#else
    (void)g; (void)idx;
#endif
}

static void drain(uint32_t g) {
    for (uint32_t idx; (idx = claim(g)) != UINT32_MAX;) run_task(g, idx);
}

static void *worker_main(void *arg) {
    (void)arg;
    uint32_t seen = atomic_load(&P.gen);
    for (;;) {
        /* Spin about 3 ms between batches - decode posts one per layer, far
         * closer together than that - then sleep on the doorbell. */
        double t0 = 0;
        uint32_t g;
        for (uint32_t spins = 0; (g = atomic_load_explicit(&P.gen, memory_order_acquire)) == seen; spins++) {
            cpu_relax();
            if ((spins & 1023) == 1023) {
                const double t = now_sec();
                if (!t0) t0 = t;
                else if (t - t0 > 3e-3) {
                    atomic_fetch_add(&P.sleepers, 1);
                    if (atomic_load(&P.gen) == seen) futex_wait(&P.gen, seen);
                    atomic_fetch_sub(&P.sleepers, 1);
                    t0 = 0;
                }
            }
        }
        seen = g;
        drain(g);
    }
    return NULL;
}

/* First logical CPU of each physical core, in core order. */
static uint32_t physical_cpus(int *out, uint32_t cap) {
    uint32_t n = 0;
    cpu_set_t seen_cores;
    CPU_ZERO(&seen_cores);
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    for (long c = 0; c < ncpu && n < cap; c++) {
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%ld/topology/thread_siblings_list", c);
        FILE *f = fopen(path, "r");
        int first = (int)c;
        if (f) {
            if (fscanf(f, "%d", &first) != 1) first = (int)c;
            fclose(f);
        }
        if (first != c || CPU_ISSET(first, &seen_cores)) continue;
        CPU_SET(first, &seen_cores);
        out[n++] = first;
    }
    return n;
}

static void pin_self(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

uint32_t ds4_cpu_experts_start(void) {
    pthread_once(&tables_once, tables_init);
    pthread_mutex_lock(&P.start_lock);
    if (!P.started) {
        int cpus[256];
        uint32_t n = physical_cpus(cpus, 256);
        const char *env = getenv("DS4_CPU_HYBRID_THREADS");
        if (env && env[0] && (uint32_t)atoi(env) >= 1 && (uint32_t)atoi(env) <= n) n = (uint32_t)atoi(env);
        const char *pin = getenv("DS4_CPU_HYBRID_PIN");
        const int do_pin = !pin || strcmp(pin, "0") != 0;
        /* The host thread takes the first core; workers get the rest. */
        if (do_pin && n) pin_self(cpus[0]);
        P.n_workers = 0;
        for (uint32_t i = 1; i < n; i++) {
            if (pthread_create(&P.threads[P.n_workers], NULL, worker_main, NULL) != 0) break;
            if (do_pin) {
                cpu_set_t set;
                CPU_ZERO(&set);
                CPU_SET(cpus[i], &set);
                (void)pthread_setaffinity_np(P.threads[P.n_workers], sizeof(set), &set);
            }
            P.n_workers++;
        }
        P.started = 1;
        fprintf(stderr, "ds4: CPU expert pool: %u threads on physical cores%s\n",
                P.n_workers + 1, do_pin ? ", pinned" : "");
    }
    pthread_mutex_unlock(&P.start_lock);
    return P.n_workers + 1;
}

static int ensure_capacity(uint32_t n_jobs, uint32_t n_x, uint32_t K, uint32_t Mp) {
    if (n_jobs > P.cap_jobs || Mp != P.mid_store_mp) {
        const uint32_t cap = n_jobs > P.cap_jobs ? n_jobs : P.cap_jobs;
        free(P.state); free(P.mid_store); free(P.midq_store);
        P.state = calloc(cap, sizeof(job_state));
        P.mid_store = aligned_alloc(64, (size_t)cap * Mp * sizeof(float));
        P.midq_store = aligned_alloc(64, (size_t)cap * (Mp / QK_K) * sizeof(cpu_block_q8_K));
        if (!P.state || !P.mid_store || !P.midq_store) { P.cap_jobs = 0; return 0; }
        P.cap_jobs = cap;
        P.mid_store_mp = Mp;
    }
    if (n_x * (K / QK_K) > P.cap_x) {
        free(P.xq);
        P.cap_x = n_x * (K / QK_K);
        P.xq = aligned_alloc(64, (size_t)P.cap_x * sizeof(cpu_block_q8_K));
        if (!P.xq) { P.cap_x = 0; return 0; }
    }
    return 1;
}

int ds4_cpu_experts_submit(const ds4_cpu_expert_shape *s, const ds4_cpu_expert_job *jobs, uint32_t n) {
    if (!n) return 1;
    if (!ds4_cpu_experts_supported(s)) return 0;
    if (!P.started) ds4_cpu_experts_start();
    const uint32_t Mp = (s->M + QK_K - 1) / QK_K * QK_K, nbx = s->K / QK_K;
    if (!ensure_capacity(n, n, s->K, Mp)) return 0;
    const uint32_t g = atomic_load(&P.gen) + 1;
    /* Distinct activations quantize once: decode rows repeat across slots. */
    uint32_t n_x = 0;
    for (uint32_t j = 0; j < n; j++) {
        uint32_t k = 0;
        while (k < j && jobs[k].x != jobs[j].x) k++;
        job_state *st = &P.state[j];
        if (k < j) st->xq = P.state[k].xq;
        else {
            cpu_block_q8_K *dst = P.xq + (size_t)n_x++ * nbx;
            quantize_q8_K(jobs[j].x, dst, s->K);
            st->xq = dst;
        }
        st->mid = P.mid_store + (size_t)j * Mp;
        st->midq = P.midq_store + (size_t)j * (Mp / QK_K);
        atomic_store_explicit(&st->a_done, 0, memory_order_relaxed);
        atomic_store_explicit(&st->ready, 0, memory_order_relaxed);
    }
    P.shape = *s;
    P.jobs = jobs;
    P.n_jobs = n;
    P.Mp = Mp;
    P.a_per_job = (s->M + ROWS_A - 1) / ROWS_A;
    P.b_per_job = (s->K + ROWS_B - 1) / ROWS_B;
    P.n_a = n * P.a_per_job;
    P.n_total = P.n_a + n * P.b_per_job;
    atomic_store_explicit(&P.b_done, 0, memory_order_relaxed);
    atomic_store_explicit(&P.ticket, (uint64_t)g << 32, memory_order_release);
    atomic_store_explicit(&P.gen, g, memory_order_release);
    if (atomic_load(&P.sleepers)) futex_wake_all(&P.gen);
    return 1;
}

double ds4_cpu_experts_wait(void) {
    const uint32_t g = atomic_load(&P.gen);
    if ((uint32_t)atomic_load(&P.ticket) == UINT32_MAX) return 0;
    drain(g);
    const uint32_t n_b = P.n_total - P.n_a;
    double t0 = 0, waited = 0;
    if (atomic_load_explicit(&P.b_done, memory_order_acquire) != n_b) {
        t0 = now_sec();
        while (atomic_load_explicit(&P.b_done, memory_order_acquire) != n_b) cpu_relax();
        waited = now_sec() - t0;
    }
    /* Close the batch before anything in it can change. */
    atomic_store_explicit(&P.ticket, ((uint64_t)g << 32) | UINT32_MAX, memory_order_release);
    return waited;
}
