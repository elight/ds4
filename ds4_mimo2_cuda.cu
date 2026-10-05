/* CUDA kernels for MiMo V2.6 Flash (see ds4_mimo2.h).
 *
 * Plain warp-per-row kernels that dequantize GGUF blocks in registers; they
 * are written for a single RTX 3090 (sm_86) and favour being obviously right
 * over being the last word in speed. Expert weights arrive in VRAM slots from
 * the host-side tier manager; nothing here knows where they came from. */
#include "ds4_mimo2.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

static cudaStream_t g_s0, g_s1, g_s2;   /* compute, host->device, device->host */
static cudaEvent_t g_copy_ev, g_d2h_ev;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "mimo2 cuda: %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return -1; } } while (0)

extern "C" int m2g_init(void) {
    CK(cudaSetDevice(0));
    CK(cudaStreamCreateWithFlags(&g_s0, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&g_s1, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&g_s2, cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&g_d2h_ev, cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&g_copy_ev, cudaEventDisableTiming));
    return 0;
}

extern "C" void m2g_free_all(void) { cudaDeviceReset(); }

extern "C" void *m2g_alloc(size_t bytes) {
    void *p = NULL;
    if (cudaMalloc(&p, bytes) != cudaSuccess) return NULL;
    return p;
}

extern "C" int m2g_mem_info(size_t *free_bytes, size_t *total_bytes) {
    CK(cudaMemGetInfo(free_bytes, total_bytes));
    return 0;
}

extern "C" void *m2g_host_alloc(size_t bytes) {
    void *p = NULL;
    if (cudaHostAlloc(&p, bytes, cudaHostAllocPortable) != cudaSuccess) return NULL;
    return p;
}

extern "C" void m2g_host_free(void *p) { if (p) cudaFreeHost(p); }

extern "C" int m2g_upload(void *dst, const void *src, size_t bytes) {
    CK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
    return 0;
}

extern "C" int m2g_upload_async(void *dst, const void *src, size_t bytes) {
    CK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, g_s0));
    return 0;
}

extern "C" int m2g_copy_async(void *dst, const void *src, size_t bytes) {
    CK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, g_s1));
    return 0;
}

extern "C" int m2g_d2h_async(void *dst, const void *src, size_t bytes) {
    CK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, g_s2));
    return 0;
}

extern "C" int m2g_d2h_order(void) {
    CK(cudaEventRecord(g_d2h_ev, g_s2));
    CK(cudaStreamWaitEvent(g_s1, g_d2h_ev, 0));
    return 0;
}

extern "C" int m2g_copy_fence(void) {
    CK(cudaEventRecord(g_copy_ev, g_s1));
    CK(cudaStreamWaitEvent(g_s0, g_copy_ev, 0));
    return 0;
}

extern "C" int m2g_download(void *dst, const void *src, size_t bytes) {
    CK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, g_s0));
    CK(cudaStreamSynchronize(g_s0));
    return 0;
}

extern "C" int m2g_copy_rows(float *dst, int ldd, const float *src, int lds, int n, int rows) {
    CK(cudaMemcpy2DAsync(dst, (size_t)ldd * sizeof(float), src, (size_t)lds * sizeof(float),
                         (size_t)n * sizeof(float), (size_t)rows, cudaMemcpyDeviceToDevice, g_s0));
    return 0;
}

extern "C" int m2g_sync(void) {
    CK(cudaStreamSynchronize(g_s0));
    CK(cudaStreamSynchronize(g_s1));
    CK(cudaStreamSynchronize(g_s2));
    return 0;
}

/* ---- dequantization of one unit (16 or 32 consecutive weights) ---------- */

__device__ __forceinline__ float m2_e8m0_half(uint8_t e) {
    unsigned bits = e < 2 ? (0x00200000u << e) : ((unsigned)(e - 1) << 23);
    return __uint_as_float(bits);
}

__constant__ float c_kv_fp4[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };

template <int TYPE> struct m2_unit;

template <> struct m2_unit<M2_T_F32> {
    static constexpr int N = 32;
    static __device__ __forceinline__ int units(int cols) { return cols / 32; }
    static __device__ __forceinline__ int deq(const uint8_t *row, int u, float *w) {
        const float4 *p = (const float4 *)(row + (size_t)u * 128);
#pragma unroll
        for (int i = 0; i < 8; i++) { float4 v = p[i]; w[4*i] = v.x; w[4*i+1] = v.y; w[4*i+2] = v.z; w[4*i+3] = v.w; }
        return u * 32;
    }
};

template <> struct m2_unit<M2_T_Q8_0> {
    static constexpr int N = 32;
    static __device__ __forceinline__ int units(int cols) { return cols / 32; }
    static __device__ __forceinline__ int deq(const uint8_t *row, int u, float *w) {
        const uint8_t *b = row + (size_t)u * 34;
        const float d = __half2float(*(const __half *)b);
        const int8_t *q = (const int8_t *)(b + 2);
#pragma unroll
        for (int i = 0; i < 32; i++) w[i] = d * (float)q[i];
        return u * 32;
    }
};

template <> struct m2_unit<M2_T_MXFP4> {
    static constexpr int N = 32;
    static __device__ __forceinline__ int units(int cols) { return cols / 32; }
    static __device__ __forceinline__ int deq(const uint8_t *row, int u, float *w) {
        const uint8_t *b = row + (size_t)u * 17;
        const float d = m2_e8m0_half(b[0]);
#pragma unroll
        for (int j = 0; j < 16; j++) {
            const uint8_t q = b[1 + j];
            w[j]      = c_kv_fp4[q & 15] * d;
            w[j + 16] = c_kv_fp4[q >> 4] * d;
        }
        return u * 32;
    }
};

template <> struct m2_unit<M2_T_Q2_K> {
    static constexpr int N = 16;
    static __device__ __forceinline__ int units(int cols) { return cols / 16; }
    static __device__ __forceinline__ int deq(const uint8_t *row, int u, float *w) {
        const int b = u >> 4, s = u & 15;
        const uint8_t *blk = row + (size_t)b * 84;
        const float d    = __half2float(*(const __half *)(blk + 80));
        const float dmin = __half2float(*(const __half *)(blk + 82));
        const int half = s >> 3, j = (s & 7) >> 1, hi = s & 1;
        const uint8_t sc = blk[s];
        const float dl = d * (float)(sc & 15), ml = dmin * (float)(sc >> 4);
        const uint8_t *q = blk + 16 + half * 32 + hi * 16;
        const int shift = 2 * j;
#pragma unroll
        for (int l = 0; l < 16; l++) w[l] = dl * (float)((q[l] >> shift) & 3) - ml;
        return b * 256 + half * 128 + j * 32 + hi * 16;
    }
};

template <> struct m2_unit<M2_T_Q6_K> {
    static constexpr int N = 16;
    static __device__ __forceinline__ int units(int cols) { return cols / 16; }
    static __device__ __forceinline__ int deq(const uint8_t *row, int u, float *w) {
        const int b = u >> 4, s = u & 15;
        const uint8_t *blk = row + (size_t)b * 210;
        const uint8_t *ql = blk, *qh = blk + 128;
        const int8_t *sc = (const int8_t *)(blk + 192);
        const float d = __half2float(*(const __half *)(blk + 208)) * (float)sc[s];
        const int c = s >> 3, q4 = (s & 7) >> 1, is = s & 1;
        const uint8_t *qlp = ql + 64 * c + (q4 & 1) * 32 + 16 * is;
        const uint8_t *qhp = qh + 32 * c + 16 * is;
        const int nshift = (q4 >= 2) ? 4 : 0, hshift = 2 * q4;
#pragma unroll
        for (int l = 0; l < 16; l++) {
            const int q = (((qlp[l] >> nshift) & 15) | (((qhp[l] >> hshift) & 3) << 4)) - 32;
            w[l] = d * (float)q;
        }
        return b * 256 + 128 * c + 32 * q4 + 16 * is;
    }
};

static size_t m2_row_bytes(int type, int cols) {
    switch (type) {
    case M2_T_F32:   return (size_t)cols * 4;
    case M2_T_Q8_0:  return (size_t)cols / 32 * 34;
    case M2_T_MXFP4: return (size_t)cols / 32 * 17;
    case M2_T_Q2_K:  return (size_t)cols / 256 * 84;
    case M2_T_Q6_K:  return (size_t)cols / 256 * 210;
    }
    return 0;
}

template <int N>
__device__ __forceinline__ float m2_dot(const float *w, const float *x) {
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < N; i += 4) {
        const float4 v = *(const float4 *)(x + i);
        s += w[i] * v.x + w[i+1] * v.y + w[i+2] * v.z + w[i+3] * v.w;
    }
    return s;
}

__device__ __forceinline__ float m2_warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float m2_warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

#define M2_WPB 8   /* warps (rows) per block */

/* ---- dense matmul -------------------------------------------------------- */

template <int TYPE, int NT>
__global__ void k_mm(const uint8_t *W, size_t rb, int rows, int cols,
                     const float *X, int ldx, float *Y, int ldy, int ntok, int acc) {
    typedef m2_unit<TYPE> U;
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.x * M2_WPB + (threadIdx.x >> 5);
    if (r >= rows) return;
    const uint8_t *row = W + (size_t)r * rb;
    const int units = U::units(cols);
    for (int t0 = 0; t0 < ntok; t0 += NT) {
        const int nt = min(NT, ntok - t0);
        float a[NT];
#pragma unroll
        for (int t = 0; t < NT; t++) a[t] = 0.f;
        for (int u = lane; u < units; u += 32) {
            float w[U::N];
            const int start = U::deq(row, u, w);
#pragma unroll
            for (int t = 0; t < NT; t++)
                if (t < nt) a[t] += m2_dot<U::N>(w, X + (size_t)(t0 + t) * ldx + start);
        }
#pragma unroll
        for (int t = 0; t < NT; t++) {
            const float v = m2_warp_sum(a[t]);
            if (lane == 0 && t < nt) {
                float *o = Y + (size_t)(t0 + t) * ldy + r;
                *o = acc ? *o + v : v;
            }
        }
    }
}

template <int TYPE>
static void m2_mm_launch(const void *W, int rows, int cols, const float *x, int ldx,
                         float *y, int ldy, int ntok, int acc) {
    const size_t rb = m2_row_bytes(TYPE, cols);
    dim3 grid((rows + M2_WPB - 1) / M2_WPB), block(32 * M2_WPB);
    const uint8_t *w = (const uint8_t *)W;
    if (ntok >= 8)      k_mm<TYPE, 8><<<grid, block, 0, g_s0>>>(w, rb, rows, cols, x, ldx, y, ldy, ntok, acc);
    else if (ntok >= 4) k_mm<TYPE, 4><<<grid, block, 0, g_s0>>>(w, rb, rows, cols, x, ldx, y, ldy, ntok, acc);
    else if (ntok >= 2) k_mm<TYPE, 2><<<grid, block, 0, g_s0>>>(w, rb, rows, cols, x, ldx, y, ldy, ntok, acc);
    else                k_mm<TYPE, 1><<<grid, block, 0, g_s0>>>(w, rb, rows, cols, x, ldx, y, ldy, ntok, acc);
}

extern "C" int m2g_matmul(int type, const void *W, int rows, int cols,
                          const float *x, int ldx, float *y, int ldy, int ntok, int accumulate) {
    switch (type) {
    case M2_T_F32:   m2_mm_launch<M2_T_F32>(W, rows, cols, x, ldx, y, ldy, ntok, accumulate); break;
    case M2_T_Q8_0:  m2_mm_launch<M2_T_Q8_0>(W, rows, cols, x, ldx, y, ldy, ntok, accumulate); break;
    case M2_T_Q2_K:  m2_mm_launch<M2_T_Q2_K>(W, rows, cols, x, ldx, y, ldy, ntok, accumulate); break;
    case M2_T_Q6_K:  m2_mm_launch<M2_T_Q6_K>(W, rows, cols, x, ldx, y, ldy, ntok, accumulate); break;
    case M2_T_MXFP4: m2_mm_launch<M2_T_MXFP4>(W, rows, cols, x, ldx, y, ldy, ntok, accumulate); break;
    default: fprintf(stderr, "mimo2 cuda: matmul type %d unsupported\n", type); return -1;
    }
    CK(cudaGetLastError());
    return 0;
}

/* ---- elementwise --------------------------------------------------------- */

__global__ void k_rmsnorm(const float *x, const float *w, float *y, int n, float eps) {
    const float *xr = x + (size_t)blockIdx.x * n;
    float *yr = y + (size_t)blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    __shared__ float red[32];
    s = m2_warp_sum(s);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
        s = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
        s = m2_warp_sum(s);
        if (threadIdx.x == 0) red[0] = s;
    }
    __syncthreads();
    const float r = rsqrtf(red[0] / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * r * w[i];
}

extern "C" int m2g_rmsnorm(const float *x, const float *w, float *y, int n, int ntok, float eps) {
    k_rmsnorm<<<ntok, 256, 0, g_s0>>>(x, w, y, n, eps);
    CK(cudaGetLastError());
    return 0;
}

__device__ __forceinline__ float m2_silu(float g) { return g / (1.f + expf(-g)); }

__global__ void k_swiglu(const float *g, const float *u, float *h, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) h[i] = m2_silu(g[i]) * u[i];
}

extern "C" int m2g_swiglu2(const float *g, const float *u, float *h, int count) {
    k_swiglu<<<(count + 255) / 256, 256, 0, g_s0>>>(g, u, h, count);
    CK(cudaGetLastError());
    return 0;
}

__global__ void k_add(float *y, const float *x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}

extern "C" int m2g_add(float *y, const float *x, int n) {
    k_add<<<(n + 255) / 256, 256, 0, g_s0>>>(y, x, n);
    CK(cudaGetLastError());
    return 0;
}

__global__ void k_scale(float *y, float s, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] *= s;
}

extern "C" int m2g_scale(float *y, float s, int n) {
    k_scale<<<(n + 255) / 256, 256, 0, g_s0>>>(y, s, n);
    CK(cudaGetLastError());
    return 0;
}

/* ---- attention ----------------------------------------------------------- */

#define M2_HK 192
#define M2_HV 128

/* NEOX RoPE on the first n_rot dims of every q and k head, then store k and v
 * as f16 into the cache ring. One block per token. */
__global__ void k_rope_store(float *qkv, int ld, int n_head, int n_kv, int n_rot,
                             float base, int pos0, __half *kc, __half *vc, int cache_len) {
    const int t = blockIdx.x, p = pos0 + t;
    float *row = qkv + (size_t)t * ld;
    const int half = n_rot / 2;
    const int nh = n_head + n_kv;
    for (int idx = threadIdx.x; idx < nh * half; idx += blockDim.x) {
        const int h = idx / half, i = idx % half;
        float *v = row + h * M2_HK;
        const double theta = (double)p * pow((double)base, -2.0 * i / (double)n_rot);
        double sn, cs;
        sincos(theta, &sn, &cs);
        const float x0 = v[i], x1 = v[i + half];
        v[i]        = (float)(x0 * cs - x1 * sn);
        v[i + half] = (float)(x0 * sn + x1 * cs);
    }
    __syncthreads();
    const int slot = p % cache_len;
    const float *k = row + n_head * M2_HK;
    const float *vv = k + n_kv * M2_HK;
    __half *kd = kc + (size_t)slot * n_kv * M2_HK;
    __half *vd = vc + (size_t)slot * n_kv * M2_HV;
    for (int i = threadIdx.x; i < n_kv * M2_HK; i += blockDim.x) kd[i] = __float2half(k[i]);
    for (int i = threadIdx.x; i < n_kv * M2_HV; i += blockDim.x) vd[i] = __float2half(vv[i]);
}

#define M2_ACH 32          /* keys per chunk */
#define M2_KPAD (M2_HK + 2) /* odd word stride: conflict-free half2 rows */

/* One block per (token, kv head), covering the G query heads that share it.
 * Online softmax over 32-key chunks; a sink starts each head at
 * (max = sink, sum = 1), which is exactly softmax with the sink in the
 * denominator. */
template <int G>
__global__ void __launch_bounds__(256) k_attn(const float *qkv, int ld, int n_head, int n_kv,
                                              int pos0, int window, const __half *kc,
                                              const __half *vc, int cache_len,
                                              const float *sinks, float *out, float scale) {
    __shared__ float qs[G][M2_HK];
    __shared__ __half ks[M2_ACH][M2_KPAD];
    __shared__ __half vs[M2_ACH][M2_HV];
    __shared__ float S[G][M2_ACH];
    __shared__ float m_s[G], l_s[G], a_s[G];

    const int t = blockIdx.x, g = blockIdx.y, tid = threadIdx.x;
    const int p = pos0 + t;
    const float *row = qkv + (size_t)t * ld;
    for (int i = tid; i < G * M2_HK; i += 256) {
        const int hh = i / M2_HK, d = i % M2_HK;
        qs[hh][d] = row[(g * G + hh) * M2_HK + d] * scale;
    }
    if (tid < G) {
        const int h = g * G + tid;
        m_s[tid] = sinks ? sinks[h] : -INFINITY;
        l_s[tid] = sinks ? 1.f : 0.f;
    }
    constexpr int NA = G * M2_HV / 256;
    float acc[NA];
#pragma unroll
    for (int k = 0; k < NA; k++) acc[k] = 0.f;

    const int lo = window > 0 ? max(0, p - window + 1) : 0;
    const int warp = tid >> 5, lane = tid & 31;
    __syncthreads();

    for (int j0 = lo; j0 <= p; j0 += M2_ACH) {
        for (int i = tid; i < M2_ACH * (M2_HK / 2); i += 256) {
            const int jj = i / (M2_HK / 2), d2 = i % (M2_HK / 2);
            const int j = j0 + jj;
            __half2 v = __float2half2_rn(0.f);
            if (j <= p) v = ((const __half2 *)(kc + ((size_t)(j % cache_len) * n_kv + g) * M2_HK))[d2];
            *(__half2 *)&ks[jj][2 * d2] = v;
        }
        for (int i = tid; i < M2_ACH * (M2_HV / 2); i += 256) {
            const int jj = i / (M2_HV / 2), d2 = i % (M2_HV / 2);
            const int j = j0 + jj;
            __half2 v = __float2half2_rn(0.f);
            if (j <= p) v = ((const __half2 *)(vc + ((size_t)(j % cache_len) * n_kv + g) * M2_HV))[d2];
            *(__half2 *)&vs[jj][2 * d2] = v;
        }
        __syncthreads();
        for (int i = tid; i < G * M2_ACH; i += 256) {
            const int hh = i / M2_ACH, jj = i % M2_ACH;
            float s = -INFINITY;
            if (j0 + jj <= p) {
                s = 0.f;
#pragma unroll 8
                for (int d2 = 0; d2 < M2_HK / 2; d2++) {
                    const float2 kf = __half22float2(*(const __half2 *)&ks[jj][2 * d2]);
                    s += qs[hh][2 * d2] * kf.x + qs[hh][2 * d2 + 1] * kf.y;
                }
            }
            S[hh][jj] = s;
        }
        __syncthreads();
        for (int hh = warp; hh < G; hh += 8) {
            const float s = S[hh][lane];
            const float mx = m2_warp_max(s);
            const float m_old = m_s[hh];
            const float m_new = fmaxf(m_old, mx);
            const float pr = (s == -INFINITY) ? 0.f : __expf(s - m_new);
            const float sum = m2_warp_sum(pr);
            S[hh][lane] = pr;
            __syncwarp();
            if (lane == 0) {
                const float a = (m_old == -INFINITY) ? 0.f : __expf(m_old - m_new);
                a_s[hh] = a;
                l_s[hh] = l_s[hh] * a + sum;
                m_s[hh] = m_new;
            }
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < NA; k++) {
            const int idx = tid + k * 256;
            const int hh = idx / M2_HV, d = idx % M2_HV;
            float a = acc[k] * a_s[hh];
#pragma unroll 8
            for (int jj = 0; jj < M2_ACH; jj++) a += S[hh][jj] * __half2float(vs[jj][d]);
            acc[k] = a;
        }
        __syncthreads();
    }
#pragma unroll
    for (int k = 0; k < NA; k++) {
        const int idx = tid + k * 256;
        const int hh = idx / M2_HV, d = idx % M2_HV;
        const float l = l_s[hh];
        out[(size_t)t * n_head * M2_HV + (g * G + hh) * M2_HV + d] = l > 0.f ? acc[k] / l : 0.f;
    }
}

extern "C" int m2g_attention(float *qkv, int ntok, int pos0, int n_head, int n_kv,
                             int n_rot, float rope_base, int window,
                             void *kcache, void *vcache, int cache_len,
                             const float *sinks, float *out) {
    const int ld = n_head * M2_HK + n_kv * M2_HK + n_kv * M2_HV;
    k_rope_store<<<ntok, 256, 0, g_s0>>>(qkv, ld, n_head, n_kv, n_rot, rope_base, pos0,
                                         (__half *)kcache, (__half *)vcache, cache_len);
    CK(cudaGetLastError());
    const float scale = 1.0f / sqrtf((float)M2_HK);
    dim3 grid(ntok, n_kv);
    const int G = n_head / n_kv;
    if (G == 16)
        k_attn<16><<<grid, 256, 0, g_s0>>>(qkv, ld, n_head, n_kv, pos0, window,
                                           (const __half *)kcache, (const __half *)vcache,
                                           cache_len, sinks, out, scale);
    else if (G == 8)
        k_attn<8><<<grid, 256, 0, g_s0>>>(qkv, ld, n_head, n_kv, pos0, window,
                                          (const __half *)kcache, (const __half *)vcache,
                                          cache_len, sinks, out, scale);
    else { fprintf(stderr, "mimo2 cuda: GQA group %d unsupported\n", G); return -1; }
    CK(cudaGetLastError());
    return 0;
}

/* ---- MoE ----------------------------------------------------------------- */

#define M2_MOE_NT 4

/* h[a] = silu(gate . x_t) * (up . x_t) for every assignment a of every job. */
__global__ void k_moe_gate_up(const m2_moe_job *jobs, const int *assign, int topk,
                              const float *X, float *H, int n_embd, int n_ff, size_t gate_bytes) {
    typedef m2_unit<M2_T_Q2_K> U;
    const m2_moe_job job = jobs[blockIdx.y];
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.x * M2_WPB + (threadIdx.x >> 5);
    if (r >= n_ff) return;
    const size_t rb = (size_t)n_embd / 256 * 84;
    const uint8_t *grow = job.w + (size_t)r * rb;
    const uint8_t *urow = job.w + gate_bytes + (size_t)r * rb;
    const int units = U::units(n_embd);
    for (int i0 = 0; i0 < job.n; i0 += M2_MOE_NT) {
        const int nt = min(M2_MOE_NT, job.n - i0);
        int a[M2_MOE_NT];
        const float *x[M2_MOE_NT];
        float ga[M2_MOE_NT], ua[M2_MOE_NT];
#pragma unroll
        for (int t = 0; t < M2_MOE_NT; t++) {
            a[t] = t < nt ? assign[job.off + i0 + t] : 0;
            x[t] = X + (size_t)(a[t] / topk) * n_embd;
            ga[t] = ua[t] = 0.f;
        }
        for (int u = lane; u < units; u += 32) {
            float w[16];
            const int start = U::deq(grow, u, w);
#pragma unroll
            for (int t = 0; t < M2_MOE_NT; t++) if (t < nt) ga[t] += m2_dot<16>(w, x[t] + start);
            U::deq(urow, u, w);
#pragma unroll
            for (int t = 0; t < M2_MOE_NT; t++) if (t < nt) ua[t] += m2_dot<16>(w, x[t] + start);
        }
#pragma unroll
        for (int t = 0; t < M2_MOE_NT; t++) {
            const float gs = m2_warp_sum(ga[t]);
            const float us = m2_warp_sum(ua[t]);
            if (lane == 0 && t < nt) H[(size_t)a[t] * n_ff + r] = m2_silu(gs) * us;
        }
    }
}

/* y[a] = down . h[a]. */
__global__ void k_moe_down(const m2_moe_job *jobs, const int *assign,
                           const float *H, float *Y, int n_embd, int n_ff, size_t down_off) {
    typedef m2_unit<M2_T_MXFP4> U;
    const m2_moe_job job = jobs[blockIdx.y];
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.x * M2_WPB + (threadIdx.x >> 5);
    if (r >= n_embd) return;
    const size_t rb = (size_t)n_ff / 32 * 17;
    const uint8_t *row = job.w + down_off + (size_t)r * rb;
    const int units = U::units(n_ff);
    for (int i0 = 0; i0 < job.n; i0 += M2_MOE_NT) {
        const int nt = min(M2_MOE_NT, job.n - i0);
        int a[M2_MOE_NT];
        float acc[M2_MOE_NT];
#pragma unroll
        for (int t = 0; t < M2_MOE_NT; t++) { a[t] = t < nt ? assign[job.off + i0 + t] : 0; acc[t] = 0.f; }
        for (int u = lane; u < units; u += 32) {
            float w[32];
            const int start = U::deq(row, u, w);
#pragma unroll
            for (int t = 0; t < M2_MOE_NT; t++)
                if (t < nt) acc[t] += m2_dot<32>(w, H + (size_t)a[t] * n_ff + start);
        }
#pragma unroll
        for (int t = 0; t < M2_MOE_NT; t++) {
            const float v = m2_warp_sum(acc[t]);
            if (lane == 0 && t < nt) Y[(size_t)a[t] * n_embd + r] = v;
        }
    }
}

__global__ void k_moe_combine(const float *Y, const float *wts, float *out, int topk, int n_embd) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    float s = 0.f;
    for (int k = 0; k < topk; k++) {   /* weight 0: computed elsewhere (CPU), Y row is stale */
        const float w = wts[t * topk + k];
        if (w != 0.f) s += w * Y[((size_t)t * topk + k) * n_embd + i];
    }
    out[(size_t)t * n_embd + i] += s;
}

extern "C" int m2g_moe(const m2_moe_job *jobs_dev, int njobs, const int *assign_dev,
                       const float *weights_dev, int ntok, int topk,
                       const float *x, float *h_buf, float *y_buf, float *out,
                       int n_embd, int n_ff, size_t gate_bytes, size_t up_bytes) {
    if (njobs > 0) {
        dim3 g1((n_ff + M2_WPB - 1) / M2_WPB, njobs);
        k_moe_gate_up<<<g1, 32 * M2_WPB, 0, g_s0>>>(jobs_dev, assign_dev, topk, x, h_buf,
                                                    n_embd, n_ff, gate_bytes);
        CK(cudaGetLastError());
        dim3 g2((n_embd + M2_WPB - 1) / M2_WPB, njobs);
        k_moe_down<<<g2, 32 * M2_WPB, 0, g_s0>>>(jobs_dev, assign_dev, h_buf, y_buf,
                                                 n_embd, n_ff, gate_bytes + up_bytes);
        CK(cudaGetLastError());
    }
    dim3 g3((n_embd + 255) / 256, ntok);
    k_moe_combine<<<g3, 256, 0, g_s0>>>(y_buf, weights_dev, out, topk, n_embd);
    CK(cudaGetLastError());
    return 0;
}
