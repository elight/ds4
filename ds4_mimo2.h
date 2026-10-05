/* MiMo V2.6 Flash (GGUF arch "mimo2") on one CUDA GPU with a three-tier
 * expert cache: VRAM slots, a pinned RAM arena, and the GGUF on SSD.
 *
 * ds4_mimo2.c owns the GGUF, the tokenizer, the expert tiers and the forward
 * pass; ds4_mimo2_cuda.cu owns the kernels. This header is the boundary. */
#ifndef DS4_MIMO2_H
#define DS4_MIMO2_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* GGUF tensor types used by the MiMo GGUFs. */
enum {
    M2_T_F32   = 0,
    M2_T_Q8_0  = 8,
    M2_T_Q2_K  = 10,
    M2_T_Q6_K  = 14,
    M2_T_MXFP4 = 39,
};

/* One routed expert in a VRAM slot: gate | up | down, back to back. */
typedef struct {
    const uint8_t *w;   /* slot base on the device */
    int n;              /* tokens routed to this expert in the batch */
    int off;            /* first entry of this job in the assignment list */
} m2_moe_job;

int   m2g_init(void);
void  m2g_free_all(void);
void *m2g_alloc(size_t bytes);
int   m2g_mem_info(size_t *free_bytes, size_t *total_bytes);
void *m2g_host_alloc(size_t bytes);     /* pinned, portable */
void  m2g_host_free(void *p);
int   m2g_upload(void *dst, const void *src, size_t bytes);           /* sync */
int   m2g_upload_async(void *dst, const void *src, size_t bytes);     /* compute stream */
int   m2g_copy_async(void *dst, const void *src, size_t bytes);       /* copy stream */
int   m2g_download(void *dst, const void *src, size_t bytes);         /* sync */
int   m2g_d2h_async(void *dst, const void *src, size_t bytes);        /* demotion stream */
int   m2g_d2h_order(void);           /* later host->device copies wait for demotions so far */
int   m2g_copy_fence(void);          /* compute stream waits for copies issued so far */
int   m2g_sync(void);

/* y[t][r] = sum_k W[r][k] x[t][k]; W is rows x cols in GGUF type `type`. */
int m2g_matmul(int type, const void *W, int rows, int cols,
               const float *x, int ldx, float *y, int ldy, int ntok, int accumulate);
int m2g_rmsnorm(const float *x, const float *w, float *y, int n, int ntok, float eps);
int m2g_swiglu2(const float *g, const float *u, float *h, int count); /* h = silu(g)*u */
int m2g_add(float *y, const float *x, int n);               /* y += x, n floats */
int m2g_scale(float *y, float s, int n);

/* Attention for ntok consecutive tokens starting at pos0.
 * qkv: [t][n_head*192 + n_kv*192 + n_kv*128]; RoPE (NEOX, first n_rot dims)
 * is applied to q and k in place; k and v are stored into the f16 cache
 * (ring of `cache_len` positions); out: [t][n_head*128]. */
int m2g_attention(float *qkv, int ntok, int pos0, int n_head, int n_kv,
                  int n_rot, float rope_base, int window,
                  void *kcache, void *vcache, int cache_len,
                  const float *sinks, float *out);

/* MoE over a batch: x [ntok][4096]; the assignment list is grouped by job,
 * entry a = t*topk + k; weights [ntok*topk]. out[t] += sum_k w * expert(x). */
int m2g_moe(const m2_moe_job *jobs_dev, int njobs, const int *assign_dev,
            const float *weights_dev, int ntok, int topk,
            const float *x, float *h_buf, float *y_buf, float *out,
            int n_embd, int n_ff, size_t gate_bytes, size_t up_bytes);

#ifdef __cplusplus
}
#endif
#endif
