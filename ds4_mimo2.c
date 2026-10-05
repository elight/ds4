/* ds4-mimo2: MiMo V2.6 Flash (GGUF arch "mimo2") on one CUDA GPU.
 *
 * Non-routed weights (attention, routers, layer-0 dense FFN, head) and the KV
 * cache live in VRAM. The 12,032 routed experts live in three tiers:
 *
 *   VRAM slots   the hottest experts; the GPU computes them in place
 *   RAM arena    pinned host memory, filled from SSD; copied up on a miss
 *   SSD          the GGUF itself, read with O_DIRECT into the arena
 *
 * Placement is seeded from a routing profile when one is given and then kept
 * by recency. Design notes: docs/RAM_EXPERT_TIER.md. The idea of a pinned RAM
 * tier between SSD and VRAM, profile-seeded placement and CPU help for RAM
 * hits come from Strata (github.com/Niko1221/Strata, MIT).
 *
 * The model graph mirrors llama.cpp src/models/mimo2.cpp (MIT). */
#define _GNU_SOURCE
#include "ds4_mimo2.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* ---- shape (checked against the GGUF at load) ---------------------------- */

#define M2_NL      48
#define M2_EMBD    4096
#define M2_NHEAD   64
#define M2_HK      192
#define M2_HV      128
#define M2_NROT    64
#define M2_NE      256
#define M2_TOPK    8
#define M2_FF_EXP  2048
#define M2_FF      16384
#define M2_SWA     128
#define M2_EPS     1e-6f
#define M2_VSCALE  0.707f

static void die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "ds4-mimo2: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory (%zu bytes)", n);
    return p;
}

static void *xcalloc(size_t n, size_t s) {
    void *p = calloc(n ? n : 1, s ? s : 1);
    if (!p) die("out of memory");
    return p;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#define GCK(x) do { if ((x) != 0) die("gpu call failed: %s (%s:%d)", #x, __FILE__, __LINE__); } while (0)

/* ---- GGUF (split-aware) -------------------------------------------------- */

enum { GV_U8, GV_I8, GV_U16, GV_I16, GV_U32, GV_I32, GV_F32, GV_BOOL, GV_STR, GV_ARR, GV_U64, GV_I64, GV_F64 };

typedef struct { const char *ptr; uint64_t len; } m2_str;

typedef struct {
    m2_str key;
    uint32_t type;
    const uint8_t *val;
} m2_kv;

typedef struct {
    m2_str name;
    uint32_t type;
    uint32_t nd;
    uint64_t ne[4];
    uint64_t off;       /* absolute file offset */
    int shard;
    uint64_t bytes;
} m2_tensor;

typedef struct {
    int nshards;
    int fd[8];
    int fd_direct[8];
    const uint8_t *map[8];
    uint64_t size[8];
    m2_kv *kv;
    int nkv;
    m2_tensor *t;
    int nt;
} m2_gguf;

static size_t gv_scalar_size(uint32_t t) {
    switch (t) {
    case GV_U8: case GV_I8: case GV_BOOL: return 1;
    case GV_U16: case GV_I16: return 2;
    case GV_U32: case GV_I32: case GV_F32: return 4;
    case GV_U64: case GV_I64: case GV_F64: return 8;
    }
    return 0;
}

static const uint8_t *gv_skip(const uint8_t *p, uint32_t t) {
    if (t == GV_STR) { uint64_t n; memcpy(&n, p, 8); return p + 8 + n; }
    if (t == GV_ARR) {
        uint32_t et; uint64_t n;
        memcpy(&et, p, 4); memcpy(&n, p + 4, 8);
        p += 12;
        if (et == GV_STR) { for (uint64_t i = 0; i < n; i++) p = gv_skip(p, GV_STR); return p; }
        return p + n * gv_scalar_size(et);
    }
    return p + gv_scalar_size(t);
}

static uint64_t m2_type_bytes(uint32_t type, uint64_t nelem) {
    switch (type) {
    case M2_T_F32:   return nelem * 4;
    case M2_T_Q8_0:  return nelem / 32 * 34;
    case M2_T_MXFP4: return nelem / 32 * 17;
    case M2_T_Q2_K:  return nelem / 256 * 84;
    case M2_T_Q6_K:  return nelem / 256 * 210;
    }
    die("tensor type %u is not supported by ds4-mimo2", type);
    return 0;
}

static void gguf_parse_one(m2_gguf *g, int s, bool keep_kv) {
    const uint8_t *p = g->map[s];
    if (memcmp(p, "GGUF", 4) != 0) die("shard %d is not a GGUF file", s + 1);
    uint64_t nt, nkv;
    memcpy(&nt, p + 8, 8);
    memcpy(&nkv, p + 16, 8);
    p += 24;
    uint32_t align = 32;
    for (uint64_t i = 0; i < nkv; i++) {
        m2_str key;
        memcpy(&key.len, p, 8);
        key.ptr = (const char *)p + 8;
        p += 8 + key.len;
        uint32_t type;
        memcpy(&type, p, 4);
        p += 4;
        if (key.len == 17 && !memcmp(key.ptr, "general.alignment", 17)) memcpy(&align, p, 4);
        if (keep_kv) {
            g->kv = realloc(g->kv, (size_t)(g->nkv + 1) * sizeof(m2_kv));
            g->kv[g->nkv++] = (m2_kv){ key, type, p };
        }
        p = gv_skip(p, type);
    }
    const int t0 = g->nt;
    g->t = realloc(g->t, (size_t)(g->nt + nt) * sizeof(m2_tensor));
    for (uint64_t i = 0; i < nt; i++) {
        m2_tensor *t = &g->t[g->nt++];
        memset(t, 0, sizeof(*t));
        memcpy(&t->name.len, p, 8);
        t->name.ptr = (const char *)p + 8;
        p += 8 + t->name.len;
        memcpy(&t->nd, p, 4);
        p += 4;
        uint64_t nel = 1;
        for (uint32_t d = 0; d < t->nd; d++) { memcpy(&t->ne[d], p, 8); nel *= t->ne[d]; p += 8; }
        memcpy(&t->type, p, 4);
        memcpy(&t->off, p + 4, 8);
        p += 12;
        t->shard = s;
        t->bytes = m2_type_bytes(t->type, nel);
    }
    uint64_t data = (uint64_t)(p - g->map[s]);
    data = (data + align - 1) / align * align;
    for (int i = t0; i < g->nt; i++) {
        g->t[i].off += data;
        if (g->t[i].off + g->t[i].bytes > g->size[s]) die("tensor %.*s runs past the end of shard %d (incomplete download?)",
                                                          (int)g->t[i].name.len, g->t[i].name.ptr, s + 1);
    }
}

static void gguf_map(m2_gguf *g, int s, const char *path) {
    g->fd[s] = open(path, O_RDONLY);
    if (g->fd[s] < 0) die("cannot open %s: %s", path, strerror(errno));
    g->fd_direct[s] = open(path, O_RDONLY | O_DIRECT);
    if (g->fd_direct[s] < 0) g->fd_direct[s] = -1;
    struct stat st;
    fstat(g->fd[s], &st);
    g->size[s] = (uint64_t)st.st_size;
    g->map[s] = mmap(NULL, g->size[s], PROT_READ, MAP_SHARED, g->fd[s], 0);
    if (g->map[s] == MAP_FAILED) die("mmap %s failed", path);
}

static void gguf_open(m2_gguf *g, const char *path, bool meta_only) {
    memset(g, 0, sizeof(*g));
    gguf_map(g, 0, path);
    gguf_parse_one(g, 0, true);
    g->nshards = 1;
    /* split.count > 1: the siblings differ only in the "-0000N-of-" index */
    for (int i = 0; i < g->nkv; i++) {
        if (g->kv[i].key.len == 11 && !memcmp(g->kv[i].key.ptr, "split.count", 11)) {
            uint16_t n;
            memcpy(&n, g->kv[i].val, 2);
            g->nshards = n;
        }
    }
    if (g->nshards > 8) die("too many shards");
    if (meta_only) return;
    for (int s = 1; s < g->nshards; s++) {
        char *p2 = strdup(path);
        char *m = strstr(p2, "-00001-of-");
        if (!m) die("cannot derive shard %d name from %s", s + 1, path);
        char idx[8];
        snprintf(idx, sizeof idx, "%05d", s + 1);
        memcpy(m + 1, idx, 5);
        gguf_map(g, s, p2);
        gguf_parse_one(g, s, false);
        free(p2);
    }
}

static const m2_kv *gguf_kv(const m2_gguf *g, const char *key) {
    const size_t n = strlen(key);
    for (int i = 0; i < g->nkv; i++)
        if (g->kv[i].key.len == n && !memcmp(g->kv[i].key.ptr, key, n)) return &g->kv[i];
    return NULL;
}

static int64_t gguf_int(const m2_gguf *g, const char *key, int64_t def) {
    const m2_kv *kv = gguf_kv(g, key);
    if (!kv) return def;
    switch (kv->type) {
    case GV_U8: return *(const uint8_t *)kv->val;
    case GV_I8: return *(const int8_t *)kv->val;
    case GV_U16: { uint16_t v; memcpy(&v, kv->val, 2); return v; }
    case GV_I16: { int16_t v; memcpy(&v, kv->val, 2); return v; }
    case GV_U32: { uint32_t v; memcpy(&v, kv->val, 4); return v; }
    case GV_I32: { int32_t v; memcpy(&v, kv->val, 4); return v; }
    case GV_U64: case GV_I64: { int64_t v; memcpy(&v, kv->val, 8); return v; }
    case GV_BOOL: return *kv->val;
    }
    return def;
}

static double gguf_float(const m2_gguf *g, const char *key, double def) {
    const m2_kv *kv = gguf_kv(g, key);
    if (!kv) return def;
    if (kv->type == GV_F32) { float v; memcpy(&v, kv->val, 4); return v; }
    if (kv->type == GV_F64) { double v; memcpy(&v, kv->val, 8); return v; }
    return (double)gguf_int(g, key, (int64_t)def);
}

static int gguf_arr_i32(const m2_gguf *g, const char *key, int *out, int max) {
    const m2_kv *kv = gguf_kv(g, key);
    if (!kv || kv->type != GV_ARR) return -1;
    uint32_t et; uint64_t n;
    memcpy(&et, kv->val, 4); memcpy(&n, kv->val + 4, 8);
    const uint8_t *p = kv->val + 12;
    if (n > (uint64_t)max) n = (uint64_t)max;
    for (uint64_t i = 0; i < n; i++) {
        if (et == GV_I32 || et == GV_U32) { int32_t v; memcpy(&v, p + 4 * i, 4); out[i] = v; }
        else if (et == GV_U8 || et == GV_I8 || et == GV_BOOL) out[i] = p[i];
        else return -1;
    }
    return (int)n;
}

static const m2_tensor *gguf_tensor(const m2_gguf *g, const char *name, bool required) {
    const size_t n = strlen(name);
    for (int i = 0; i < g->nt; i++)
        if (g->t[i].name.len == n && !memcmp(g->t[i].name.ptr, name, n)) return &g->t[i];
    if (required) die("tensor %s missing from the GGUF", name);
    return NULL;
}

static const uint8_t *tensor_data(const m2_gguf *g, const m2_tensor *t) {
    return g->map[t->shard] + t->off;
}

/* ---- tokenizer (GPT-2 byte BPE, qwen2 pre-tokenizer) --------------------- */

typedef struct {
    m2_str *keys;
    int *vals;
    uint32_t cap;
} m2_map;

static uint32_t fnv1a(const char *s, uint64_t n) {
    uint32_t h = 2166136261u;
    for (uint64_t i = 0; i < n; i++) { h ^= (uint8_t)s[i]; h *= 16777619u; }
    return h;
}

static void map_init(m2_map *m, uint32_t n) {
    m->cap = 1;
    while (m->cap < n * 2) m->cap <<= 1;
    m->keys = xcalloc(m->cap, sizeof(m2_str));
    m->vals = xcalloc(m->cap, sizeof(int));
}

static void map_put(m2_map *m, m2_str k, int v) {
    uint32_t i = fnv1a(k.ptr, k.len) & (m->cap - 1);
    while (m->keys[i].ptr) {
        if (m->keys[i].len == k.len && !memcmp(m->keys[i].ptr, k.ptr, k.len)) return; /* first wins */
        i = (i + 1) & (m->cap - 1);
    }
    m->keys[i] = k;
    m->vals[i] = v;
}

static int map_get(const m2_map *m, const char *s, uint64_t n) {
    uint32_t i = fnv1a(s, n) & (m->cap - 1);
    while (m->keys[i].ptr) {
        if (m->keys[i].len == n && !memcmp(m->keys[i].ptr, s, n)) return m->vals[i];
        i = (i + 1) & (m->cap - 1);
    }
    return -1;
}

typedef struct {
    int n_vocab;
    m2_str *tok;
    int *type;
    m2_map tok_id;
    m2_map merge;
    int *special;     /* ids of control / user-defined tokens, longest first */
    int nspecial;
    int byte_of_cp[512];
    int eos, im_end, im_start, eot;
} m2_vocab;

static uint32_t gpt2_cp(uint8_t b) {
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174) return b;
    uint32_t n = 0;
    for (uint32_t x = 0; x < 256; x++) {
        if ((x >= 33 && x <= 126) || (x >= 161 && x <= 172) || x >= 174) continue;
        if (x == b) return 256 + n;
        n++;
    }
    return b;
}

static void utf8_put(char **p, uint32_t cp) {
    if (cp < 0x80) *(*p)++ = (char)cp;
    else if (cp < 0x800) { *(*p)++ = (char)(0xc0 | (cp >> 6)); *(*p)++ = (char)(0x80 | (cp & 0x3f)); }
    else if (cp < 0x10000) {
        *(*p)++ = (char)(0xe0 | (cp >> 12)); *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else {
        *(*p)++ = (char)(0xf0 | (cp >> 18)); *(*p)++ = (char)(0x80 | ((cp >> 12) & 0x3f));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f)); *(*p)++ = (char)(0x80 | (cp & 0x3f));
    }
}

static uint32_t utf8_get(const char *s, uint64_t len, uint64_t *pos) {
    const uint8_t c = (uint8_t)s[*pos];
    int n = c < 0x80 ? 1 : (c & 0xe0) == 0xc0 ? 2 : (c & 0xf0) == 0xe0 ? 3 : (c & 0xf8) == 0xf0 ? 4 : 1;
    if (*pos + (uint64_t)n > len) n = 1;
    uint32_t cp = n == 1 ? c : n == 2 ? (c & 0x1f) : n == 3 ? (c & 0x0f) : (c & 0x07);
    for (int i = 1; i < n; i++) cp = (cp << 6) | ((uint8_t)s[*pos + i] & 0x3f);
    *pos += (uint64_t)n;
    return cp;
}

#include "ds4_qwen4_unicode.inc"

static bool urange_has(const ds4_qwen4_urange *r, size_t n, uint32_t cp) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (cp < r[mid].lo) hi = mid;
        else if (cp > r[mid].hi) lo = mid + 1;
        else return true;
    }
    return false;
}

#define UCLASS(name_, cp_) urange_has(ds4_qwen4_##name_##_ranges, \
    sizeof(ds4_qwen4_##name_##_ranges) / sizeof(ds4_qwen4_##name_##_ranges[0]), (cp_))

typedef struct { uint32_t cp; uint64_t next; bool valid, letter, number, space; } m2_ch;

static m2_ch ch_at(const char *s, uint64_t len, uint64_t pos) {
    m2_ch c;
    memset(&c, 0, sizeof c);
    if (pos >= len) return c;
    c.valid = true;
    c.next = pos;
    c.cp = utf8_get(s, len, &c.next);
    c.letter = UCLASS(letter, c.cp);
    c.number = UCLASS(number, c.cp);
    c.space = UCLASS(space, c.cp);
    return c;
}

typedef struct { int *v; int n, cap; } m2_toks;

static void toks_push(m2_toks *t, int id) {
    if (t->n == t->cap) { t->cap = t->cap ? t->cap * 2 : 256; t->v = realloc(t->v, (size_t)t->cap * sizeof(int)); }
    t->v[t->n++] = id;
}

static void bpe_piece(const m2_vocab *vb, const char *raw, uint64_t rlen, m2_toks *out) {
    char *enc = xmalloc(rlen * 4 + 1), *p = enc;
    for (uint64_t i = 0; i < rlen; i++) utf8_put(&p, gpt2_cp((uint8_t)raw[i]));
    const uint64_t elen = (uint64_t)(p - enc);
    int n = 0;
    uint64_t *st = xmalloc((elen + 1) * sizeof(uint64_t)), *ln = xmalloc((elen + 1) * sizeof(uint64_t));
    for (uint64_t off = 0; off < elen;) {
        uint64_t nx = off;
        utf8_get(enc, elen, &nx);
        st[n] = off; ln[n] = nx - off; n++;
        off = nx;
    }
    char buf[1024];
    for (;;) {
        int best = -1, best_rank = INT32_MAX;
        for (int i = 0; i + 1 < n; i++) {
            const uint64_t l = ln[i] + 1 + ln[i + 1];
            if (l > sizeof buf) continue;
            memcpy(buf, enc + st[i], ln[i]);
            buf[ln[i]] = ' ';
            memcpy(buf + ln[i] + 1, enc + st[i + 1], ln[i + 1]);
            const int r = map_get(&vb->merge, buf, l);
            if (r >= 0 && r < best_rank) { best_rank = r; best = i; }
        }
        if (best < 0) break;
        ln[best] += ln[best + 1];
        for (int j = best + 1; j + 1 < n; j++) { st[j] = st[j + 1]; ln[j] = ln[j + 1]; }
        n--;
    }
    for (int i = 0; i < n; i++) {
        int id = map_get(&vb->tok_id, enc + st[i], ln[i]);
        if (id >= 0) { toks_push(out, id); continue; }
        uint64_t q = st[i];   /* fall back to single characters */
        while (q < st[i] + ln[i]) {
            uint64_t nx = q;
            utf8_get(enc, elen, &nx);
            id = map_get(&vb->tok_id, enc + q, nx - q);
            if (id >= 0) toks_push(out, id);
            q = nx;
        }
    }
    free(st); free(ln); free(enc);
}

/* qwen2 pre-tokenizer:
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N} |
 *    ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+            */
static void bpe_text(const m2_vocab *vb, const char *text, uint64_t len, m2_toks *out) {
    uint64_t pos = 0;
    while (pos < len) {
        const uint64_t start = pos;
        m2_ch cur = ch_at(text, len, pos);
        if (cur.cp == '\'' && cur.next < len) {
            m2_ch n1 = ch_at(text, len, cur.next);
            const uint32_t c1 = n1.cp < 128 ? (uint32_t)tolower((int)n1.cp) : n1.cp;
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                pos = n1.next; bpe_piece(vb, text + start, pos - start, out); continue;
            }
            if (n1.next < len) {
                m2_ch n2 = ch_at(text, len, n1.next);
                const uint32_t c2 = n2.cp < 128 ? (uint32_t)tolower((int)n2.cp) : n2.cp;
                if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
                    pos = n2.next; bpe_piece(vb, text + start, pos - start, out); continue;
                }
            }
        }
        {
            uint64_t run = UINT64_MAX;
            if (cur.letter) run = cur.next;
            else if (cur.cp != '\r' && cur.cp != '\n' && !cur.number) {
                m2_ch n1 = ch_at(text, len, cur.next);
                if (n1.valid && n1.letter) run = n1.next;
            }
            if (run != UINT64_MAX) {
                pos = run;
                while (pos < len) { m2_ch s = ch_at(text, len, pos); if (!s.letter) break; pos = s.next; }
                bpe_piece(vb, text + start, pos - start, out);
                continue;
            }
        }
        if (cur.number) { pos = cur.next; bpe_piece(vb, text + start, pos - start, out); continue; }
        {
            m2_ch pu = cur;
            uint64_t pp = pos;
            if (cur.cp == ' ') { pp = cur.next; pu = ch_at(text, len, pp); }
            if (pu.valid && !pu.space && !pu.letter && !pu.number) {
                pos = pp;
                while (pos < len) { m2_ch s = ch_at(text, len, pos); if (s.space || s.letter || s.number) break; pos = s.next; }
                while (pos < len) { m2_ch s = ch_at(text, len, pos); if (s.cp != '\r' && s.cp != '\n') break; pos = s.next; }
                bpe_piece(vb, text + start, pos - start, out);
                continue;
            }
        }
        if (cur.space) {
            uint64_t p = pos, last_nl_end = 0, last_ws = pos;
            int ns = 0;
            while (p < len) {
                m2_ch s = ch_at(text, len, p);
                if (!s.space) break;
                last_ws = p;
                if (s.cp == '\r' || s.cp == '\n') last_nl_end = s.next;
                p = s.next;
                ns++;
            }
            if (last_nl_end) pos = last_nl_end;
            else if (ns > 1 && p < len) pos = last_ws;
            else pos = p;
            bpe_piece(vb, text + start, pos - start, out);
            continue;
        }
        pos = cur.next;
        bpe_piece(vb, text + start, pos - start, out);
    }
}

/* Splits out special tokens (exact string match), BPEs the rest. */
static void tokenize(const m2_vocab *vb, const char *text, m2_toks *out) {
    const uint64_t len = strlen(text);
    uint64_t seg = 0, pos = 0;
    while (pos < len) {
        int hit = -1;
        if (text[pos] == '<' || text[pos] == '[') {
            for (int i = 0; i < vb->nspecial; i++) {
                const m2_str s = vb->tok[vb->special[i]];
                if (s.len <= len - pos && !memcmp(text + pos, s.ptr, s.len)) { hit = vb->special[i]; break; }
            }
        }
        if (hit < 0) { pos++; continue; }
        if (pos > seg) bpe_text(vb, text + seg, pos - seg, out);
        toks_push(out, hit);
        pos += vb->tok[hit].len;
        seg = pos;
    }
    if (pos > seg) bpe_text(vb, text + seg, pos - seg, out);
}

/* Appends the bytes of one token to buf; returns the count. */
static int detok(const m2_vocab *vb, int id, char *buf, int cap) {
    if (id < 0 || id >= vb->n_vocab) return 0;
    const m2_str s = vb->tok[id];
    if (vb->type[id] == 3 || vb->type[id] == 4) {
        const int n = (int)s.len < cap ? (int)s.len : cap;
        memcpy(buf, s.ptr, (size_t)n);
        return n;
    }
    int n = 0;
    uint64_t p = 0;
    while (p < s.len && n < cap) {
        const uint32_t cp = utf8_get(s.ptr, s.len, &p);
        buf[n++] = (char)(cp < 512 && vb->byte_of_cp[cp] >= 0 ? vb->byte_of_cp[cp] : '?');
    }
    return n;
}

static int cmp_special_len(const void *a, const void *b, void *ctx) {
    const m2_vocab *vb = ctx;
    const uint64_t la = vb->tok[*(const int *)a].len, lb = vb->tok[*(const int *)b].len;
    return la < lb ? 1 : la > lb ? -1 : 0;
}

static void vocab_load(m2_vocab *vb, const m2_gguf *g) {
    memset(vb, 0, sizeof *vb);
    const m2_kv *kt = gguf_kv(g, "tokenizer.ggml.tokens");
    const m2_kv *ky = gguf_kv(g, "tokenizer.ggml.token_type");
    const m2_kv *km = gguf_kv(g, "tokenizer.ggml.merges");
    if (!kt || !ky || !km) die("tokenizer missing from GGUF");
    uint64_t n;
    memcpy(&n, kt->val + 4, 8);
    vb->n_vocab = (int)n;
    vb->tok = xmalloc(n * sizeof(m2_str));
    vb->type = xmalloc(n * sizeof(int));
    map_init(&vb->tok_id, (uint32_t)n);
    const uint8_t *p = kt->val + 12;
    for (uint64_t i = 0; i < n; i++) {
        memcpy(&vb->tok[i].len, p, 8);
        vb->tok[i].ptr = (const char *)p + 8;
        p += 8 + vb->tok[i].len;
        map_put(&vb->tok_id, vb->tok[i], (int)i);
    }
    gguf_arr_i32(g, "tokenizer.ggml.token_type", vb->type, (int)n);
    uint64_t nm;
    memcpy(&nm, km->val + 4, 8);
    map_init(&vb->merge, (uint32_t)nm);
    p = km->val + 12;
    for (uint64_t i = 0; i < nm; i++) {
        m2_str s;
        memcpy(&s.len, p, 8);
        s.ptr = (const char *)p + 8;
        p += 8 + s.len;
        map_put(&vb->merge, s, (int)i);
    }
    vb->special = xmalloc(n * sizeof(int));
    for (uint64_t i = 0; i < n; i++)
        if (vb->type[i] == 3 || vb->type[i] == 4) vb->special[vb->nspecial++] = (int)i;
    qsort_r(vb->special, (size_t)vb->nspecial, sizeof(int), cmp_special_len, vb);
    for (int i = 0; i < 512; i++) vb->byte_of_cp[i] = -1;
    for (int b = 0; b < 256; b++) vb->byte_of_cp[gpt2_cp((uint8_t)b)] = b;
    vb->eos = (int)gguf_int(g, "tokenizer.ggml.eos_token_id", 151645);
    vb->im_end = map_get(&vb->tok_id, "<|im_end|>", 10);
    vb->im_start = map_get(&vb->tok_id, "<|im_start|>", 12);
    vb->eot = map_get(&vb->tok_id, "<|endoftext|>", 13);
}

/* ---- model --------------------------------------------------------------- */

typedef struct {
    int swa;            /* sliding-window layer */
    int n_kv;
    float rope_base;
    const float *attn_norm, *ffn_norm, *sinks;   /* device */
    const void *qkv, *wo;                        /* Q8_0, device */
    int qkv_rows;
    const void *ffn_gate, *ffn_up, *ffn_down;    /* layer 0 only, Q8_0 */
    const float *router;                         /* F32 [256][4096], device */
    float bias[M2_NE];                           /* host */
    uint64_t gate_off, up_off, down_off;         /* expert tensors in the file */
    int exp_shard;
    void *kc, *vc;
    int cache_len;
} m2_layer;

#define M2_GATE_BYTES ((size_t)M2_FF_EXP * (M2_EMBD / 256) * 84)   /* 2,752,512 */
#define M2_DOWN_BYTES ((size_t)M2_EMBD * (M2_FF_EXP / 32) * 17)    /* 4,456,448 */
#define M2_EXP_BYTES  (2 * M2_GATE_BYTES + M2_DOWN_BYTES)

/* ---- SSD reads: a small pool doing O_DIRECT preads ----------------------- */

typedef struct { int fd; uint64_t off; size_t len; void *dst; } m2_io;

typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t cv_work, cv_done;
    m2_io *q;
    int qn, qcap, qhead;
    int pending;
    int stop;
    pthread_t th[16];
    int nth;
} m2_iopool;

static void *io_main(void *arg) {
    m2_iopool *p = arg;
    for (;;) {
        pthread_mutex_lock(&p->mu);
        while (p->qhead == p->qn && !p->stop) pthread_cond_wait(&p->cv_work, &p->mu);
        if (p->stop) { pthread_mutex_unlock(&p->mu); return NULL; }
        m2_io r = p->q[p->qhead++];
        pthread_mutex_unlock(&p->mu);
        size_t done = 0;
        while (done < r.len) {
            ssize_t n = pread(r.fd, (uint8_t *)r.dst + done, r.len - done, (off_t)(r.off + done));
            if (n <= 0) {
                if (n < 0 && errno == EINTR) continue;
                die("expert read failed at %llu: %s", (unsigned long long)(r.off + done), n < 0 ? strerror(errno) : "EOF");
            }
            done += (size_t)n;
        }
        pthread_mutex_lock(&p->mu);
        if (--p->pending == 0) pthread_cond_broadcast(&p->cv_done);
        pthread_mutex_unlock(&p->mu);
    }
}

static void io_init(m2_iopool *p, int nth) {
    memset(p, 0, sizeof *p);
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->cv_work, NULL);
    pthread_cond_init(&p->cv_done, NULL);
    p->nth = nth;
    for (int i = 0; i < nth; i++) pthread_create(&p->th[i], NULL, io_main, p);
}

static void io_submit(m2_iopool *p, m2_io r) {
    pthread_mutex_lock(&p->mu);
    if (p->qhead == p->qn) p->qhead = p->qn = 0;
    if (p->qn == p->qcap) { p->qcap = p->qcap ? p->qcap * 2 : 1024; p->q = realloc(p->q, (size_t)p->qcap * sizeof(m2_io)); }
    p->q[p->qn++] = r;
    p->pending++;
    pthread_cond_signal(&p->cv_work);
    pthread_mutex_unlock(&p->mu);
}

static void io_wait(m2_iopool *p) {
    pthread_mutex_lock(&p->mu);
    while (p->pending > 0) pthread_cond_wait(&p->cv_done, &p->mu);
    pthread_mutex_unlock(&p->mu);
}

/* ---- expert tiers -------------------------------------------------------- */

#define M2_NEXP (M2_NL * M2_NE)

typedef struct {
    int nslots;
    uint8_t *slots;                 /* device, nslots * M2_EXP_BYTES */
    int slot_of[M2_NEXP];
    int *slot_owner;
    uint64_t *slot_used;

    int nram;
    uint8_t *arena;                 /* pinned, nram * ram_stride */
    size_t region[3], ram_stride;
    int ram_of[M2_NEXP];
    int *ram_owner;
    uint64_t *ram_used;

    uint64_t tick;
    uint64_t hit_vram, hit_ram, miss_ssd, ssd_bytes;
    double t_ssd, t_wait;
    uint32_t count[M2_NEXP];        /* routing profile */
} m2_tiers;

typedef struct {
    m2_gguf g;
    m2_vocab vb;
    m2_layer L[M2_NL];
    const m2_tensor *tok_embd;
    const float *out_norm;
    const void *output;
    int ctx, ubatch;
    m2_tiers tr;
    m2_iopool io;
    /* device activations */
    float *x, *xn, *qkv, *att, *tmp, *gt, *u, *h, *y, *rlog, *logits, *wts;
    int *assign;
    m2_moe_job *jobs;
    /* host staging (pinned) */
    float *h_x, *h_rlog, *h_logits, *h_wts;
    int *h_assign;
    m2_moe_job *h_jobs;
    int verbose;
} m2_model;

static void *upload_tensor(const m2_gguf *g, const m2_tensor *t) {
    void *d = m2g_alloc(t->bytes);
    if (!d) die("out of VRAM loading %.*s", (int)t->name.len, t->name.ptr);
    GCK(m2g_upload(d, tensor_data(g, t), t->bytes));
    posix_fadvise(g->fd[t->shard], (off_t)t->off, (off_t)t->bytes, POSIX_FADV_DONTNEED);
    return d;
}

static const m2_tensor *layer_tensor(const m2_gguf *g, int l, const char *what, bool req) {
    char name[96];
    snprintf(name, sizeof name, "blk.%d.%s", l, what);
    return gguf_tensor(g, name, req);
}

static void check_tensor(const m2_tensor *t, uint32_t type, uint64_t ne0, uint64_t ne1, uint64_t ne2) {
    if (t->type != type || t->ne[0] != ne0 || (ne1 && t->ne[1] != ne1) || (ne2 && t->ne[2] != ne2))
        die("tensor %.*s: type %u shape [%llu,%llu,%llu], expected type %u [%llu,%llu,%llu]",
            (int)t->name.len, t->name.ptr, t->type,
            (unsigned long long)t->ne[0], (unsigned long long)t->ne[1], (unsigned long long)t->ne[2],
            type, (unsigned long long)ne0, (unsigned long long)ne1, (unsigned long long)ne2);
}

static void model_load(m2_model *M, const char *path, bool meta_only) {
    m2_gguf *g = &M->g;
    gguf_open(g, path, meta_only);
    const m2_kv *arch = gguf_kv(g, "general.architecture");
    if (!arch || memcmp(arch->val + 8, "mimo2", 5) != 0) die("%s is not a mimo2 GGUF", path);
    if (gguf_int(g, "mimo2.block_count", 0) != M2_NL || gguf_int(g, "mimo2.embedding_length", 0) != M2_EMBD ||
        gguf_int(g, "mimo2.attention.head_count", 0) != M2_NHEAD || gguf_int(g, "mimo2.expert_count", 0) != M2_NE ||
        gguf_int(g, "mimo2.expert_used_count", 0) != M2_TOPK || gguf_int(g, "mimo2.attention.key_length", 0) != M2_HK ||
        gguf_int(g, "mimo2.attention.value_length", 0) != M2_HV || gguf_int(g, "mimo2.rope.dimension_count", 0) != M2_NROT ||
        gguf_int(g, "mimo2.attention.sliding_window", 0) != M2_SWA ||
        gguf_int(g, "mimo2.expert_feed_forward_length", 0) != M2_FF_EXP)
        die("unexpected mimo2 hyperparameters (this build knows MiMo V2.6 Flash only)");
    int n_kv[M2_NL], swa[M2_NL];
    if (gguf_arr_i32(g, "mimo2.attention.head_count_kv", n_kv, M2_NL) != M2_NL) die("head_count_kv array missing");
    if (gguf_arr_i32(g, "mimo2.attention.sliding_window_pattern", swa, M2_NL) != M2_NL) die("sliding_window_pattern missing");
    const float base_full = (float)gguf_float(g, "mimo2.rope.freq_base", 1e7);
    const float base_swa = (float)gguf_float(g, "mimo2.rope.freq_base_swa", 1e4);
    vocab_load(&M->vb, g);
    for (int l = 0; l < M2_NL; l++) {
        M->L[l].swa = swa[l];
        M->L[l].n_kv = n_kv[l];
        M->L[l].rope_base = swa[l] ? base_swa : base_full;
    }
}

/* Puts the non-routed weights in VRAM and records where the experts live. */
static void model_upload(m2_model *M) {
    m2_gguf *g = &M->g;

    M->tok_embd = gguf_tensor(g, "token_embd.weight", true);
    check_tensor(M->tok_embd, M2_T_Q8_0, M2_EMBD, (uint64_t)M->vb.n_vocab, 0);
    const m2_tensor *on = gguf_tensor(g, "output_norm.weight", true);
    M->out_norm = upload_tensor(g, on);
    const m2_tensor *out = gguf_tensor(g, "output.weight", true);
    check_tensor(out, M2_T_Q6_K, M2_EMBD, (uint64_t)M->vb.n_vocab, 0);
    M->output = upload_tensor(g, out);

    for (int l = 0; l < M2_NL; l++) {
        m2_layer *L = &M->L[l];
        L->attn_norm = upload_tensor(g, layer_tensor(g, l, "attn_norm.weight", true));
        L->ffn_norm = upload_tensor(g, layer_tensor(g, l, "ffn_norm.weight", true));
        const m2_tensor *s = layer_tensor(g, l, "attn_sinks.weight", false);
        L->sinks = s ? upload_tensor(g, s) : NULL;
        const m2_tensor *qkv = layer_tensor(g, l, "attn_qkv.weight", true);
        L->qkv_rows = M2_NHEAD * M2_HK + L->n_kv * (M2_HK + M2_HV);
        check_tensor(qkv, M2_T_Q8_0, M2_EMBD, (uint64_t)L->qkv_rows, 0);
        L->qkv = upload_tensor(g, qkv);
        const m2_tensor *wo = layer_tensor(g, l, "attn_output.weight", true);
        check_tensor(wo, M2_T_Q8_0, M2_NHEAD * M2_HV, M2_EMBD, 0);
        L->wo = upload_tensor(g, wo);
        const m2_tensor *gi = layer_tensor(g, l, "ffn_gate_inp.weight", false);
        if (!gi) {
            const m2_tensor *fg = layer_tensor(g, l, "ffn_gate.weight", true);
            const m2_tensor *fu = layer_tensor(g, l, "ffn_up.weight", true);
            const m2_tensor *fd = layer_tensor(g, l, "ffn_down.weight", true);
            check_tensor(fg, M2_T_Q8_0, M2_EMBD, M2_FF, 0);
            check_tensor(fu, M2_T_Q8_0, M2_EMBD, M2_FF, 0);
            check_tensor(fd, M2_T_Q8_0, M2_FF, M2_EMBD, 0);
            L->ffn_gate = upload_tensor(g, fg);
            L->ffn_up = upload_tensor(g, fu);
            L->ffn_down = upload_tensor(g, fd);
        } else {
            check_tensor(gi, M2_T_F32, M2_EMBD, M2_NE, 0);
            L->router = upload_tensor(g, gi);
            const m2_tensor *b = layer_tensor(g, l, "exp_probs_b.bias", true);
            memcpy(L->bias, tensor_data(g, b), sizeof L->bias);
            const m2_tensor *eg = layer_tensor(g, l, "ffn_gate_exps.weight", true);
            const m2_tensor *eu = layer_tensor(g, l, "ffn_up_exps.weight", true);
            const m2_tensor *ed = layer_tensor(g, l, "ffn_down_exps.weight", true);
            check_tensor(eg, M2_T_Q2_K, M2_EMBD, M2_FF_EXP, M2_NE);
            check_tensor(eu, M2_T_Q2_K, M2_EMBD, M2_FF_EXP, M2_NE);
            check_tensor(ed, M2_T_MXFP4, M2_FF_EXP, M2_EMBD, M2_NE);
            if (eg->shard != eu->shard || eg->shard != ed->shard) die("expert tensors split across shards");
            L->gate_off = eg->off;
            L->up_off = eu->off;
            L->down_off = ed->off;
            L->exp_shard = eg->shard;
        }
    }
}

static void alloc_runtime(m2_model *M) {
    const int U = M->ubatch;
    for (int l = 0; l < M2_NL; l++) {
        m2_layer *L = &M->L[l];
        L->cache_len = L->swa ? (U + M2_SWA) : M->ctx;
        L->kc = m2g_alloc((size_t)L->cache_len * L->n_kv * M2_HK * 2);
        L->vc = m2g_alloc((size_t)L->cache_len * L->n_kv * M2_HV * 2);
        if (!L->kc || !L->vc) die("out of VRAM for the KV cache");
    }
#define DALLOC(p, n) do { (p) = m2g_alloc((size_t)(n) * sizeof(*(p))); if (!(p)) die("out of VRAM (" #p ")"); } while (0)
    DALLOC(M->x, (size_t)U * M2_EMBD);
    DALLOC(M->xn, (size_t)U * M2_EMBD);
    DALLOC(M->qkv, (size_t)U * (M2_NHEAD * M2_HK + 8 * (M2_HK + M2_HV)));
    DALLOC(M->att, (size_t)U * M2_NHEAD * M2_HV);
    DALLOC(M->tmp, (size_t)U * M2_EMBD);
    DALLOC(M->gt, (size_t)U * M2_FF);
    DALLOC(M->u, (size_t)U * M2_FF);
    M->h = M->gt;    /* MoE scratch reuses the dense-FFN buffers: [U*8][2048] fits [U][16384] */
    DALLOC(M->y, (size_t)U * M2_TOPK * M2_EMBD);
    DALLOC(M->rlog, (size_t)U * M2_NE);
    DALLOC(M->logits, (size_t)M->vb.n_vocab);
    DALLOC(M->wts, (size_t)U * M2_TOPK);
    DALLOC(M->assign, (size_t)U * M2_TOPK);
    DALLOC(M->jobs, M2_NE);
#undef DALLOC
    M->h_x = m2g_host_alloc((size_t)U * M2_EMBD * sizeof(float));
    M->h_rlog = m2g_host_alloc((size_t)U * M2_NE * sizeof(float));
    M->h_logits = m2g_host_alloc((size_t)M->vb.n_vocab * sizeof(float));
    M->h_wts = m2g_host_alloc((size_t)U * M2_TOPK * sizeof(float));
    M->h_assign = m2g_host_alloc((size_t)U * M2_TOPK * sizeof(int));
    M->h_jobs = m2g_host_alloc(M2_NE * sizeof(m2_moe_job));
    if (!M->h_x || !M->h_rlog || !M->h_logits || !M->h_wts || !M->h_assign || !M->h_jobs) die("pinned alloc failed");
}

static size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }

static void tiers_init(m2_model *M, double vram_reserve_gb, int max_slots, double ram_gb) {
    m2_tiers *T = &M->tr;
    size_t fr, tot;
    GCK(m2g_mem_info(&fr, &tot));
    const size_t reserve = (size_t)(vram_reserve_gb * 1e9);
    int n = fr > reserve ? (int)((fr - reserve) / M2_EXP_BYTES) : 0;
    if (max_slots >= 0 && n > max_slots) n = max_slots;
    if (n < M2_NE) die("only %d VRAM expert slots fit; need at least %d", n, M2_NE);
    T->nslots = n;
    T->slots = m2g_alloc((size_t)n * M2_EXP_BYTES);
    if (!T->slots) die("VRAM slot allocation failed");
    T->slot_owner = xmalloc((size_t)n * sizeof(int));
    T->slot_used = xcalloc((size_t)n, sizeof(uint64_t));
    for (int i = 0; i < n; i++) T->slot_owner[i] = -1;
    for (int i = 0; i < M2_NEXP; i++) { T->slot_of[i] = -1; T->ram_of[i] = -1; }

    T->region[0] = align_up(M2_GATE_BYTES, 4096) + 4096;
    T->region[1] = T->region[0];
    T->region[2] = align_up(M2_DOWN_BYTES, 4096) + 4096;
    T->ram_stride = T->region[0] + T->region[1] + T->region[2];
    int nr = (int)(ram_gb * 1e9 / (double)T->ram_stride);
    if (nr < M2_NE + 32) nr = M2_NE + 32;
    if (nr > M2_NEXP) nr = M2_NEXP;
    const double t0 = now_s();
    T->arena = m2g_host_alloc((size_t)nr * T->ram_stride);
    if (!T->arena) die("could not pin %.1f GB of RAM for the expert arena", nr * T->ram_stride / 1e9);
    T->nram = nr;
    T->ram_owner = xmalloc((size_t)nr * sizeof(int));
    T->ram_used = xcalloc((size_t)nr, sizeof(uint64_t));
    for (int i = 0; i < nr; i++) T->ram_owner[i] = -1;
    fprintf(stderr, "ds4-mimo2: tiers: %d VRAM slots (%.1f GB), %d RAM entries (%.1f GB pinned in %.1fs), %d experts total\n",
            n, n * (double)M2_EXP_BYTES / 1e9, nr, nr * (double)T->ram_stride / 1e9, now_s() - t0, M2_NEXP);
}

/* Least recently used entry not touched in the current tick. */
static int lru_pick(const uint64_t *used, int n, uint64_t tick) {
    int best = -1;
    uint64_t bu = UINT64_MAX;
    for (int i = 0; i < n; i++) {
        if (used[i] < bu && used[i] != tick) { bu = used[i]; best = i; if (bu == 0) break; }
    }
    return best;
}

static const uint8_t *ram_part(const m2_tiers *T, const m2_model *M, int r, int id, int part) {
    const m2_layer *L = &M->L[id / M2_NE];
    const int e = id % M2_NE;
    const uint64_t off = part == 0 ? L->gate_off + (uint64_t)e * M2_GATE_BYTES
                       : part == 1 ? L->up_off + (uint64_t)e * M2_GATE_BYTES
                                   : L->down_off + (uint64_t)e * M2_DOWN_BYTES;
    size_t roff = 0;
    for (int i = 0; i < part; i++) roff += T->region[i];
    return T->arena + (size_t)r * T->ram_stride + roff + (off & 4095);
}

/* Queue the SSD reads that bring expert `id` into RAM entry r. */
static void ram_fill(m2_model *M, int r, int id) {
    m2_tiers *T = &M->tr;
    const m2_layer *L = &M->L[id / M2_NE];
    const int e = id % M2_NE;
    const int fdd = M->g.fd_direct[L->exp_shard];
    const int fd = fdd >= 0 ? fdd : M->g.fd[L->exp_shard];
    size_t roff = 0;
    for (int part = 0; part < 3; part++) {
        const uint64_t off = part == 0 ? L->gate_off + (uint64_t)e * M2_GATE_BYTES
                           : part == 1 ? L->up_off + (uint64_t)e * M2_GATE_BYTES
                                       : L->down_off + (uint64_t)e * M2_DOWN_BYTES;
        const size_t bytes = part == 2 ? M2_DOWN_BYTES : M2_GATE_BYTES;
        const uint64_t a0 = off & ~(uint64_t)4095;
        const size_t len = align_up((size_t)(off - a0) + bytes, 4096);
        io_submit(&M->io, (m2_io){ fd, a0, len, T->arena + (size_t)r * T->ram_stride + roff });
        T->ssd_bytes += len;
        roff += T->region[part];
    }
}

static int ram_claim(m2_model *M, int id) {
    m2_tiers *T = &M->tr;
    const int r = lru_pick(T->ram_used, T->nram, T->tick);
    if (r < 0) die("RAM arena exhausted within one layer");
    if (T->ram_owner[r] >= 0) T->ram_of[T->ram_owner[r]] = -1;
    T->ram_owner[r] = id;
    T->ram_of[id] = r;
    T->ram_used[r] = T->tick;
    return r;
}

static void slot_copy_from_ram(m2_model *M, int s, int r, int id) {
    m2_tiers *T = &M->tr;
    uint8_t *dst = T->slots + (size_t)s * M2_EXP_BYTES;
    GCK(m2g_copy_async(dst, ram_part(T, M, r, id, 0), M2_GATE_BYTES));
    GCK(m2g_copy_async(dst + M2_GATE_BYTES, ram_part(T, M, r, id, 1), M2_GATE_BYTES));
    GCK(m2g_copy_async(dst + 2 * M2_GATE_BYTES, ram_part(T, M, r, id, 2), M2_DOWN_BYTES));
}

/* Make every expert in need[0..n) resident in a VRAM slot for this tick. */
static void tiers_fetch(m2_model *M, const int *need, int n, int *slot_out) {
    m2_tiers *T = &M->tr;
    T->tick++;
    int nmiss = 0, *miss = xmalloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) {
        const int id = need[i];
        T->count[id]++;
        const int s = T->slot_of[id];
        if (s >= 0) { T->slot_used[s] = T->tick; slot_out[i] = s; T->hit_vram++; continue; }
        miss[nmiss++] = i;
        if (T->ram_of[id] >= 0) T->hit_ram++;
    }
    /* protect the RAM copies of misses already resident, then read the rest */
    for (int k = 0; k < nmiss; k++) {
        const int r = T->ram_of[need[miss[k]]];
        if (r >= 0) T->ram_used[r] = T->tick;
    }
    const double t0 = now_s();
    int nssd = 0;
    char *from_ssd = xcalloc((size_t)(nmiss ? nmiss : 1), 1);
    for (int k = 0; k < nmiss; k++) {
        const int id = need[miss[k]];
        if (T->ram_of[id] >= 0) continue;
        ram_fill(M, ram_claim(M, id), id);
        from_ssd[k] = 1;
        nssd++;
    }
    T->miss_ssd += (uint64_t)nssd;
    /* pass 0 copies RAM hits up while the SSD reads run; pass 1 the rest */
    for (int pass = 0; pass < 2; pass++) {
        if (pass == 1 && nssd) { io_wait(&M->io); T->t_ssd += now_s() - t0; }
        for (int k = 0; k < nmiss; k++) {
            if (from_ssd[k] != pass) continue;
            const int i = miss[k], id = need[i];
            const int s = lru_pick(T->slot_used, T->nslots, T->tick);
        if (s < 0) die("VRAM slots exhausted within one layer");
        if (T->slot_owner[s] >= 0) T->slot_of[T->slot_owner[s]] = -1;
        T->slot_owner[s] = id;
        T->slot_of[id] = s;
        T->slot_used[s] = T->tick;
        slot_copy_from_ram(M, s, T->ram_of[id], id);
        slot_out[i] = s;
        }
    }
    if (nmiss) GCK(m2g_copy_fence());
    free(from_ssd);
    free(miss);
}

/* ---- forward ------------------------------------------------------------- */

static void embed_rows(const m2_model *M, const int *tok, int n, float *out) {
    const size_t rb = M2_EMBD / 32 * 34;
    const uint8_t *base = tensor_data(&M->g, M->tok_embd);
    for (int t = 0; t < n; t++) {
        const uint8_t *row = base + (size_t)tok[t] * rb;
        float *o = out + (size_t)t * M2_EMBD;
        for (int b = 0; b < M2_EMBD / 32; b++) {
            const uint8_t *blk = row + b * 34;
            uint16_t hb;
            memcpy(&hb, blk, 2);
            const float d = _cvtsh_ss(hb);
            const int8_t *q = (const int8_t *)(blk + 2);
            for (int i = 0; i < 32; i++) o[b * 32 + i] = d * q[i];
        }
    }
}

static float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }

/* Router on the host: sigmoid scores, top-8 by score + bias, weights are the
 * plain scores renormalised (llama.cpp build_moe_ffn, gating SIGMOID, norm_w). */
static void route(const float *logits, const float *bias, int *ids, float *w) {
    float p[M2_NE], sel[M2_NE];
    for (int e = 0; e < M2_NE; e++) { p[e] = sigmoidf(logits[e]); sel[e] = p[e] + bias[e]; }
    for (int k = 0; k < M2_TOPK; k++) {
        int best = -1;
        float bv = -INFINITY;
        for (int e = 0; e < M2_NE; e++) if (sel[e] > bv) { bv = sel[e]; best = e; }
        ids[k] = best;
        w[k] = p[best];
        sel[best] = -INFINITY;
    }
    float s = 0.f;
    for (int k = 0; k < M2_TOPK; k++) s += w[k];
    if (s < 6.103515625e-5f) s = 6.103515625e-5f;
    for (int k = 0; k < M2_TOPK; k++) w[k] /= s;
}

static void moe_layer(m2_model *M, int l, int n) {
    m2_layer *L = &M->L[l];
    GCK(m2g_matmul(M2_T_F32, L->router, M2_NE, M2_EMBD, M->xn, M2_EMBD, M->rlog, M2_NE, n, 0));
    GCK(m2g_download(M->h_rlog, M->rlog, (size_t)n * M2_NE * sizeof(float)));
    static int ids[8192 * M2_TOPK];
    int cnt[M2_NE] = { 0 };
    for (int t = 0; t < n; t++) {
        route(M->h_rlog + (size_t)t * M2_NE, L->bias, ids + t * M2_TOPK, M->h_wts + t * M2_TOPK);
        for (int k = 0; k < M2_TOPK; k++) cnt[ids[t * M2_TOPK + k]]++;
    }
    int need[M2_NE], expert_of_job[M2_NE], nj = 0;
    for (int e = 0; e < M2_NE; e++) if (cnt[e]) { expert_of_job[nj] = e; need[nj] = l * M2_NE + e; nj++; }
    int slot[M2_NE];
    for (int j = 0; j < nj; j++) slot[j] = -1;
    tiers_fetch(M, need, nj, slot);
    int job_of_e[M2_NE], fill[M2_NE];
    int off = 0;
    for (int j = 0; j < nj; j++) {
        const int e = expert_of_job[j];
        job_of_e[e] = j;
        M->h_jobs[j] = (m2_moe_job){ M->tr.slots + (size_t)slot[j] * M2_EXP_BYTES, cnt[e], off };
        fill[j] = off;
        off += cnt[e];
    }
    for (int t = 0; t < n; t++)
        for (int k = 0; k < M2_TOPK; k++) {
            const int j = job_of_e[ids[t * M2_TOPK + k]];
            M->h_assign[fill[j]++] = t * M2_TOPK + k;
        }
    GCK(m2g_upload_async(M->jobs, M->h_jobs, (size_t)nj * sizeof(m2_moe_job)));
    GCK(m2g_upload_async(M->assign, M->h_assign, (size_t)n * M2_TOPK * sizeof(int)));
    GCK(m2g_upload_async(M->wts, M->h_wts, (size_t)n * M2_TOPK * sizeof(float)));
    GCK(m2g_moe(M->jobs, nj, M->assign, M->wts, n, M2_TOPK, M->xn, M->h, M->y, M->x,
                M2_EMBD, M2_FF_EXP, M2_GATE_BYTES, M2_GATE_BYTES));
    /* the pinned staging is reused next layer; the download there orders it */
}

/* Runs n tokens at positions pos0.. through the model; if want_logits, the
 * logits of the last token land in M->h_logits. */
static void forward(m2_model *M, const int *tok, int n, int pos0, bool want_logits) {
    if (n > M->ubatch) die("batch %d exceeds ubatch %d", n, M->ubatch);
    if (pos0 + n > M->ctx) die("context full (%d)", M->ctx);
    embed_rows(M, tok, n, M->h_x);
    GCK(m2g_upload_async(M->x, M->h_x, (size_t)n * M2_EMBD * sizeof(float)));
    for (int l = 0; l < M2_NL; l++) {
        m2_layer *L = &M->L[l];
        const int ldq = L->qkv_rows;
        GCK(m2g_rmsnorm(M->x, L->attn_norm, M->xn, M2_EMBD, n, M2_EPS));
        GCK(m2g_matmul(M2_T_Q8_0, L->qkv, ldq, M2_EMBD, M->xn, M2_EMBD, M->qkv, ldq, n, 0));
        GCK(m2g_attention(M->qkv, n, pos0, M2_NHEAD, L->n_kv, M2_NROT, L->rope_base,
                          L->swa ? M2_SWA : 0, L->kc, L->vc, L->cache_len, L->sinks, M->att));
        GCK(m2g_matmul(M2_T_Q8_0, L->wo, M2_EMBD, M2_NHEAD * M2_HV, M->att, M2_NHEAD * M2_HV,
                       M->tmp, M2_EMBD, n, 0));
        GCK(m2g_scale(M->tmp, M2_VSCALE, n * M2_EMBD));
        GCK(m2g_add(M->x, M->tmp, n * M2_EMBD));
        GCK(m2g_rmsnorm(M->x, L->ffn_norm, M->xn, M2_EMBD, n, M2_EPS));
        if (L->ffn_gate) {
            GCK(m2g_matmul(M2_T_Q8_0, L->ffn_gate, M2_FF, M2_EMBD, M->xn, M2_EMBD, M->gt, M2_FF, n, 0));
            GCK(m2g_matmul(M2_T_Q8_0, L->ffn_up, M2_FF, M2_EMBD, M->xn, M2_EMBD, M->u, M2_FF, n, 0));
            GCK(m2g_swiglu2(M->gt, M->u, M->gt, n * M2_FF));
            GCK(m2g_matmul(M2_T_Q8_0, L->ffn_down, M2_EMBD, M2_FF, M->gt, M2_FF, M->x, M2_EMBD, n, 1));
        } else {
            moe_layer(M, l, n);
        }
    }
    if (want_logits) {
        const float *last = M->x + (size_t)(n - 1) * M2_EMBD;
        GCK(m2g_rmsnorm(last, M->out_norm, M->xn, M2_EMBD, 1, M2_EPS));
        GCK(m2g_matmul(M2_T_Q6_K, M->output, M->vb.n_vocab, M2_EMBD, M->xn, M2_EMBD, M->logits, M->vb.n_vocab, 1, 0));
        GCK(m2g_download(M->h_logits, M->logits, (size_t)M->vb.n_vocab * sizeof(float)));
    } else {
        GCK(m2g_sync());
    }
}

/* ---- profile -------------------------------------------------------------- */

static void profile_save(const m2_model *M, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "ds4-mimo2: cannot write profile %s\n", path); return; }
    fwrite(M->tr.count, sizeof(uint32_t), M2_NEXP, f);
    fclose(f);
}

typedef struct { uint32_t c; int id; } m2_rank;

static int cmp_rank(const void *a, const void *b) {
    const m2_rank *x = a, *y = b;
    return x->c < y->c ? 1 : x->c > y->c ? -1 : (x->id - y->id);
}

/* Seed VRAM with the hottest experts and RAM with the next ones. */
static void profile_seed(m2_model *M, const char *path) {
    m2_tiers *T = &M->tr;
    uint32_t *c = xcalloc(M2_NEXP, sizeof(uint32_t));
    FILE *f = path ? fopen(path, "rb") : NULL;
    if (f) {
        if (fread(c, sizeof(uint32_t), M2_NEXP, f) != M2_NEXP) die("short profile %s", path);
        fclose(f);
    } else if (path) {
        fprintf(stderr, "ds4-mimo2: profile %s not found, seeding uniformly\n", path);
    }
    m2_rank *rk = xmalloc(M2_NEXP * sizeof(m2_rank));
    int n = 0;
    for (int l = 0; l < M2_NL; l++) {
        if (!M->L[l].router) continue;
        for (int e = 0; e < M2_NE; e++) rk[n++] = (m2_rank){ c[l * M2_NE + e], l * M2_NE + e };
    }
    /* uniform seed (no profile): interleave layers so every layer gets a share */
    if (!f) for (int i = 0; i < n; i++) rk[i].c = (uint32_t)(M2_NEXP - ((rk[i].id % M2_NE) * M2_NL + rk[i].id / M2_NE));
    qsort(rk, (size_t)n, sizeof(m2_rank), cmp_rank);
    const double t0 = now_s();
    const int nram = T->nram < n ? T->nram : n;
    const int batch = 64;
    for (int i0 = 0; i0 < nram; i0 += batch) {
        T->tick++;
        const int i1 = i0 + batch < nram ? i0 + batch : nram;
        for (int i = i0; i < i1; i++) ram_fill(M, ram_claim(M, rk[i].id), rk[i].id);
        io_wait(&M->io);
        for (int i = i0; i < i1; i++) {
            if (i >= T->nslots) break;
            const int id = rk[i].id, s = i;
            T->slot_owner[s] = id;
            T->slot_of[id] = s;
            T->slot_used[s] = T->tick;
            slot_copy_from_ram(M, s, T->ram_of[id], id);
        }
        GCK(m2g_sync());
        if (M->verbose && (i0 / batch) % 20 == 0)
            fprintf(stderr, "\rds4-mimo2: seeding %d/%d experts", i1, nram);
    }
    T->tick++;
    const double dt = now_s() - t0;
    fprintf(stderr, "\rds4-mimo2: seeded %d VRAM + %d RAM experts from %s in %.1fs (%.2f GB/s from SSD)\n",
            T->nslots < nram ? T->nslots : nram, nram, f ? path : "uniform order", dt, T->ssd_bytes / 1e9 / dt);
    T->ssd_bytes = 0;
    free(rk);
    free(c);
}

/* ---- CLI ----------------------------------------------------------------- */

static void usage(void) {
    fprintf(stderr,
        "usage: ds4-mimo2 -m MODEL-00001-of-0000N.gguf [options]\n"
        "  -p TEXT            prompt (chat-formatted unless --raw)\n"
        "  -f FILE            read the prompt from a file\n"
        "  -n N               tokens to generate (default 128)\n"
        "  --ctx N            context length (default 8192)\n"
        "  --ubatch N         prefill batch (default 1024)\n"
        "  --raw              do not apply the chat template\n"
        "  --think            leave thinking on (default: <think></think> prefilled)\n"
        "  --ram-gb G         pinned RAM arena (default: min(40, MemAvailable - 8))\n"
        "  --slots N          cap the VRAM expert slots\n"
        "  --vram-reserve G   VRAM left free after slots (default 0.6)\n"
        "  --profile FILE     seed tiers from a routing profile\n"
        "  --profile-out FILE write the routing profile on exit\n"
        "  --no-seed          start with empty tiers\n"
        "  --bench N          prefill N synthetic tokens, then decode -n tokens; print t/s\n"
        "  --tokens           print token ids of the prompt and exit\n"
        "  --top-logits K     print the top K logits after the prompt\n"
        "  --io-threads N     SSD reader threads (default 8)\n"
        "  -v                 verbose\n");
    exit(2);
}

static double mem_available_gb(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 16;
    char line[256];
    double kb = 0;
    while (fgets(line, sizeof line, f))
        if (sscanf(line, "MemAvailable: %lf kB", &kb) == 1) break;
    fclose(f);
    return kb / 1e6;
}

static int argmax(const float *v, int n) {
    int b = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[b]) b = i;
    return b;
}

static void print_stats(const m2_model *M, const char *what, const m2_tiers *before) {
    const m2_tiers *T = &M->tr;
    const uint64_t hv = T->hit_vram - before->hit_vram, hr = T->hit_ram - before->hit_ram,
                   ms = T->miss_ssd - before->miss_ssd;
    const uint64_t tot = hv + hr + ms;
    fprintf(stderr, "ds4-mimo2: %s experts: %llu lookups, VRAM %.1f%%, RAM %.1f%%, SSD %.1f%% (%.2f GB read)\n",
            what, (unsigned long long)tot, tot ? 100.0 * hv / tot : 0, tot ? 100.0 * hr / tot : 0,
            tot ? 100.0 * ms / tot : 0, (T->ssd_bytes - before->ssd_bytes) / 1e9);
}

int main(int argc, char **argv) {
    const char *model = NULL, *prompt = NULL, *pfile = NULL, *prof = NULL, *prof_out = NULL;
    int n_gen = 128, ctx = 8192, ubatch = 1024, raw = 0, think = 0, slots = -1, seed = 1, bench = 0;
    int show_tokens = 0, top_logits = 0, io_threads = 8, verbose = 0;
    double ram_gb = -1, reserve = 0.6;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
#define NEXT() (i + 1 < argc ? argv[++i] : (usage(), ""))
        if (!strcmp(a, "-m")) model = NEXT();
        else if (!strcmp(a, "-p")) prompt = NEXT();
        else if (!strcmp(a, "-f")) pfile = NEXT();
        else if (!strcmp(a, "-n")) n_gen = atoi(NEXT());
        else if (!strcmp(a, "--ctx")) ctx = atoi(NEXT());
        else if (!strcmp(a, "--ubatch")) ubatch = atoi(NEXT());
        else if (!strcmp(a, "--raw")) raw = 1;
        else if (!strcmp(a, "--think")) think = 1;
        else if (!strcmp(a, "--ram-gb")) ram_gb = atof(NEXT());
        else if (!strcmp(a, "--slots")) slots = atoi(NEXT());
        else if (!strcmp(a, "--vram-reserve")) reserve = atof(NEXT());
        else if (!strcmp(a, "--profile")) prof = NEXT();
        else if (!strcmp(a, "--profile-out")) prof_out = NEXT();
        else if (!strcmp(a, "--no-seed")) seed = 0;
        else if (!strcmp(a, "--bench")) bench = atoi(NEXT());
        else if (!strcmp(a, "--tokens")) show_tokens = 1;
        else if (!strcmp(a, "--top-logits")) top_logits = atoi(NEXT());
        else if (!strcmp(a, "--io-threads")) io_threads = atoi(NEXT());
        else if (!strcmp(a, "-v")) verbose = 1;
        else usage();
#undef NEXT
    }
    if (!model) usage();
    if (ubatch > 8192) ubatch = 8192;

    static m2_model M;
    M.ctx = ctx;
    M.ubatch = ubatch;
    M.verbose = verbose;
    double t0 = now_s();
    model_load(&M, model, show_tokens);

    char *text = NULL;
    if (pfile) {
        FILE *f = fopen(pfile, "rb");
        if (!f) die("cannot open %s", pfile);
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        text = xmalloc((size_t)sz + 1);
        if (fread(text, 1, (size_t)sz, f) != (size_t)sz) die("read %s", pfile);
        text[sz] = 0;
        fclose(f);
    } else if (prompt) {
        text = strdup(prompt);
    } else {
        text = strdup("Hello");
    }
    if (!raw && !bench) {
        const char *pre = "<|im_start|>user\n", *post = think ? "<|im_end|><|im_start|>assistant\n"
                                                              : "<|im_end|><|im_start|>assistant\n<think></think>";
        char *w = xmalloc(strlen(pre) + strlen(text) + strlen(post) + 1);
        sprintf(w, "%s%s%s", pre, text, post);
        free(text);
        text = w;
    }
    m2_toks pt = { 0 };
    tokenize(&M.vb, text, &pt);
    if (show_tokens) {
        for (int i = 0; i < pt.n; i++) printf("%d%s", pt.v[i], i + 1 < pt.n ? " " : "\n");
        return 0;
    }
    if (bench) {   /* synthetic prompt: the text repeated up to N tokens */
        m2_toks b = { 0 };
        while (b.n < bench) for (int i = 0; i < pt.n && b.n < bench; i++) toks_push(&b, pt.v[i]);
        free(pt.v);
        pt = b;
    }
    if (pt.n + n_gen > ctx) die("prompt (%d) + generation (%d) exceed --ctx %d", pt.n, n_gen, ctx);

    GCK(m2g_init());
    model_upload(&M);
    alloc_runtime(&M);
    fprintf(stderr, "ds4-mimo2: non-routed weights in VRAM in %.1fs\n", now_s() - t0);
    io_init(&M.io, io_threads < 1 ? 1 : io_threads > 16 ? 16 : io_threads);
    if (ram_gb < 0) {
        ram_gb = mem_available_gb() - 10;
        if (ram_gb > 36) ram_gb = 36;
        if (ram_gb < 4) ram_gb = 4;
    }
    tiers_init(&M, reserve, slots, ram_gb);
    if (seed) profile_seed(&M, prof);

    /* prefill */
    m2_tiers snap = M.tr;
    double tp = now_s();
    for (int i = 0; i < pt.n; i += M.ubatch) {
        const int nb = pt.n - i < M.ubatch ? pt.n - i : M.ubatch;
        forward(&M, pt.v + i, nb, i, i + nb == pt.n);
    }
    tp = now_s() - tp;
    fprintf(stderr, "ds4-mimo2: prefill %d tokens in %.2fs = %.2f t/s\n", pt.n, tp, pt.n / tp);
    print_stats(&M, "prefill", &snap);
    if (top_logits > 0) {
        float *lg = xmalloc((size_t)M.vb.n_vocab * sizeof(float));
        memcpy(lg, M.h_logits, (size_t)M.vb.n_vocab * sizeof(float));
        for (int k = 0; k < top_logits; k++) {
            const int b = argmax(lg, M.vb.n_vocab);
            char piece[256];
            const int pl = detok(&M.vb, b, piece, (int)sizeof piece - 1);
            piece[pl] = 0;
            printf("top %2d: %6d %10.4f '%s'\n", k, b, lg[b], piece);
            lg[b] = -INFINITY;
        }
        free(lg);
    }

    /* greedy decode */
    snap = M.tr;
    double td = now_s();
    int pos = pt.n, ngen = 0;
    int tok = argmax(M.h_logits, M.vb.n_vocab);
    for (; ngen < n_gen; ngen++) {
        if (!bench && (tok == M.vb.eos || tok == M.vb.im_end || tok == M.vb.eot)) break;
        if (!bench) {
            char piece[256];
            const int pl = detok(&M.vb, tok, piece, (int)sizeof piece);
            fwrite(piece, 1, (size_t)pl, stdout);
            fflush(stdout);
        }
        if (ngen + 1 == n_gen) { ngen++; break; }
        forward(&M, &tok, 1, pos++, true);
        tok = argmax(M.h_logits, M.vb.n_vocab);
    }
    td = now_s() - td;
    if (!bench) printf("\n");
    const int ndec = ngen > 1 ? ngen - 1 : 0;   /* forwards actually run */
    fprintf(stderr, "ds4-mimo2: decode %d tokens in %.2fs = %.2f t/s (SSD wait %.2fs)\n",
            ndec, td, ndec ? ndec / td : 0, M.tr.t_ssd - snap.t_ssd);
    print_stats(&M, "decode", &snap);
    if (prof_out) profile_save(&M, prof_out);
    return 0;
}
