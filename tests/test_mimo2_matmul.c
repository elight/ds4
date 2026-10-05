/* Checks m2g_matmul for every weight type against gguf-py's dequantizers.
 * Data comes from tests/mimo2_matmul_ref.py. Needs the GPU.
 *
 *   tests/test_mimo2_matmul DIR/mimo2_matmul.bin */
#include "ds4_mimo2.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s mimo2_matmul.bin\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 2; }
    int hdr[4];
    if (fread(hdr, sizeof hdr, 1, f) != 1) return 2;
    const int rows = hdr[0], cols = hdr[1], ntok = hdr[2], ntypes = hdr[3];
    float *x = malloc((size_t)ntok * cols * sizeof(float));
    float *y = malloc((size_t)ntok * rows * sizeof(float));
    float *ref = malloc((size_t)ntok * rows * sizeof(float));
    if (fread(x, sizeof(float), (size_t)ntok * cols, f) != (size_t)ntok * cols) return 2;
    if (m2g_init()) return 1;
    float *dx = m2g_alloc((size_t)ntok * cols * sizeof(float));
    float *dy = m2g_alloc((size_t)ntok * rows * sizeof(float));
    m2g_upload(dx, x, (size_t)ntok * cols * sizeof(float));
    int fails = 0;
    for (int i = 0; i < ntypes; i++) {
        int type;
        long long nbytes;
        if (fread(&type, 4, 1, f) != 1 || fread(&nbytes, 8, 1, f) != 1) return 2;
        void *w = malloc((size_t)nbytes);
        if (fread(w, 1, (size_t)nbytes, f) != (size_t)nbytes) return 2;
        if (fread(ref, sizeof(float), (size_t)ntok * rows, f) != (size_t)ntok * rows) return 2;
        void *dw = m2g_alloc((size_t)nbytes);
        m2g_upload(dw, w, (size_t)nbytes);
        m2g_matmul(type, dw, rows, cols, dx, cols, dy, rows, ntok, 0);
        m2g_download(y, dy, (size_t)ntok * rows * sizeof(float));
        double maxerr = 0, maxref = 0;
        for (int k = 0; k < ntok * rows; k++) {
            maxerr = fmax(maxerr, fabs((double)y[k] - ref[k]));
            maxref = fmax(maxref, fabs((double)ref[k]));
        }
        const double rel = maxerr / (maxref > 0 ? maxref : 1);
        const int ok = rel < 1e-4;
        fails += !ok;
        printf("type %2d: max |err| %.3g of max |y| %.3g (rel %.2g) %s\n", type, maxerr, maxref, rel, ok ? "ok" : "FAIL");
        free(w);
    }
    printf(fails ? "FAILED\n" : "all ok\n");
    return fails != 0;
}
