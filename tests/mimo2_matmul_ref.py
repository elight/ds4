"""Reference data for tests/test_mimo2_matmul.c.

Writes random weights in each GGUF type ds4-mimo2 reads, an activation batch,
and the expected product computed with gguf-py's own dequantizers. Run with a
python that has numpy and gguf-py:

    python tests/mimo2_matmul_ref.py OUT_DIR
"""
import struct
import sys

import numpy as np
from gguf import GGMLQuantizationType as Q
from gguf.quants import dequantize

ROWS, COLS, NTOK = 96, 512, 7
rng = np.random.default_rng(1234)


def half_bytes(n, lo, hi):
    return rng.uniform(lo, hi, n).astype(np.float16).view(np.uint8).reshape(n, 2)


def make(t):
    nb32, nb256 = ROWS * COLS // 32, ROWS * COLS // 256
    if t == Q.F32:
        return rng.standard_normal(ROWS * COLS).astype(np.float32).view(np.uint8)
    if t == Q.Q8_0:
        b = np.zeros((nb32, 34), np.uint8)
        b[:, 0:2] = half_bytes(nb32, 0.001, 0.02)
        b[:, 2:] = rng.integers(0, 256, (nb32, 32), dtype=np.uint8)
        return b.ravel()
    if t == Q.MXFP4:
        b = np.zeros((nb32, 17), np.uint8)
        b[:, 0] = rng.integers(118, 130, nb32, dtype=np.uint8)
        b[:, 1:] = rng.integers(0, 256, (nb32, 16), dtype=np.uint8)
        return b.ravel()
    if t == Q.Q2_K:
        b = rng.integers(0, 256, (nb256, 84), dtype=np.uint8)
        b[:, 80:82] = half_bytes(nb256, 0.001, 0.02)
        b[:, 82:84] = half_bytes(nb256, 0.0, 0.01)
        return b.ravel()
    if t == Q.Q6_K:
        b = rng.integers(0, 256, (nb256, 210), dtype=np.uint8)
        b[:, 208:210] = half_bytes(nb256, 0.0005, 0.005)
        return b.ravel()
    raise ValueError(t)


def main():
    out = sys.argv[1]
    x = rng.standard_normal((NTOK, COLS)).astype(np.float32)
    with open(f"{out}/mimo2_matmul.bin", "wb") as f:
        types = [Q.F32, Q.Q8_0, Q.Q2_K, Q.Q6_K, Q.MXFP4]
        f.write(struct.pack("<4i", ROWS, COLS, NTOK, len(types)))
        f.write(x.tobytes())
        for t in types:
            w = make(t)
            if t == Q.F32:
                wf = w.view(np.float32).reshape(ROWS, COLS)
            else:
                wf = dequantize(w.reshape(ROWS, -1), t).reshape(ROWS, COLS).astype(np.float64)
            y = (x.astype(np.float64) @ wf.T).astype(np.float32)
            f.write(struct.pack("<iq", int(t), w.nbytes))
            f.write(w.tobytes())
            f.write(y.tobytes())


if __name__ == "__main__":
    main()
