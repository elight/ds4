#!/usr/bin/env python3
"""Turn one mimo-bench step's captured output into one JSON record per run.

A step script prints a "=== <name>" header before each run and echoes the
engine's metric lines under it, so the step's stdout and stderr together read
as blocks. Splitting on those headers keeps a sweep's runs as separate records
instead of one record where the first cpu-experts setting wins.

The engine prints metrics to its own stderr; the step scripts grep them onto
their stdout. Both streams are searched, because which one a line reaches
depends on the script, not on the engine.
"""
import json
import re
import sys

DS4 = {
    "prefill_tokens": r"prefill (\d+) tokens in [\d.]+s = [\d.]+ t/s",
    "prefill_tps": r"prefill \d+ tokens in [\d.]+s = ([\d.]+) t/s",
    "prefill_lookups": r"prefill experts: (\d+) lookups",
    "prefill_tiers": r"prefill experts: \d+ lookups, VRAM ([\d.]+)%, RAM ([\d.]+)%, SSD ([\d.]+)%",
    "decode_tokens": r"decode (\d+) tokens in [\d.]+s = [\d.]+ t/s",
    "decode_tps": r"decode \d+ tokens in [\d.]+s = ([\d.]+) t/s",
    "ssd_wait_s": r"decode \d+ tokens in [\d.]+s = [\d.]+ t/s \(SSD wait ([\d.]+)s\)",
    "decode_lookups": r"decode experts: (\d+) lookups",
    "decode_tiers": r"decode experts: \d+ lookups, VRAM ([\d.]+)%, RAM ([\d.]+)%, SSD ([\d.]+)% \(([\d.]+) GB read\)",
    "cpu_share": r"decode RAM->CPU: (\d+) experts \(([\d.]+)% of lookups\), host idle waiting on CPU ([\d.]+)s",
    "mtp": r"MTP (\d+) verify passes, (\d+) drafted, (\d+) accepted \(([\d.]+)%\), ([\d.]+) tokens/pass",
    "score": r"score: (\d+) tokens, ppl ([\d.]+), top-1 agreement ([\d.]+)%",
    "tiers": r"tiers: (\d+) VRAM slots \(([\d.]+) GB\), (\d+) RAM entries \(([\d.]+) GB",
}
LLAMA = {
    "llama_prefill": r"prompt eval time =\s+[\d.]+ ms /\s+(\d+) tokens \(.*?([\d.]+) tokens per second\)",
    "llama_decode": r"common_perf_print:\s+eval time =\s+[\d.]+ ms /\s+(\d+) runs.*?([\d.]+) tokens per second",
}


def blocks(text):
    """Split on '=== <name>' headers; text before the first header is a preamble.
    Guarded on the header having been seen, not on the buffer being non-empty:
    the first line under a header is the one that carries the metrics."""
    out, name, buf = [], None, []
    for line in text.splitlines():
        m = re.match(r"^=== (.*)$", line)
        if m:
            if name is not None:
                out.append((name, "\n".join(buf)))
            name, buf = m.group(1).strip(), []
        elif name is not None:
            buf.append(line)
    if name is not None:
        out.append((name, "\n".join(buf)))
    return out


def grab(rx, text):
    m = re.search(rx, text)
    return [float(x) if "." in x else int(x) for x in m.groups()] if m else None


def main():
    text = "".join(open(p, errors="replace").read() + "\n" for p in sys.argv[1:3])
    recs = []
    for name, body in blocks(text):
        rec = {"run": name}
        for k, rx in DS4.items():
            g = grab(rx, body)
            if g:
                rec[k] = g
        for k, rx in LLAMA.items():
            g = grab(rx, body)
            if g:
                rec[k] = g
        if "IDENTICAL to plain greedy" in body:
            rec["mtp_output_identical"] = True
        m = re.search(r"DIFFERS from plain at char (\d+)", body)
        if m:
            rec["mtp_output_identical"] = False
            rec["mtp_differs_at_char"] = int(m.group(1))
        m = re.search(r"(\d+) of (\d+) llama words match", body)
        if m:
            rec["word_match"] = [int(m.group(1)), int(m.group(2))]
        if len(rec) > 1:
            recs.append(rec)
    for rec in recs:
        print(json.dumps(rec, sort_keys=True))


main()
