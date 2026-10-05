#!/usr/bin/env python3
"""Prefill and decode rates from a running Strata server, on the ds4-bench story.

Each frontier gets the story cut to about that many tokens, behind a distinct
first line so Strata's prompt cache cannot reuse an earlier, shorter run.
prefill = prompt_tokens / time to first token; decode = completion tokens over
the rest (reasoning tokens count: they are produced at the decode rate).
"""
import argparse
import json
import time
import urllib.request

CHARS_PER_TOKEN = 4.49  # tests/long_context_story_prompt.txt: 140293 chars, 31199 tokens


def run(base, prompt, max_tokens):
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0, ttft, chunks, usage = time.time(), None, 0, None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            j = json.loads(line[5:])
            usage = j.get("usage") or usage
            for c in j.get("choices") or []:
                d = c.get("delta") or {}
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    ttft = ttft or time.time() - t0
                    chunks += 1
    el = time.time() - t0
    pt = (usage or {}).get("prompt_tokens")
    ct = (usage or {}).get("completion_tokens") or chunks
    return {"prompt_tokens": pt, "completion_tokens": ct, "ttft_s": round(ttft, 3),
            "prefill_tok_s": round(pt / ttft, 1) if pt else None,
            "decode_tok_s": round((ct - 1) / (el - ttft), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--story", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctx", type=int, nargs="+", default=[8192, 32768])
    ap.add_argument("--gen", type=int, default=128)
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}/v1"
    story = open(a.story).read()
    res = {"warmup": run(base, "Say hello.", 16)}
    for ctx in a.ctx:
        n = int((ctx - a.gen - 64) * CHARS_PER_TOKEN)
        prompt = f"Run {ctx} at {time.time()}.\n\n" + story[:n]
        res[str(ctx)] = r = run(base, prompt, a.gen)
        print(ctx, json.dumps(r), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
