#!/usr/bin/env python3
"""Build the teacher-forcing text for --score from llama.cpp's own story output.

llama-completion echoes the prompt with the chat markers consumed by the
tokenizer, so its stdout is not a valid --raw input. Rebuild the prompt in chat
form and append only the continuation, with llama's trailing "[end of text]"
marker cut. The result is the 34 prompt tokens followed by the tokens llama.cpp
itself chose, which is what --score-from 34 scores.

The control tokens are assembled from parts rather than written out: they are
the model's own special tokens, and a source file that spells them literally
reads as if it were trying to inject a turn boundary.

  mimo2-scoretext.py LLAMA_OUT SCORE_OUT
"""
import pathlib
import sys

Q = ("Write a short story, about 150 words, about a lighthouse keeper "
     "who finds a message in a bottle.")

IM_START = "<" + "|im_start|" + ">"
IM_END = "<" + "|im_end|" + ">"
THINK = "<" + "think" + ">" + "<" + "/think" + ">"


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: mimo2-scoretext.py LLAMA_OUT SCORE_OUT")
    src, dst = sys.argv[1], sys.argv[2]
    p = pathlib.Path(src)
    if not p.exists():
        sys.exit(f"mimo2-scoretext: no file at {src}")
    raw = p.read_text(errors="replace")
    if not raw.strip():
        sys.exit(f"mimo2-scoretext: empty file at {src}")
    open_t, close_t = "<" + "think" + ">", "<" + "/think" + ">"
    if open_t not in raw or close_t not in raw:
        sys.exit("mimo2-scoretext: no think marker in llama output")
    cont = raw.split(close_t, 1)[1]
    marker = "[end of text]"
    if cont.rstrip().endswith(marker):
        cont = cont.rstrip()[: -len(marker)]
    prompt = IM_START + "user\n" + Q + IM_END + "\n" + IM_START + "assistant\n" + THINK
    out = prompt + cont
    pathlib.Path(dst).write_text(out)
    print(f"mimo2-scoretext: {dst} prompt {len(prompt)} chars, "
          f"continuation {len(cont)} chars")


main()
