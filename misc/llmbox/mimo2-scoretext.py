#!/usr/bin/env python3
"""Build the teacher-forcing text for --score from llama.cpp's own story output.

llama-completion echoes the prompt with the chat markers consumed by the
tokenizer, so its stdout is not a valid --raw input. Rebuild the prompt in chat
form and append only the continuation, with llama's trailing "[end of text]"
marker cut. The result is the 34 prompt tokens followed by the tokens llama.cpp
itself chose, which is what --score-from 34 scores.

  mimo2-scoretext.py LLAMA_OUT SCORE_OUT
"""
import pathlib
import sys

Q = ("Write a short story, about 150 words, about a lighthouse keeper "
     "who finds a message in a bottle.")

def main():
    if len(sys.argv) < 3:
        sys.exit("usage: mimo2-scoretext.py LLAMA_OUT SCORE_OUT")
    src, dst = sys.argv[1], sys.argv[2]
    p = pathlib.Path(src)
    if not p.exists() or not p.read_text(errors="replace").strip():
        sys.exit(f"mimo2-scoretext: no text at {src}")
    raw = p.read_text(errors="replace")
    if "<think></think>" not in raw:
        sys.exit("mimo2-scoretext: no <think></think> marker in llama output")
    cont = raw.split("<think></think>", 1)[1]
    marker = "[end of text]"
    if cont.rstrip().endswith(marker):
        cont = cont.rstrip()[: -len(marker)]
    prompt = "<|im_start|>user\n" + Q + "