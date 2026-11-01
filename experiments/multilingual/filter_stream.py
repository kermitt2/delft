#!/usr/bin/env python3
"""Keep from a text vector stream (stdin) only the words of the corpora, as `rank<space>word<space>vector`.

Nothing but the filtered lines reaches the disk. With --numberbatch the keys are
/c/<lang>/<word> and one output per language is written next to the given path.
"""
import sys

vocab_path, out_path = sys.argv[1], sys.argv[2]
numberbatch = "--numberbatch" in sys.argv
needed = set()
with open(vocab_path, encoding="utf-8", errors="surrogateescape") as f:
    for line in f:
        w = line.rstrip("\n").split("\t")[0]
        for form in (w, w.lower()):
            needed.add(form.encode("utf-8", "surrogateescape"))

outs = {}


def out(lang):
    if lang not in outs:
        outs[lang] = open(out_path.replace("LANG", lang), "wb")
    return outs[lang]


kept = total = 0
for rank, line in enumerate(sys.stdin.buffer):
    total += 1
    head, _, rest = line.partition(b" ")
    if numberbatch:
        parts = head.split(b"/")
        if len(parts) < 4:
            continue
        lang, word = parts[2].decode(), parts[3]
        if word in needed:
            out(lang).write(b"%d %s %s" % (rank, word, rest))
            kept += 1
    elif head in needed:
        out("x").write(b"%d %s %s" % (rank, head, rest))
        kept += 1
print(f"{out_path}: read {total} kept {kept}", flush=True)
