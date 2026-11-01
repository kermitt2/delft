#!/usr/bin/env python3
"""Merge the filtered language parts into one lookup table per candidate.

Keys are the corpus tokens as DeLFT looks them up (digits normalised, case kept), so the
tables need no change to the library: a token found only in lowercase gets the vector of
its lowercase form, which is what a lowercasing option would do for these vectors.

Rules:
  aligned   languages share one space: lowest frequency rank wins among the languages,
            an exact match is preferred over the lowercase form
  cc        languages do NOT share a space: a fixed priority order, English first, and a
            later language only fills what the earlier ones miss
  cc-en     the English table alone, as the control for what the merge adds
  numberbatch  as aligned, over all the languages of the file
"""

import glob
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LANGS = "en fr de es it pt nl ru zh ja ko ar".split()


def vocab():
    with open(HERE / "data/vocab.txt", encoding="utf-8", errors="surrogateescape") as f:
        return [line.rstrip("\n").split("\t")[0] for line in f]


def load(path):
    table = {}
    with open(path, "rb") as f:
        for line in f:
            rank, word, rest = line.split(b" ", 2)
            rest = rest.rstrip(b" \n")
            if word not in table:
                table[word] = (int(rank), rest)
    return table


def by_rank(tables, key):
    hits = [t[key] for t in tables if key in t]
    return min(hits, key=lambda h: h[0])[1] if hits else None


def by_priority(tables, key, lower):
    for t in tables:
        if key in t:
            return t[key][1]
        if lower in t:
            return t[lower][1]
    return None


def write(name, rows, dim):
    (HERE / "tables").mkdir(exist_ok=True)
    tmp = HERE / f"tables/{name}.vec.tmp"
    with open(tmp, "wb") as f:
        f.write(b"%d %d\n" % (len(rows), dim))
        for key, vec in rows:
            f.write(key + b" " + vec + b"\n")
    os.replace(tmp, HERE / f"tables/{name}.vec")
    print(name, len(rows), "keys", flush=True)


def tokens():
    for w in vocab():
        try:
            key = w.encode("utf-8")
        except UnicodeEncodeError:
            continue  # the compile decodes strictly, and such a token cannot be looked up
        if b" " in key or not key:
            continue
        yield key, w.lower().encode("utf-8")


def build(name, tables, pick):
    rows = []
    for key, lower in tokens():
        vec = pick(tables, key, lower)
        if vec is not None:
            rows.append((key, vec))
    write(name, rows, 300)


def main():
    p = HERE / "parts"
    what = sys.argv[1:] or ["aligned", "cc", "numberbatch"]
    if "aligned" in what:
        tables = [load(p / f"aligned_{l}.vec") for l in LANGS if (p / f"aligned_{l}.vec").exists()]
        build("ft-aligned", tables, lambda t, k, lo: by_rank(t, k) or by_rank(t, lo))
    if "cc" in what:
        tables = [load(p / f"cc_{l}.vec") for l in LANGS if (p / f"cc_{l}.vec").exists()]
        build("cc-merged", tables, by_priority)
        build("cc-en", tables[:1], by_priority)
    if "numberbatch" in what:
        paths = sorted(glob.glob(str(p / "numberbatch_*.vec")))
        en = [x for x in paths if x.endswith("_en.vec")]
        tables = [load(x) for x in en + [x for x in paths if x not in en]]
        print("numberbatch languages", len(tables))
        build("numberbatch", tables, lambda t, k, lo: by_rank(t, lo))


if __name__ == "__main__":
    main()
