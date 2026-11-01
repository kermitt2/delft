#!/usr/bin/env python3
"""Lookup table for BPEmb multi: each corpus token is the mean of its subword vectors.

BPEmb is trained on lowercased text with digits replaced by zero, so the token is
lowercased first; the digits are already normalised in the vocabulary.

argv: <word2vec .bin> <sentencepiece .model>
"""

import sys
from pathlib import Path

import numpy as np
import sentencepiece

HERE = Path(__file__).resolve().parent


def read_w2v_bin(path):
    with open(path, "rb") as f:
        count, dim = map(int, f.readline().split())
        words, matrix = {}, np.zeros((count, dim), dtype=np.float32)
        for i in range(count):
            w = bytearray()
            while (c := f.read(1)) != b" ":
                if c != b"\n":
                    w += c
            words[w.decode("utf-8", "replace")] = i
            matrix[i] = np.frombuffer(f.read(4 * dim), dtype=np.float32)
    return words, matrix


def main():
    words, matrix = read_w2v_bin(sys.argv[1])
    sp = sentencepiece.SentencePieceProcessor()
    sp.load(sys.argv[2])
    rows = []
    with open(HERE / "data/vocab.txt", encoding="utf-8", errors="surrogateescape") as f:
        for line in f:
            token = line.rstrip("\n").split("\t")[0]
            try:
                key = token.encode("utf-8")
            except UnicodeEncodeError:
                continue
            ids = [words[p] for p in sp.encode_as_pieces(token.lower()) if p in words]
            if not ids or b" " in key or not key:
                continue
            vec = matrix[ids].mean(axis=0)
            rows.append(key + b" " + " ".join(f"{x:.5f}" for x in vec).encode())
    (HERE / "tables").mkdir(exist_ok=True)
    with open(HERE / "tables/bpemb-multi.vec", "wb") as f:
        f.write(b"%d %d\n" % (len(rows), matrix.shape[1]))
        f.write(b"\n".join(rows) + b"\n")
    print("bpemb-multi", len(rows), "keys")


if __name__ == "__main__":
    main()
