#!/usr/bin/env python3
"""Token-weighted coverage per corpus and language stratum.

Exact lookup, as DeLFT does it: a miss here is a zero vector at training and tagging.
For the model-backed candidates a token is only "missed" when the tokenizer gives
nothing but the unknown token, and it still gets the unknown vector.
"""

import json
import sys
from pathlib import Path

import lmdb
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
CORPORA = ["citation", "affiliation-address", "header"]
STRATA = ["latin-en", "latin-non-en", "cjk", "cyrillic"]


def vec_keys(path):
    with open(path, "rb") as f:
        next(f)
        return {line.split(b" ", 1)[0].decode("utf-8") for line in f}


def lmdb_has(path):
    txn = lmdb.open(str(path), readonly=True, lock=False, max_readers=8).begin()
    return lambda w: txn.get(w.encode("utf-8", "replace")[:511] or b"\0") is not None


def tokenizer_has(name):
    from huggingface_hub import hf_hub_download

    tok = Tokenizer.from_pretrained(name) if False else Tokenizer.from_file(hf_hub_download(*name))
    tok.no_padding()
    unk = tok.token_to_id("[UNK]")

    def has(w):
        ids = [i for i in tok.encode(w, add_special_tokens=False).ids]
        return bool(ids) and any(i != unk for i in ids)

    return has


def main():
    candidates = {"glove-840B": lmdb_has(sys.argv[1])}
    for p in sorted((HERE / "tables").glob("*.vec")):
        keys = vec_keys(p)
        candidates[p.stem] = keys.__contains__
    candidates["potion-multilingual-128M"] = tokenizer_has(("minishlab/potion-multilingual-128M", "tokenizer.json"))
    candidates["static-similarity-mrl-multilingual"] = tokenizer_has(
        ("sentence-transformers/static-similarity-mrl-multilingual-v1", "0_StaticEmbedding/tokenizer.json")
    )
    result = {}
    for corpus in CORPORA:
        strata = json.loads((HERE / f"data/{corpus}/strata_vocab.json").read_text())
        for s in STRATA:
            if s not in strata or sum(strata[s].values()) < 150:
                continue
            total = sum(strata[s].values())
            # tokens with at least one letter: punctuation and numbers say nothing here
            words = {w: c for w, c in strata[s].items() if any(ch.isalpha() for ch in w)}
            wtotal = sum(words.values()) or 1
            for name, has in candidates.items():
                hit = sum(c for w, c in words.items() if has(w))
                result.setdefault(f"{corpus}/{s} ({wtotal}/{total} tok)", {})[name] = round(100 * hit / wtotal, 1)
    (HERE / "coverage.json").write_text(json.dumps(result, indent=1))
    names = list(candidates)
    print("stratum".ljust(48), *[n[:14].rjust(15) for n in names])
    for k, row in result.items():
        print(k.ljust(48), *[f"{row[n]:15.1f}" for n in names])


if __name__ == "__main__":
    main()
