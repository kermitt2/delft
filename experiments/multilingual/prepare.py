#!/usr/bin/env python3
"""Fixed train/test split, language strata and vocabulary for the embedding comparison.

The split hashes the token sequence, so duplicates of a sequence always land on the
same side and the test set cannot leak through them.

Strata are per sequence and heuristic: a script bucket when non-Latin letters dominate,
otherwise a stopword/diacritic vote between English and other Latin-script languages.
"""

import collections
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GROBID = ROOT / "data/sequenceLabelling/grobid"
CORPORA = {
    "citation": GROBID / "citation/citation-251225.train",
    "affiliation-address": GROBID / "affiliation-address/affiliation-address-261011.train",
    "header": GROBID / "header/header-261010.train",
}
FOLDS = 3  # every non-English sequence is tested once, in one of the folds
EN_TEST_BUCKETS = 30  # English: one in ten per fold, so 30% in all over the folds

EN = set(
    "the of and in for on with from by to at an is are was were this that these those as using study "
    "analysis based between effect effects patients".split()
)
OTHER = (
    set(
        # fr
        "le la les des du une et pour dans sur avec par au aux est sont cette ces étude université "
        # de
        "der die das und für von mit im ein eine zur zum bei über auf nach universität "
        # es / pt / it
        "el los las del y en para con por una un de que estudio universidad "
        "os as dos das da do em uma um com universidade "
        "il di della delle dei per con nel nella università studio".split()
    )
    - EN
    - {"de", "en"}
)  # both are too common in names and affiliations to vote


def script_of(ch):
    cp = ord(ch)
    if 0x1100 <= cp <= 0x11FF or 0xAC00 <= cp <= 0xD7AF or 0x3130 <= cp <= 0x318F:
        return "hangul"
    if 0x3040 <= cp <= 0x30FF:
        return "kana"
    if 0x3400 <= cp <= 0x9FFF or 0xF900 <= cp <= 0xFAFF:
        return "han"
    if 0x0400 <= cp <= 0x052F:
        return "cyrillic"
    if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
        return "arabic"
    if 0x0370 <= cp <= 0x03FF or 0x1F00 <= cp <= 0x1FFF:
        return "greek"
    return "latin" if ch.isalpha() else None


def stratum(tokens):
    scripts = collections.Counter(s for t in tokens for s in map(script_of, t) if s)
    letters = sum(scripts.values())
    if not letters:
        return "other"
    cjk = scripts["han"] + scripts["kana"] + scripts["hangul"]
    if cjk / letters >= 0.2:
        return "cjk"
    if scripts["cyrillic"] / letters >= 0.2:
        return "cyrillic"
    if scripts["arabic"] / letters >= 0.2:
        return "arabic"
    lower = [t.lower() for t in tokens]
    en = sum(t in EN for t in lower)
    other = sum(t in OTHER for t in lower)
    accented = sum(unicodedata.normalize("NFD", ch) != ch and script_of(ch) == "latin" for t in tokens for ch in t)
    if other >= 2 and other > en:
        return "latin-non-en"
    if accented >= 3 and accented / letters >= 0.03 and en <= other:
        return "latin-non-en"
    return "latin-en"


def sequences(path):
    block = []
    with open(path, encoding="utf-8", errors="surrogateescape") as f:
        for line in f:
            if line.strip():
                block.append(line)
            elif block:
                yield block
                block = []
    if block:
        yield block


_DIGITS = re.compile(r"[0-9０１２３４５６７８９]")  # delft.utilities.preprocess.normalize_num


def main():
    out = HERE / "data"
    out.mkdir(exist_ok=True)
    vocab = collections.Counter()
    report = {}
    for name, path in CORPORA.items():
        d = out / name
        d.mkdir(exist_ok=True)
        blocks = []
        for block in sequences(path):
            tokens = [line.split()[0] for line in block]
            h = int(hashlib.sha1("\x00".join(tokens).encode("utf-8", "surrogateescape")).hexdigest(), 16)
            blocks.append((block, stratum(tokens), h, tokens))
        strata_tokens = collections.defaultdict(collections.Counter)
        for block, s, h, tokens in blocks:
            for t in tokens:
                n = _DIGITS.sub("0", t)
                vocab[n] += 1
                strata_tokens[s][n] += 1
        counts = collections.Counter(b[1] for b in blocks)
        report[name] = {
            "sequences": dict(sorted(counts.items())),
            "tokens": {s: sum(c.values()) for s, c in sorted(strata_tokens.items())},
        }
        for fold in range(FOLDS):
            fd = d / f"fold{fold}"
            fd.mkdir(exist_ok=True)
            enc = dict(encoding="utf-8", errors="surrogateescape")
            tests = {}
            with open(fd / "train.txt", "w", **enc) as train:
                for block, s, h, _ in blocks:
                    is_test = (
                        (h % EN_TEST_BUCKETS) in (fold, fold + FOLDS, fold + 2 * FOLDS)
                        if s == "latin-en"
                        else h % FOLDS == fold
                    )
                    if not is_test:
                        train.writelines(block)
                        train.write("\n")
                        continue
                    for target in (s, "all"):
                        if target not in tests:
                            tests[target] = open(fd / f"test.{target}.txt", "w", **enc)
                        tests[target].writelines(block)
                        tests[target].write("\n")
            for f in tests.values():
                f.close()
        with open(d / "strata_vocab.json", "w", encoding="utf-8") as f:
            json.dump(strata_tokens, f, ensure_ascii=True)
    with open(out / "vocab.txt", "w", encoding="utf-8", errors="surrogateescape") as f:
        for w, c in vocab.most_common():
            f.write(f"{w}\t{c}\n")
    (out / "split_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print("vocabulary", len(vocab), file=sys.stderr)


if __name__ == "__main__":
    main()
