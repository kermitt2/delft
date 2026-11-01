#!/usr/bin/env python3
"""Pool the per-label counts of each stratum across folds into micro F1, per corpus and embedding."""

import collections
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
line = re.compile(r"^\s*(<\S+>)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$")
data = collections.defaultdict(lambda: collections.Counter())
per_fold = collections.defaultdict(dict)
seconds = collections.defaultdict(list)
for d in sorted((HERE / "results").glob("*/")):
    if not (d / "DONE").exists():
        continue
    corpus, emb, fold = re.match(r"(.+?)\.(.+)\.fold(\d)$", d.name).groups()
    seconds[(corpus, emb)].append(int((d / "train_seconds").read_text()))
    for e in d.glob("eval.*.log"):
        s = e.name[5:-4]
        tp = fp = fn = 0.0
        for l in e.read_text().splitlines():
            m = line.match(l)
            if m:
                p, r, n = float(m[2]), float(m[3]), int(m[5])
                t = r * n
                tp += t
                fn += n - t
                fp += (t / p - t) if p else 0
        data[(corpus, emb, s)].update(tp=tp, fp=fp, fn=fn)
        per_fold[(corpus, emb, s)][int(fold)] = 2 * tp / max(2 * tp + fp + fn, 1e-9)


def f1(c):
    return 2 * c["tp"] / max(2 * c["tp"] + c["fp"] + c["fn"], 1e-9)


out = {}
corpora = sorted({k[0] for k in data})
strata = ["all", "latin-en", "latin-non-en", "cjk", "cyrillic"]
for c in corpora:
    print(f"\n== {c}  (pooled micro F1, folds done in brackets; train minutes)")
    embs = sorted({k[1] for k in data if k[0] == c}, key=lambda e: -f1(data[(c, e, "latin-non-en")]))
    ss = [s for s in strata if any((c, e, s) in data for e in embs)]
    print("embedding".ljust(40), *[s.rjust(13) for s in ss], "min".rjust(6))
    for e in embs:
        row = [
            f"{100 * f1(data[(c, e, s)]):8.2f} [{len(per_fold[(c, e, s)])}]" if (c, e, s) in data else "-" for s in ss
        ]
        print(e.ljust(40), *[r.rjust(13) for r in row], f"{sum(seconds[(c, e)]) / len(seconds[(c, e)]) / 60:6.1f}")
        out[f"{c}|{e}"] = {
            s: {"f1": f1(data[(c, e, s)]), "folds": per_fold[(c, e, s)]} for s in ss if (c, e, s) in data
        }
(HERE / "summary.json").write_text(json.dumps(out, indent=1))
