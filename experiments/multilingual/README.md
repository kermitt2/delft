# Multilingual static embeddings as a GloVe replacement

Benchmark run on 2026-11-01 on the GROBID `affiliation-address` and `citation` training
corpora (`data/sequenceLabelling/grobid`), architecture `BidLSTM_CRF_FEATURES`, CPU,
3 folds, seeds 1..3. Every non-English sequence is tested exactly once over the 3 folds,
English sequences 30% in all. Strata are assigned by script of the sequence
(`latin-en`, `latin-non-en`, `cjk`, `cyrillic`), see `prepare.py`.

## Result: BPEmb multilingual (vs1000000, d300) replaces GloVe best

Pooled micro F1 over the 3 folds (folds where the candidate beats glove-840B in brackets).
Per-fold values are in `summary.json`, the evaluation reports in `results/`.

| embedding | affil. non-en | affil. en | citation non-en latin | citation cjk | citation en |
|---|---|---|---|---|---|
| glove-840B | 80.88 | 87.09 | 79.99 | 92.15 | 94.27 |
| **bpemb-multi** | **82.35** (3/3) | **87.65** (3/3) | 80.79 (3/3) | **94.51** (3/3) | 94.10 (1/3) |
| ft-aligned (fastText aligned, 11 languages merged) | 81.85 (2/3) | 87.04 (2/3) | **80.83** (2/3) | 92.45 (1/3) | 94.10 (0/3) |
| cc-en (fastText crawl en) | 80.57 (2/3) | 87.37 (2/3) | 79.36 (1/3) | 93.02 (3/3) | 94.14 (1/3) |
| cc-merged (fastText crawl, 11 languages) | 79.38 (0/3) | 87.15 (2/3) | 79.61 (2/3) | 92.79 (2/3) | 94.29 (1/3) |
| numberbatch 19.08 | 80.86 (2/3) | 86.07 (0/3) | not run | | |
| potion-multilingual-128M | 79.51 (0/3) | 86.14 (1/3) | 79.63 (1/3) | 93.29 (2/3) | 93.87 (0/3) |
| static-similarity-mrl-multilingual | 78.75 (0/3) | 85.88 (1/3) | 79.38 (1/3) | 93.41 (3/3) | 93.76 (0/3) |
| static-similarity-mrl-multilingual-256 | 80.15 (1/3) | 86.28 (0/3) | 79.37 (2/3) | 93.49 (3/3) | 93.92 (0/3) |
| amgix/static-retrieval-multilingual-69m-v1 | 76.83 (0/3) | 83.81 (0/3) | not run | | |
| bpemb-multi+ft-aligned (stacked) | 80.64 (1/3) | 87.04 (2/3) | not run | | |
| none | 78.13 (0/3) | 85.30 (0/3) | 78.31 (1/3) | 92.54 (2/3) | 93.76 (0/3) |

Cyrillic has about 60 test tokens per fold and is not reported. Samples are small:
163 non-English affiliation sequences, 280 non-English Latin and 338 CJK citations.

Token coverage (`coverage.json`): glove-840B covers 8.8% of CJK citation tokens and
about 86% of non-English Latin tokens; the sub-word models cover 100%.

## BPEmb

Source: https://bpemb.h-its.org/multi/multi/ (`multi.wiki.bpe.vs1000000.model`,
`multi.wiki.bpe.vs1000000.d300.w2v.bin`), Heinzerling & Strube 2018. The vectors are
lowercased with digits mapped to 0; the repository code is MIT, the vector files state
no licence.

`convert_bpemb.py` packages it as a static embedding directory (`tokenizer.json` +
`model.safetensors`, normalizer NFKC+lowercase, Metaspace pre-tokenizer) that DeLFT loads
through the `static-transformer` path with no code change. The runs used an equivalent
word table (`build_bpemb.py`); the only differences are words containing characters
outside the BPEmb vocabulary, which the table dropped and the package maps to `<unk>`.

## Reproducing

1. `prepare.py` builds the folds and `data/vocab.txt` from the GROBID corpora.
2. `fetch_all.sh` streams the fastText and Numberbatch files, keeping only corpus words
   (`filter_stream.py`); `merge.py` merges the languages into `tables/`.
3. `build_bpemb.py` / `convert_bpemb.py` build the BPEmb table and the packaged model.
4. Add `registry-entries.json` to `delft/resources-registry.json`.
5. `xargs -P 4 -L 1 ./job.sh < jobs.txt`, then `summarize.py`.

## Model card

`model-card.md` is the README of the packaged model (`models/bpemb-multi-1M/README.md`),
with the attribution to the original authors, for the Hugging Face upload.
