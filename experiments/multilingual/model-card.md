---
# license: not yet confirmed by the original authors, see the Licence section below
language:
  - multilingual
tags:
  - feature-extraction
  - static-embeddings
  - word-embeddings
  - subword-embeddings
  - bpe
  - sentencepiece
  - glove
  - bpemb
  - delft
pipeline_tag: feature-extraction
---

# BPEmb multilingual (1M vocabulary, 300d), repackaged for DeLFT

This is a **repackaging, not a new model**. The sub-word vocabulary and all the vectors
are the multilingual BPEmb embeddings (*MultiBPEmb*) created and published by
**Benjamin Heinzerling and Michael Strube** (Heidelberg Institute for Theoretical
Studies). Nothing was retrained or modified: the two original files were converted,
without any change to their values, into the `tokenizer.json` + `model.safetensors`
layout that [DeLFT](https://github.com/kermitt2/delft) loads as a static embedding
model. All credit for the embeddings goes to the original authors; please cite their
papers (below), not this repository.

## Original work

| | |
|---|---|
| Authors | Benjamin Heinzerling and Michael Strube |
| Project page | https://bpemb.h-its.org/ |
| Code | https://github.com/bheinzerling/bpemb |
| Papers | Heinzerling & Strube, LREC 2018 (BPEmb); Heinzerling & Strube, ACL 2019 (MultiBPEmb) |
| Files used | https://bpemb.h-its.org/multi/multi/multi.wiki.bpe.vs1000000.model (SentencePiece BPE model, 20,636,145 bytes, sha256 `7f4c3730bb7ded03…`) and https://bpemb.h-its.org/multi/multi/multi.wiki.bpe.vs1000000.d300.w2v.bin.tar.gz (vectors, word2vec binary format, 1,212,398,411 bytes unpacked) |

How the authors built the multilingual model, in their own words (ACL 2019, footnote 10):

> we extract up to 500k randomly selected paragraphs from articles in each Wikipedia
> edition, yielding 16GB of text in 265 languages. Then, we train BPE models with
> vocabulary sizes 100k, 320k, and 1000k using SentencePiece (Kudo and Richardson,
> 2018), and finally train subword embeddings using GloVe.

Preprocessing of the training text (LREC 2018, footnote 2): the authors "lowercase all
characters where applicable and map all digits to zero". The vectors are therefore
**case-insensitive** and expect digits normalised to `0`.

This repackaging uses the largest of the three vocabularies (1,000,000 pieces) and the
300-dimensional vectors.

## What is in this repository

| File | Content |
|---|---|
| `model.safetensors` | tensor `embeddings`, shape `[1000000, 300]`, float32: the original GloVe vectors, one row per SentencePiece id, copied unchanged |
| `tokenizer.json` | a Hugging Face `tokenizers` BPE tokenizer equivalent to the original SentencePiece model: same 1,000,000 pieces, merges derived from the SentencePiece piece scores, `<unk>` as unknown token, normalizer NFKC + lowercase, Metaspace pre-tokenizer (`▁`, prepended to every word) |
| `config.json` | `{"normalize": false}`: vectors are used as they are, not L2-normalised |

A word's vector is the mean of the vectors of its pieces, computed by the loader. Words
are never out of vocabulary: any string gets a vector from its pieces, with `<unk>` for
characters outside the vocabulary.

The conversion script is `experiments/multilingual/convert_bpemb.py` on the
`experiment/multilingual-embeddings` branch of DeLFT (tokenizers 0.23.2, sentencepiece
0.2.1, transformers 5.17.0, safetensors 0.8.0). Fidelity to the original was checked on
the 66,241 distinct words of the GROBID training corpora: the tokenizer produces the
same segmentation as the original SentencePiece model for 66,240 of them. The single
exception (`Java™`) comes from the NFKC normalisation, which turns `™` into `tm` where
SentencePiece maps it to `<unk>`. Words containing characters outside the BPEmb
vocabulary get the `<unk>` vector, as in the original.

## Use with DeLFT

Any static embedding model of this layout can be given to the sequence labelling and
text classification applications by Hub id or local directory:

```bash
python -m delft.applications.grobidTagger affiliation-address train_eval \
    --architecture BidLSTM_CRF_FEATURES \
    --embedding sciencialab/bpemb-multi-1M
```

or through an entry of `delft/resources-registry.json`:

```json
{
    "name": "bpemb-multi",
    "model": "sciencialab/bpemb-multi-1M",
    "type": "bpemb",
    "format": "static-transformer",
    "lang": "multi",
    "item": "word"
}
```

DeLFT's sequence labelling already maps digits to `0` before the lookup, as the
embeddings expect.

## Why: evaluation as a GloVe replacement

The embeddings were benchmarked in November 2026 against `glove-840B` and other static
embeddings on two GROBID training corpora (`affiliation-address` and `citation`),
architecture `BidLSTM_CRF_FEATURES`, 3 folds, test sequences stratified by script.
Micro F1 pooled over the 3 folds, with the number of folds where BPEmb beats GloVe:

| | affiliation non-English | affiliation English | citation non-English Latin | citation CJK | citation English |
|---|---|---|---|---|---|
| glove-840B | 80.88 | 87.09 | 79.99 | 92.15 | 94.27 |
| **this model** | **82.35** (3/3) | **87.65** (3/3) | **80.79** (3/3) | **94.51** (3/3) | 94.10 (1/3) |

GloVe covers 8.8% of the CJK citation tokens and about 86% of the non-English Latin
ones; this model covers all of them. The full comparison, including fastText aligned
vectors, Numberbatch, potion and static sentence-transformers models, is in
`experiments/multilingual/README.md` on the branch named above. The non-English test
sets are small (163 affiliation sequences, 280 non-English Latin and 338 CJK citations).

## Limitations

- Lowercased, digits mapped to zero: casing has to come from other features of the
  model using these vectors.
- A word vector is the mean of its piece vectors: unrelated words sharing pieces get
  related vectors.
- Trained on Wikipedia text extracted in 2018-2019, in 265 languages; vocabulary and
  usage of later years are not covered.
- Non-Python runtimes have to reproduce the tokenization and the mean pooling; the
  vectors alone are not a word lookup table.

## Licence

The BPEmb code repository is under the MIT licence. The download page of the embedding
files does not state a licence for the files themselves, and this repackaging adds no
rights of its own. The licence of this repository is the licence of the original files,
to be confirmed with the original authors; until then, treat the files as the authors'
work released for research use and keep the attribution above.

## Citation

Please cite the original work:

```bibtex
@InProceedings{heinzerling2018bpemb,
  author = {Benjamin Heinzerling and Michael Strube},
  title = "{BPEmb: Tokenization-free Pre-trained Subword Embeddings in 275 Languages}",
  booktitle = {Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)},
  year = {2018},
  month = {May 7-12, 2018},
  address = {Miyazaki, Japan},
  editor = {Nicoletta Calzolari (Conference chair) and Khalid Choukri and Christopher Cieri and Thierry Declerck and Sara Goggi and Koiti Hasida and Hitoshi Isahara and Bente Maegaard and Joseph Mariani and Hélène Mazo and Asuncion Moreno and Jan Odijk and Stelios Piperidis and Takenobu Tokunaga},
  publisher = {European Language Resources Association (ELRA)},
  isbn = {979-10-95546-00-9},
  language = {english}
}

@inproceedings{heinzerling-strube-2019-sequence,
    title = "Sequence Tagging with Contextual and Non-Contextual Subword Representations: A Multilingual Evaluation",
    author = "Heinzerling, Benjamin and Strube, Michael",
    editor = "Korhonen, Anna and Traum, David and M{\`a}rquez, Llu{\'i}s",
    booktitle = "Proceedings of the 57th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-1027/",
    doi = "10.18653/v1/P19-1027",
    pages = "273--291"
}
```

The GloVe algorithm used to train the vectors is Pennington, Socher & Manning, *GloVe:
Global Vectors for Word Representation*, EMNLP 2014. SentencePiece is Kudo &
Richardson, EMNLP 2018 (system demonstrations).
