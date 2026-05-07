# DeLFT Documentation

**DeLFT** (**De**ep **L**earning **F**ramework for **T**ext) is a Keras and TensorFlow framework for text processing, focusing on **sequence labelling** (named-entity tagging, information extraction, document-structure tagging) and **text classification** (e.g. comment classification, citation classification). It re-implements standard state-of-the-art deep-learning architectures — both classical RNN/CNN models and transformer-based models loaded via HuggingFace — under a single API.

DeLFT is designed around three goals: covering rich text (tokens with layout / structural features, not just plain sentences), reproducibility and benchmarking under comparable evaluation criteria, and production-level performance and integration. A native Java integration of the library is available in [GROBID](https://github.com/kermitt2/grobid) via [JEP](https://github.com/ninia/jep).

The current release is **0.4.6**, tested with Python 3.10/3.11 and TensorFlow 2.17. See [Introduction](Introduction.md) for the full feature overview, or jump straight to:

- [Install DeLFT](Install-DeLFT.md) — get a working environment in a few commands.
- [Embeddings](embeddings.md) — how DeLFT manages static word embeddings via LMDB.
- [NER](ner.md), [GROBID models](grobid.md), [Snippet classification](classifiers.md) — ready-to-use applications and reproducibility tables.
- [Sequence Labeling](sequence_labeling.md) and [Text Classification](text_classification.md) — supported architectures and how to add your own.

The full navigation is available in the sidebar on the left.
