# DeLFT Documentation

**DeLFT** (**De**ep **L**earning **F**ramework for **T**ext) is a deep-learning framework for text processing, focusing on **sequence labelling** (named-entity tagging, information extraction, document-structure tagging) and **text classification** (e.g. comment classification, citation classification). It re-implements standard state-of-the-art deep-learning architectures — both classical RNN/CNN models and transformer-based models loaded via HuggingFace — under a single API.

DeLFT is designed around three goals: covering rich text (tokens with layout / structural features, not just plain sentences), reproducibility and benchmarking under comparable evaluation criteria, and production-level performance and integration. A native Java integration of the library is available in [GROBID](https://github.com/kermitt2/grobid) via [JEP](https://github.com/ninia/jep).

The current release line is **0.5.x**, built on PyTorch and tested with Python 3.10/3.11. Upgrading from a TensorFlow-based build? See [Install DeLFT → Upgrading](Install-DeLFT.md#upgrading) first. See [Introduction](Introduction.md) for the full feature overview, or jump straight to:

- [Install DeLFT](Install-DeLFT.md) — get a working environment in a few commands, plus the [Upgrading](Install-DeLFT.md#upgrading) notes (including the 0.4.x → 0.5.x PyTorch migration).
- [Embeddings](embeddings.md) — how DeLFT manages static word embeddings via LMDB.
- [NER](ner.md), [GROBID models](grobid.md), [Snippet classification](classifiers.md) — ready-to-use applications and reproducibility tables.
- [Sequence Labeling](sequence_labeling.md) and [Text Classification](text_classification.md) — supported architectures and how to add your own.
- [Experiment tracking (W&B)](wandb.md) — log training and evaluation runs to Weights & Biases.
- [Training on a cluster (SLURM)](distributed_training.md) — single-node multi-GPU training and the SLURM submitter scripts.

The full navigation is available in the sidebar on the left.
