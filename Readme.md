<img align="right" width="150" height="150" src="doc/cat-delft-small.jpg">

[![Documentation Status](https://readthedocs.org/projects/delft/badge/?version=latest)](https://readthedocs.org/projects/delft/?badge=latest)
[![Build](https://github.com/kermitt2/delft/actions/workflows/ci-build-unstable.yml/badge.svg)](https://github.com/kermitt2/delft/actions/workflows/ci-build-unstable.yml)
[![PyPI version](https://badge.fury.io/py/delft.svg)](https://badge.fury.io/py/delft)
[![SWH](https://archive.softwareheritage.org/badge/origin/https://github.com/kermitt2/delft/)](https://archive.softwareheritage.org/browse/origin/https://github.com/kermitt2/delft/)
[![License](http://img.shields.io/:license-apache-blue.svg)](http://www.apache.org/licenses/LICENSE-2.0.html)
[![Downloads](https://static.pepy.tech/badge/delft)](https://pepy.tech/project/delft)


# DeLFT

__DeLFT__ (**De**ep **L**earning **F**ramework for **T**ext) is a deep-learning framework for text processing, focusing on sequence labeling (e.g. named entity tagging, information extraction) and text classification (e.g. comment classification). This library re-implements standard state-of-the-art Deep Learning architectures relevant to text processing tasks.  

DeLFT has three main purposes: 

1. __Covering text and rich texts__: most of the existing Deep Learning works in NLP only consider simple texts as input. In addition to simple texts, we also target _rich text_ where tokens are associated to layout information (font. style, etc.), positions in structured documents, and possibly other lexical or symbolic contextual information. Text is usually coming from large documents like PDF or HTML, and not just from segments like sentences or paragraphs, and contextual features appear very useful. Rich text is the most common textual content used by humans to communicate and work.

2. __Reproducibility and benchmarking__: by implementing several references/state-of-the-art models for both sequence labeling and text classification tasks, we want to offer the capacity to easily validate reported results and to benchmark several methods under the same conditions and criteria.

3. __Production level__, by offering optimzed performance, robustness and integration possibilities, we aim at supporting better engineering decisions/trade-off and successful production-level applications. 

Some contributions include: 

* A variety of modern NLP architectures and tasks to be used following the same API and input formats, including RNN and transformers.

* Reduction of the size of RNN models, in particular by removing word embeddings from them. For instance, the model for the toxic comment classifier went down from a size of 230 MB with embeddings to 1.8 MB. In practice the size of all the models of DeLFT is less than 2 MB, except for Ontonotes 5.0 NER model which is 4.7 MB.

* Implementation of a generic support of categorical features, available in various architectures. 

* Usage of dynamic data generator so that the training data do not need to stand completely in memory.

* Efficient loading and management of an unlimited volume of static pre-trained embeddings.

* A comprehensive evaluation framework with the standard metrics for sequence labeling and classification tasks, including n-fold cross validation. 

* Native integration of HuggingFace transformers (including recent models such as ModernBERT).

A native Java integration of the library has been realized in [GROBID](https://github.com/kermitt2/grobid) via [JEP](https://github.com/ninia/jep).

The DeLFT __0.5.x__ release line is built on PyTorch and has been tested successfully with Python 3.10/3.11 (see the PyPI badge above for the exact current version). As always, GPU(s) are required for decent training time. For example, a GeForce GTX 1050 Ti (4GB) is working very well for running RNN models and BERT or RoBERTa base models. Using BERT large model is no problem with a GeForce GTX 1080 Ti (11GB), including training with modest batch size. Using multiple GPUs (training and inference) is supported.

## Migrating to PyTorch (0.4.x → 0.5.x)

The **0.5.x** release line replaces the TensorFlow/Keras backend (used up to 0.4.x) with **PyTorch**. The public API, CLI entrypoints, architecture names and data formats are unchanged, but the runtime and saved weights are not. See the [Upgrading section of the installation guide](doc/Install-DeLFT.md#upgrading) for the full step-by-step. Highlights:

- **PyTorch backend**: `torch` 2.11, `transformers` 5.7 (native, no Keras-layer wrapping), and [`pytorch-crf`](https://pypi.org/project/pytorch-crf/) replace TensorFlow / `tf_keras` / `tensorflow-addons`. Newer transformers such as ModernBERT are supported.
- **Retrain custom models**: TensorFlow weights cannot be loaded into the PyTorch models and there is **no automatic converter** (the previous `convert_model` utility was removed). Bundled application models have been regenerated for PyTorch; custom models must be retrained with their original `train` / `train_eval` command.
- **GPU / CUDA 12.8**: GPU builds target CUDA 12.8 via the `[gpu]` extra (see installation below). Reinstall into a fresh virtual environment to avoid leftover `tensorflow` packages.
- **ELMo support removed**: the `--use-ELMo` flag and `use_ELMo` parameter are gone. Use transformer-based models (BERT, SciBERT, …) or static embeddings (GloVe, fastText) instead.
- **LMDB embedding caches** built before 0.4.x use the legacy pickle format; convert them once with `python -m delft.utilities.convert_lmdb_embeddings --input <old> --output <new>` (or simply rebuild).

### Other changes

- Weights & Biases integration for experiment tracking, with resume support (`--wandb` flag) — see [Experiment tracking (W&B)](doc/wandb.md)
- Distributed / multi-GPU training support via SLURM scripts and `accelerate` — see [Training on a cluster (SLURM)](doc/distributed_training.md)
- ONNX export for `BidLSTM_CRF` / `BidLSTM_CRF_FEATURES` models (see below)
- Additional checks for avoiding empty embeddings

## DeLFT Documentation

Visit the [DELFT documentation](https://delft.readthedocs.io) for detailed information on installation, usage and models.

## Using DeLFT 

PyPI packages are available for stable versions:

```sh
# macOS or Linux CPU
pip install delft

# Linux with CUDA 12.8 (adds PyTorch GPU support)
pip install "delft[gpu]" --extra-index-url https://download.pytorch.org/whl/cu128
```

> **Note:** The base install pulls the standard PyTorch wheel from PyPI — that's a CPU build on Linux, and a native build on macOS arm64 that includes MPS (Apple Silicon GPU) support automatically. The `[gpu]` extra is for NVIDIA CUDA 12.8 on Linux.

## DeLFT Installation

For installing DeLFT and use the current master version, get the github repo:

```sh
git clone https://github.com/kermitt2/delft
cd delft
```

It is advised to setup first a virtual environment to avoid falling into one of these gloomy python dependency marshlands:

```sh
uv venv --python=3.11
source .venv/bin/activate
uv pip install pip
```

Install the project in editable state:

```sh
# macOS or Linux CPU (torch is included automatically)
uv pip install -e .

# Linux with CUDA 12.8 (recommended for NVIDIA GPU)
uv pip install -e ".[gpu]" --extra-index-url https://download.pytorch.org/whl/cu128
```

> **For training a model:** add the `[dev]` extras — e.g. `uv pip install -e ".[gpu,dev]"`. The base install does not include `wandb`, `pytest`, or `ruff`, so training with W&B tracking (`--wandb`) requires the dev extras (or `pip install wandb` ad-hoc).

See the [DELFT documentation](https://delft.readthedocs.io) for usage. 

### Send data to Weight and Biases

0. Make sure `wandb` is installed — it ships with the `[dev]` extras (`uv pip install -e ".[gpu,dev]"`) and is **not** in the base install.
1. Create a file `.env` in the root of the project with the following content:
   
    ```
    WANDB_API_KEY=your_api_key
    WANDB_PROJECT=your_project_name
    WANDB_ENTITY=your_entity_name
    ```
2. use the parameter `--wandb` when running the scripts, e.g.
    ```sh   
    python -m delft.applications.grobidTagger date train --architecture BidLSTM --wandb
    ```

### ONNX Export

DeLFT supports exporting trained sequence labeling models (`BidLSTM_CRF` and `BidLSTM_CRF_FEATURES`) to ONNX format for inference in Java or other runtimes.

**Export a model:**

```sh
python -m delft.applications.onnx_export \
    --model grobid-date-BidLSTM_CRF_FEATURES \
    --output exported_models/date-features
```

This creates:
- `encoder.onnx` - BiLSTM encoder model
- `crf_params.json` - CRF transition matrices
- `vocab.json` - Character and label vocabularies
- `config.json` - Model configuration

**Java inference:**

A Java inference library is available in `java/delft-onnx/`. Build and run:

```sh
cd java/delft-onnx
./gradlew build
./gradlew run --args="--model ../../exported_models/date-features \
    --embeddings ../../data/db/glove-840B-raw \
    --input 'December 25, 2024'"
```

Note: For feature models, sample features are auto-generated for demo purposes.

## License and contact

Distributed under [Apache 2.0 license](http://www.apache.org/licenses/LICENSE-2.0). The dependencies used in the project are either themselves also distributed under Apache 2.0 license or distributed under a compatible license.

If you contribute to DeLFT, you agree to share your contribution following these licenses. 

Contact: Patrice Lopez (<patrice.lopez@science-miner.com>) and Luca Foppiano (@lfoppiano).

## How to cite

If you want to this work, please refer to the present GitHub project, together with the [Software Heritage](https://www.softwareheritage.org/) project-level permanent identifier. For example, with BibTeX:

```bibtex
@misc{DeLFT,
    title = {DeLFT},
    howpublished = {\url{https://github.com/kermitt2/delft}},
    publisher = {GitHub},
    year = {2018--2026},
    archivePrefix = {swh},
    eprint = {1:dir:54eb292e1c0af764e27dd179596f64679e44d06e}
}
```
