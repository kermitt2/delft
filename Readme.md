<img align="right" width="150" height="150" src="doc/cat-delft-small.jpg">

[![Documentation Status](https://readthedocs.org/projects/delft/badge/?version=latest)](https://readthedocs.org/projects/delft/?badge=latest)
[![Build](https://github.com/kermitt2/delft/actions/workflows/ci-build-unstable.yml/badge.svg)](https://github.com/kermitt2/delft/actions/workflows/ci-build-unstable.yml)
[![PyPI version](https://badge.fury.io/py/delft.svg)](https://badge.fury.io/py/delft)
[![SWH](https://archive.softwareheritage.org/badge/origin/https://github.com/kermitt2/delft/)](https://archive.softwareheritage.org/browse/origin/https://github.com/kermitt2/delft/)
[![License](http://img.shields.io/:license-apache-blue.svg)](http://www.apache.org/licenses/LICENSE-2.0.html)
[![Downloads](https://static.pepy.tech/badge/delft)](https://pepy.tech/project/delft)


# DeLFT

__DeLFT__ (**De**ep **L**earning **F**ramework for **T**ext) is a Keras and TensorFlow framework for text processing, focusing on sequence labeling (e.g. named entity tagging, information extraction) and text classification (e.g. comment classification). This library re-implements standard state-of-the-art Deep Learning architectures relevant to text processing tasks.  

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

* Integration of HuggingFace transformers as Keras layers.

A native Java integration of the library has been realized in [GROBID](https://github.com/kermitt2/grobid) via [JEP](https://github.com/ninia/jep).

The latest DeLFT release __0.4.5__ has been tested successfully with Python 3.10/3.11 and TensorFlow 2.17. As always, GPU(s) are required for decent training time. For example, a GeForce GTX 1050 Ti (4GB) is working very well for running RNN models and BERT or RoBERTa base models. Using BERT large model is no problem with a GeForce GTX 1080 Ti (11GB), including training with modest batch size. Using multiple GPUs (training and inference) is supported.

## Changes in 0.4.1

### Breaking changes

- **TensorFlow 2.17 / tf_keras 2.17**: DeLFT now requires TensorFlow 2.17.1 and the standalone `tf_keras` 2.17.0 package. All Keras imports have been updated from `tensorflow.keras` to `tf_keras`. Pre-trained model weights from 0.3.4 are **not directly compatible**, but can be converted without retraining:

  ```sh
  python -m delft.utilities.convert_model --input <old-model-dir> --output <new-model-dir> --verify
  ```

  The converter rebuilds the model architecture from the saved `config.json`, remaps weights from the old HDF5 file into the fresh model, and saves a new weights file. Additional flags: `--redownload-tokenizer` (when the saved tokenizer is incompatible with the current `transformers` version), `--force-partial` (allow partial conversion when some weights cannot be matched), `--dry-run` (inspect without writing). Use `--help` for full options.

- **Python 3.10+ required**: Python 3.8 and 3.9 are no longer supported.

- **CUDA 12.1 required for GPU**: TensorFlow 2.17 requires CUDA 12.1. On Linux, torch is no longer included in the base `pip install delft` to avoid CUDA version conflicts between torch (CUDA 12.4) and TensorFlow (CUDA 12.1). Use `pip install "delft[gpu]"` with the PyTorch cu121 index instead (see installation instructions below).

- **LMDB embedding format changed**: Embeddings are now stored as raw float32 bytes instead of pickle-serialized objects. This enables Java interoperability (used by [GROBID](https://github.com/kermitt2/grobid)) and improves performance. Existing LMDB caches must be converted using the provided utility:

  ```sh
  python -m delft.utilities.convert_lmdb_embeddings --input <old-lmdb-path> --output <new-lmdb-path>
  ```

- **ELMo support removed**: ELMo embeddings are no longer supported. The `use_ELMo` parameter has been removed from all application scripts and configurations. Use transformer-based models (BERT, SciBERT, etc.) or static embeddings (GloVe, fastText) instead.

### Other changes

- Weights & Biases integration for experiment tracking (`--wandb` flag)
- Distributed training support via SLURM scripts
- Additional checks for avoiding empty embeddings
- Updated default word2vec embedding URL
- Updated dependency versions (transformers 4.48, torch 2.5.1, numpy 1.26.4, scikit-learn 1.6.1, pandas 2.2.3)

## DeLFT Documentation

Visit the [DELFT documentation](https://delft.readthedocs.io) for detailed information on installation, usage and models.

## Using DeLFT 

PyPI packages are available for stable versions. Latest stable version is `0.4.1`:

```sh
# macOS
pip install delft==0.4.1

# Linux with CUDA 12.1 (GPU)
pip install "delft[gpu]==0.4.1" --extra-index-url https://download.pytorch.org/whl/cu121
```

## DeLFT Installation

For installing DeLFT and use the current master version, get the github repo:

```sh
git clone https://github.com/kermitt2/delft
cd delft
```

It is advised to setup first a virtual environment to avoid falling into one of these gloomy python dependency marshlands:

```sh
uv venv --python 3.11
source .venv/bin/activate
uv pip install pip
```

Install the project in editable state:

```sh
# macOS (torch is included automatically)
uv pip install -e .

# Linux with CUDA 12.1 (recommended for GPU)
uv pip install -e ".[gpu]" --extra-index-url https://download.pytorch.org/whl/cu121

# Linux with CUDA 12.1 (alternative using requirements file)
uv pip install -e . -r requirements-cuda.txt
```

See the [DELFT documentation](https://delft.readthedocs.io) for usage. 

### Send data to Weight and Biases

1. Create a file `.env` in the root of the project with the following content:
   
    ```
    WANDB_API_KEY=your_api_key
    WANDB_PROJECT=your_project_name
    WANDB_ENTITY=your_entity_name
    ```
2. use the parameter `--wandb` when running the scripts, e.g.
    ```sh   
    python -m applications.delft.grobidTagger date train --architecture BidLSTM --wandb
    ```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to get started, code style, running tests, and the pull request process.

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
