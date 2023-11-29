<img align="right" width="150" height="150" src="doc/cat-delft-small.jpg">

[![Documentation Status](https://readthedocs.org/projects/delft/badge/?version=latest)](https://readthedocs.org/projects/delft/?badge=latest)
[![Build Status](https://travis-ci.org/kermitt2/delft.svg?branch=master)](https://travis-ci.org/kermitt2/delft)
[![PyPI version](https://badge.fury.io/py/delft.svg)](https://badge.fury.io/py/delft)
[![SWH](https://archive.softwareheritage.org/badge/origin/https://github.com/kermitt2/delft/)](https://archive.softwareheritage.org/browse/origin/https://github.com/kermitt2/delft/)
[![License](http://img.shields.io/:license-apache-blue.svg)](http://www.apache.org/licenses/LICENSE-2.0.html)


# DeLFT

__DeLFT__ (**De**ep **L**earning **F**ramework for **T**ext) is a Keras and TensorFlow framework for text processing, focusing on sequence labeling (e.g. named entity tagging, information extraction) and text classification (e.g. comment classification). This library re-implements standard state-of-the-art Deep Learning architectures relevant to text processing tasks.  

DeLFT has three main purposes: 

1. __Covering text and rich texts__: most of the existing Deep Learning works in NLP only consider simple texts as input. In addition to simple texts, we also target _rich text_ where tokens are associated to layout information (font. style, etc.), positions in structured documents, and possibly other lexical or symbolic contextual information. Text is usually coming from large documents like PDF or HTML, and not just from segments like sentences or paragraphs, and contextual features appear very useful. Rich text is the most common textual content used by humans to communicate and work.

2. __Reproducibility and benchmarking__: by implementing several references/state-of-the-art models for both sequence labeling and text classification tasks, we want to offer the capacity to easily validate reported results and to benchmark several methods under the same conditions and criteria.

3. __Production level__, by offering optimzed performance, robustness and integration possibilities, we aim at supporting better engineering decisions/trade-off and successful production-level applications. 

Some contributions include: 

* A variety of modern NLP architectures and tasks to be used following the same API and input formats, including RNN, ELMo and transformers.

* Reduction of the size of RNN models, in particular by removing word embeddings from them. For instance, the model for the toxic comment classifier went down from a size of 230 MB with embeddings to 1.8 MB. In practice the size of all the models of DeLFT is less than 2 MB, except for Ontonotes 5.0 NER model which is 4.7 MB.

* Implementation of a generic support of categorical features, available in various architectures. 

* Usage of dynamic data generator so that the training data do not need to stand completely in memory.

* Efficient loading and management of an unlimited volume of static pre-trained embeddings.

* A comprehensive evaluation framework with the standard metrics for sequence labeling and classification tasks, including n-fold cross validation. 

* Integration of HuggingFace transformers as Keras layers.

A native Java integration of the library has been realized in [GROBID](https://github.com/kermitt2/grobid) via [JEP](https://github.com/ninia/jep).

The latest DeLFT release has been tested successfully with python 3.8 and Tensorflow 2.9.3. As always, GPU(s) are required for decent training time. A GeForce GTX 1050 Ti (4GB) for instance is fine for running RNN models and BERT or RoBERTa base models. Using BERT large model is possible from a GeForce GTX 1080 Ti (11GB) with modest batch size. Using multiple GPUs (training and inference) is supported.

## DeLFT Documentation

Visit the [DELFT documentation](https://delft.readthedocs.io) for detailed information on installation, usage and models.

## Using DeLFT 

PyPI packages are available for stable versions. Latest stable version is `0.3.4`:

```
pip install delft==0.3.4
```

## DeLFT Installation

For installing DeLFT and use the current master version, get the github repo:

```sh
git clone https://github.com/kermitt2/delft
cd delft
```

It is advised to setup first a virtual environment to avoid falling into one of these gloomy python dependency marshlands:

```sh
virtualenv --system-site-packages -p python3.8 env
source env/bin/activate
```

Install the dependencies:

```sh
pip3 install -r requirements.txt
```

Finally install the project, preferably in editable state

```sh
pip3 install -e .
```

See the [DELFT documentation](https://delft.readthedocs.io) for usage. 

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
    year = {2018--2023},
    archivePrefix = {swh},
    eprint = {1:dir:54eb292e1c0af764e27dd179596f64679e44d06e}
}
```
