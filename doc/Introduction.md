<img align="right" width="150" height="150" src="doc/cat-delft-small.jpg">

[![Documentation Status](https://readthedocs.org/projects/delft/badge/?version=latest)](https://readthedocs.org/projects/delft/?badge=latest)
[![PyPI version](https://badge.fury.io/py/delft.svg)](https://badge.fury.io/py/delft)
[![SWH](https://archive.softwareheritage.org/badge/origin/https://github.com/kermitt2/delft/)](https://archive.softwareheritage.org/browse/origin/https://github.com/kermitt2/delft/)
[![License](http://img.shields.io/:license-apache-blue.svg)](http://www.apache.org/licenses/LICENSE-2.0.html)


# DeLFT

__DeLFT__ (**De**ep **L**earning **F**ramework for **T**ext) is a Keras and TensorFlow framework for text processing, focusing on sequence labeling (e.g. named entity tagging, information extraction) and text classification (e.g. comment classification). This library re-implements standard state-of-the-art Deep Learning architectures relevant to text processing tasks.  

DeLFT has three main purposes: 

1. __Usefulness__: most of the existing Deep Learning works in NLP only consider simple texts as input. In addition to simple texts, we also target _rich text_ where tokens are associated to layout information (font. style, etc.), positions in structured documents, and possibly other lexical or symbolic contextual information. Text is usually coming from large documents like PDF or HTML, and not just from segments like sentences or paragraphs, so contextual features is useful. Rich text is the most common textual content used by humans to communicate and work.

2. __Reproducibility and benchmarking__: by implementing several state-of-the-art algorithms for both sequence labeling and text classification tasks, we want to offer the capacity to validate reported results and to benchmark several methods under the same conditions and criteria.

3. __Production level__, by offering optimzed performance, robustness and integration possibilities, we aim at supporting better engineering decisions/trade-off and successful production-level applications. 

Some contributions include: 

* Reduction of the size of RNN models, in particular by removing word embeddings from them. For instance, the model for the toxic comment classifier went down from a size of 230 MB with embeddings to 1.8 MB. In practice the size of all the models of DeLFT is less than 2 MB, except for Ontonotes 5.0 NER model which is 4.7 MB.

* Implementation of a generic support of categorical features, available in various architectures. 

* Usage of dynamic data generator so that the training data do not need to stand completely in memory.

* Efficiently loading and management of an unlimited volume of static pre-trained embeddings.

* A comprehensive evaluation framework with the standard metrics for sequence labeling and classification tasks, including n-fold cross validation. 

* Integration of HuggingFace transformers as Keras layers

A native Java integration of the library has been realized in [GROBID](https://github.com/kermitt2/grobid) via [JEP](https://github.com/ninia/jep).

The latest DeLFT release has been tested with python 3.7 and Tensorflow 2.7.0. As always, GPU(s) are required for decent training time: a GeForce GTX 1050 Ti for instance is absolutely fine for most RNN models. Using BERT Base model is fine with a GeForce GTX 1080 Ti.