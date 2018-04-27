[![License](http://img.shields.io/:license-apache-blue.svg)](http://www.apache.org/licenses/LICENSE-2.0.html)

# DeLFT 

DeLFT (Deep Learning Framework for Text) is a Keras framework for text processing, covering sequence labelling (e.g. entity tagging) and text classification (e.g. commenty classification). This library re-implements standard and state-of-the-art Deep Learning architectures which can all be used within the same environment. 

DeLFT has been tested with python 3.5, Keras 2.1 and Tensorflow 1.7 as backend. GPU(s) are required for decent training runtime. 


## Sequence Labelling

### Available models

- BidLSTM-CRF with words and characters input following: 

Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer. "Neural Architectures for Named Entity Recognition". Proceedings of NAACL 2016. https://arxiv.org/abs/1603.01360

- BidLSTM-CNN with words, characters and custum features input following: 

Jason P. C. Chiu, Eric Nichols. "Named Entity Recognition with Bidirectional LSTM-CNNs". 2016. https://arxiv.org/abs/1511.08308

### Usage



### Examples

#### NER

For training and evaluating based on the CONLL 2004 model:

> python3 nerTagger.py train



> python3 nerTagger.py tag

#### GROBID models

DeLFT supports GROBID training data (originally for CRF) and GROBID feature matrix to be labelled. 


#### Insult recognition

A small experimental model for recognizing insults in Wikipedia comment (from the Kaggle _Wikipedia Toxic Comments_ dataset, only English).

> python3 insultTagger.py train

> python3 insultTagger.py tag

#### Creating your own model

As long your task is a sequence labelling of text, adding a new corpus and create an additional model should be straightfoward. If you want to build a model named `toto` based on labelled data in one of the supported format (CoNLL, TEI or GROBID CRF), create the subdirectory `data/sequenceLabelling/toto` and copy your training data under it.  


## Text classification

### Available models



### Usage


### Examples

#### Toxic comment classification

The dataset of the [Kaggle Toxic Comment Classification challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) can be found here: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

This is a multi-label classification problem, where a Wikipedia comment (or any similar short texts) should be associated to 6 possible types of toxicity (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`).


#### Twitter 


#### Citation classification

We use the dataset developed and presented by A. Athar in the following article:

Awais Athar. "Sentiment Analysis of Citations using Sentence Structure-Based Features". Proceedings of the ACL 2011 Student Session, 81-87, 2011. http://www.aclweb.org/anthology/P11-3015

For a given scientific article, the task is to estimate if the occurrence of a bibliographical citation is positive, neutral or negative given its citation context. 



## License and contact

Distributed under [Apache 2.0 license](http://www.apache.org/licenses/LICENSE-2.0). 

Contact: Patrice Lopez (<patrice.lopez@science-miner.com>)

