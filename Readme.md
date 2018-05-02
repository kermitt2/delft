[![License](http://img.shields.io/:license-apache-blue.svg)](http://www.apache.org/licenses/LICENSE-2.0.html)

# DeLFT 

__Work in progress !__ This is an early alpha version and you very likely don't want to try it at this stage ;) 


__DeLFT__ (**De**ep **L**earning **F**ramework for **T**ext) is a Keras framework for text processing, covering sequence labelling (e.g. named entity tagging) and text classification (e.g. comment classification). This library re-implements standard and _augment_ state-of-the-art Deep Learning architectures which can all be used within the same environment. 

The medium term goal is then to provide good performance (accuracy and runtime) models to a production stack such as Java/Scala and C++. 

DeLFT has been tested with python 3.5, Keras 2.1 and Tensorflow 1.7 as backend. As always, GPU(s) are required for decent training time. 

## Sequence Labelling

### Available models

- BidLSTM-CRF with words and characters input following: 

Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer. "Neural Architectures for Named Entity Recognition". Proceedings of NAACL 2016. https://arxiv.org/abs/1603.01360

<!--
- BidLSTM-CNN with words, characters and custum features input following: 

Jason P. C. Chiu, Eric Nichols. "Named Entity Recognition with Bidirectional LSTM-CNNs". 2016. https://arxiv.org/abs/1511.08308
-->

### Usage

TBD

### Examples

#### NER

Assuming that the usual CoNLL-2003 NER dataset (`eng.train`, `eng.testa`, `eng.testb`) is present under `data/sequenceLabelling/CoNLL-2003/`, for training and evaluating use:

> python3 nerTagger.py train

By default, the BidLSTM-CRF model is used. 

For tagging some text, use the command:

> python3 nerTagger.py tag

#### GROBID models

DeLFT supports GROBID training data (originally for CRF) and GROBID feature matrix to be labelled. 


#### Insult recognition

A small experimental model for recognizing insults in texts, based on the Wikipedia comment from the Kaggle _Wikipedia Toxic Comments_ dataset, English only. This uses a small dataset labelled manually. 

For training:

> python3 insultTagger.py train

By default training uses the whole train set, for n-folds training, which provides better and more stable accuracy, uses: 

> python3 insultTagger.py tag --fold-count 10

#### Creating your own model

As long your task is a sequence labelling of text, adding a new corpus and create an additional model should be straightfoward. If you want to build a model named `toto` based on labelled data in one of the supported format (CoNLL, TEI or GROBID CRF), create the subdirectory `data/sequenceLabelling/toto` and copy your training data under it.  


## Text classification

### Available models

All the following models includes Dropout, Pooling and Dense layers with hyperparameters tuned for reasonable performance across standard text classification tasks. If necessary, they are good basis for further performance tuning. 
 
* `gru`: two layers Bidirectional GRU
* `gru_simple`: one layer Bidirectional GRU
* `bidLstm`: a Bidirectional LSTM layer followed by an Attention layer
* `cnn`: convolutional layers followed by a GRU
* `lstm_cnn`: LSTM followed by convolutional layers
* `mix1`: one layer Bidirectional GRU followed by a Bidirectional LSTM
* `dpcnn`: Deep Pyramid Convolutional Neural Networks (but not working as expected - to be reviewed)

Note: by default the first 300 tokens of the text to be classified are used, which is largely enough for any _short text_ classification tasks and works fine with low profile GPU (for instance GeForce GTX 1050 with 4 GB memory). For taking into account a larger portion of the text, modify the config model parameter `maxlen`. However, using more than 1000 tokens for instance requires a modern GPU with enough memory (e.g. 10 GB). 

### Usage

TBD

### Examples

#### Toxic comment classification

The dataset of the [Kaggle Toxic Comment Classification challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) can be found here: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

This is a multi-label regression problem, where a Wikipedia comment (or any similar short texts) should be associated to 6 possible types of toxicity (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`).

To launch the training: 

> python3 toxicCommentClassifier.py train

For training with n-folds, use the parameter `--fold-count`:

> python3 toxicCommentClassifier.py train --fold-count 10

After training (1 or n-folds), to process the Kaggle test set, use:

> python3 toxicCommentClassifier.py test

To classify a set of comments: 

> python3 toxicCommentClassifier.py classify


#### Twitter 

TBD

#### Citation classification

We use the dataset developed and presented by A. Athar in the following article:

Awais Athar. "Sentiment Analysis of Citations using Sentence Structure-Based Features". Proceedings of the ACL 2011 Student Session, 81-87, 2011. http://www.aclweb.org/anthology/P11-3015

For a given scientific article, the task is to estimate if the occurrence of a bibliographical citation is positive, neutral or negative given its citation context. 

In this example, we formulate the problem as a 3 class regression (`negative`. `neutral`, `positive`). To train the model:

> python3 citationClassifier.py train

with n-folds:

> python3 citationClassifier.py train --fold-count 10

Training and evalation (ratio):

> python3 citationClassifier.py train-eval

which should produce the following evaluation (using the 2-layers Bidirectional GRU model `gru`): 

```
Evaluation on 896 instances:

Class: negative
    accuracy at 0.5 = 0.9620535714285714
    log-loss = 0.10431599666467914
    roc auc = 0.9185061448514498

Class: neutral
    accuracy at 0.5 = 0.9631696428571429
    log-loss = 0.10217989354726699
    roc auc = 0.9225231674820029

Class: positive
    accuracy at 0.5 = 0.9296875
    log-loss = 0.20026920159911502
    roc auc = 0.8898697968041034

Macro-average:
    average accuracy at 0.5 = 0.9516369047619048
    average log-loss = 0.13558836393702037
    average roc auc = 0.910299703045852
```

To classify a set of citation contexts:

> python3 citationClassifier.py classify

which will produce some JSON output like this:

```json
{
    "model": "citations",
    "software": "DeLFT",
    "classifications": [
        {
            "positive": 0.8160770535469055,
            "text": "One successful strategy [15] computes the set-similarity involving (multi-word) keyphrases about the mentions and the entities, collected from the KG.",
            "negative": 0.0014772837748751044,
            "neutral": 0.002285155700519681
        },
        {
            "positive": 0.05530614033341408,
            "text": "Unfortunately, fewer than half of the OCs in the DAML02 OC catalog (Dias et al. 2002) are suitable for use with the isochrone-fitting method because of the lack of a prominent main sequence, in addition to an absence of radial velocity and proper-motion data.",
            "negative": 0.2548907399177551,
            "neutral": 0.23885516822338104
        },
        {
            "positive": 0.8472888469696045,
            "text": "However, we found that the pairwise approach LambdaMART [41] achieved the best performance on our datasets among most learning to rank algorithms.",
            "negative": 0.16778403520584106,
            "neutral": 0.21162080764770508
        }
    ],
    "date": "2018-04-30T23:33:24.840211",
    "runtime": 0.686
}
```


## TODO


__Embeddings__: 

* use a data generator for feeding the models with embeddings, so that embeddings are removed from the models (models are really big because they contain each embeddings matrix for the training vocabulary, it does not make sense to have a set of big and redundant models like that in production), see `model.fit_generator()`

* to free a lot of memory, use serialized embeddings with LMDB, similarly as in https://github.com/kermitt2/nerd/tree/0.0.3 (via the Python package called lmdb,  optionally see also Caffe for storing and using HDF5)

* use OOV (FastText) mechanisms

__Models__:

* Test Theano as alternative backend (waiting for Apache MXNet...)

* augment word vectors with features, in particular layout features generated by GROBID

__NER__:

* benchmark with OntoNores 5 and French NER

__Production stack__:

* see how efficiently feed and execute those Keras models with DL4J

__Build more models and examples__...

## License and contact

Distributed under [Apache 2.0 license](http://www.apache.org/licenses/LICENSE-2.0). 

Contact: Patrice Lopez (<patrice.lopez@science-miner.com>)

