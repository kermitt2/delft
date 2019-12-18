<img align="right" width="150" height="150" src="doc/cat-delft-small.jpg">

[![Build Status](https://travis-ci.org/kermitt2/delft.svg?branch=master)](https://travis-ci.org/kermitt2/delft)
[![PyPI version](https://badge.fury.io/py/delft.svg)](https://badge.fury.io/py/delft)
[![License](http://img.shields.io/:license-apache-blue.svg)](http://www.apache.org/licenses/LICENSE-2.0.html)



# DeLFT

__Work in progress !__

__DeLFT__ (**De**ep **L**earning **F**ramework for **T**ext) is a Keras framework for text processing, covering sequence labelling (e.g. named entity tagging) and text classification (e.g. comment classification). This library re-implements standard state-of-the-art Deep Learning architectures.

From the observation that most of the open source implementations using Keras are toy examples, our motivation is to develop a framework that can be efficient, scalable and more usable in a production environment (with all the known limitations of Python of course for this purpose). The benefits of DeLFT are:

* Re-implement a variety of state-of-the-art deep learning architectures for both sequence labelling and text classification problems, including the usage of the recent [ELMo](https://allennlp.org/elmo) and [BERT](https://github.com/google-research/bert) contextualised embeddings, which can all be used within the same environment. For instance, this allows to reproduce under similar conditions the performance of all recent NER systems, and even improve most of them.

* Reduce model size, in particular by removing word embeddings from them. For instance, the model for the toxic comment classifier went down from a size of 230 MB with embeddings to 1.8 MB. In practice the size of all the models of DeLFT is less than 2 MB, except for Ontonotes 5.0 NER model which is 4.7 MB.

* Use dynamic data generator so that the training data do not need to stand completely in memory.

* Load and manage efficiently an unlimited volume of pre-trained embeddings: instead of loading pre-trained embeddings in memory - which is horribly slow in Python and limits the number of embeddings to be used simultaneously - the pre-trained embeddings are compiled the first time they are accessed and stored efficiently in a LMDB database. This permits to have the pre-trained embeddings immediately "warm" (no load time), to free memory and to use any number of embeddings with a very negligible impact on runtime when using SSD.

The medium term goal is then to provide good performance (accuracy, runtime, compactness) models also to productions stack such as Java/Scala and C++. A native Java integration of these deep learning models has been realized in [GROBID](https://github.com/kermitt2/grobid) via [JEP](https://github.com/ninia/jep).

DeLFT has been tested with python 3.5, Keras 2.1 and Tensorflow 1.7+ as backend. At this stage, we do not guarantee that DeLFT will run with other different versions of these libraries or other Keras backend versions. As always, GPU(s) are required for decent training time: a GeForce GTX 1050 Ti for instance is absolutely OK without ELMo contextual embeddings. Using ELMo or BERT Base model is fine with a GeForce GTX 1080 Ti.

## Install

Get the github repo:

```sh
git clone https://github.com/kermitt2/delft
cd delft
```
It is advised to setup first a virtual environment to avoid falling into one of these gloomy python dependency marshlands:

```sh
virtualenv --system-site-packages -p python3 env
source env/bin/activate
```

Install the dependencies:

```sh
pip3 install -r requirements.txt
```

DeLFT uses tensorflow 1.7 as backend, and will exploit your available GPU with the condition that CUDA (>=8.0) is properly installed. 

You need then to download some pre-trained word embeddings and notify their path into the embedding registry. We suggest for exploiting the provided models:

* _glove Common Crawl_ (2.2M vocab., cased, 300 dim. vectors): [glove-840B](http://nlp.stanford.edu/data/glove.840B.300d.zip)

* _fasttext Common Crawl_ (2M vocab., cased, 300 dim. vectors): [fasttext-crawl](https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip)

* _word2vec GoogleNews_ (3M vocab., cased, 300 dim. vectors): [word2vec](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)

* _fasttext_wiki_fr_ (1.1M, NOT cased, 300 dim. vectors) for French: [wiki.fr](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.fr.vec)

* _ELMo_ trained on 5.5B word corpus (will produce 1024 dim. vectors) for English: [options](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json) and [weights](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5)

* _BERT_ for English, we are using BERT-Base, Cased, 12-layer, 768-hidden, 12-heads , 110M parameters: available [here](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip)

* _SciBERT_ for English and scientific content: [SciBERT-cased](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/tensorflow_models/scibert_scivocab_cased.tar.gz)

Then edit the file `embedding-registry.json` and modify the value for `path` according to the path where you have saved the corresponding embeddings. The embedding files must be unzipped.

```json
{
    "embeddings": [
        {
            "name": "glove-840B",
            "path": "/PATH/TO/THE/UNZIPPED/EMBEDDINGS/FILE/glove.840B.300d.txt",
            "type": "glove",
            "format": "vec",
            "lang": "en",
            "item": "word"
        },
        ...
    ]
}

```

You're ready to use DeLFT.

## Management of embeddings

The first time DeLFT starts and accesses pre-trained embeddings, these embeddings are serialised and stored in a LMDB database, a very efficient embedded database using memory page (already used in the Machine Learning world by Caffe and Torch for managing large training data). The next time these embeddings will be accessed, they will be immediately available.

Our approach solves the bottleneck problem pointed for instance [here](https://spenai.org/bravepineapple/faster_em/) in a much better way than quantising+compression or pruning. After being compiled and stored at the first access, any volume of embeddings vectors can be used immediately without any loading, with a negligible usage of memory, without any accuracy loss and with a negligible impact on runtime when using SSD. In practice, we can exploit for instance embeddings for dozen languages simultaneously, without any memory and runtime issues - a requirement for any ambitious industrial deployment of a neural NLP system. 

For instance, in a traditional approach `glove-840B` takes around 2 minutes to load and 4GB in memory. Managed with LMDB, after a first load time of around 4 minutes, `glove-840B` can be accessed immediately and takes only a couple MB in memory, for an impact on runtime negligible (around 1% slower) for any further command line calls.

By default, the LMDB databases are stored under the subdirectory `data/db`. The size of a database is roughly equivalent to the size of the original uncompressed embeddings file. To modify this path, edit the file `embedding-registry.json` and change the value of the attribute `embedding-lmdb-path`.

To get FastText .bin format support please uncomment the package `fasttextmirror==0.8.22` in `requirements.txt` or `requirements-gpu.txt` according to your system's configuration. Please note that the **.bin format is not supported on Windows platforms**. Installing the FastText .bin format support introduces the following additional dependencies:

* (gcc-4.8 or newer) or (clang-3.3 or newer)
* [Python](https://www.python.org/) version 2.7 or >=3.4
* [pybind11](https://github.com/pybind/pybind11)

While FastText .bin format are supported by DeLFT (including using ngrams for OOV words), this format will be loaded entirely in memory and does not take advantage of our memory-efficient management of embeddings.

> I have plenty of memory on my machine, I don't care about load time because I need to grab a coffee, I only process one language at the time, so I am not interested in taking advantage of the LMDB emebedding management !

Ok, ok, then set the `embedding-lmdb-path` value to `"None"` in the file `embedding-registry.json`, the embeddings will be loaded in memory as immutable data, like in the usual Keras scripts.

## Sequence Labelling

### Available models

* _BidLSTM-CRF_ with words and characters input following:

&nbsp;&nbsp;&nbsp;&nbsp; [1] Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer. "Neural Architectures for Named Entity Recognition". Proceedings of NAACL 2016. https://arxiv.org/abs/1603.01360

* _BidLSTM-CNN_ with words, characters and custom casing features input, see:

&nbsp;&nbsp;&nbsp;&nbsp; [2] Jason P. C. Chiu, Eric Nichols. "Named Entity Recognition with Bidirectional LSTM-CNNs". 2016. https://arxiv.org/abs/1511.08308

* _BidLSTM-CNN-CRF_ with words, characters and custom casing features input following:

&nbsp;&nbsp;&nbsp;&nbsp; [3] Xuezhe Ma and Eduard Hovy. "End-to-end Sequence Labelling via Bi-directional LSTM-CNNs-CRF". 2016. https://arxiv.org/abs/1603.01354

* _BidGRU-CRF_, similar to: 

&nbsp;&nbsp;&nbsp;&nbsp; [4] Matthew E. Peters, Waleed Ammar, Chandra Bhagavatula, Russell Power. "Semi-supervised sequence tagging with bidirectional language models". 2017. https://arxiv.org/pdf/1705.00108  

* the current state of the art (92.22% F1 on CoNLL2003 NER dataset, averaged over five runs), _BidLSTM-CRF_ with [ELMo](https://allennlp.org/elmo) contextualised embeddings, see:

&nbsp;&nbsp;&nbsp;&nbsp; [5] Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer. "Deep contextualized word representations". 2018. https://arxiv.org/abs/1802.05365

* Feature extraction to be used as contextual embeddings can also be obtained from BERT, as ELMo alternative, as explained in section 5.4 of: 

&nbsp;&nbsp;&nbsp;&nbsp; [6] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova, BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. 2018. https://arxiv.org/abs/1810.04805

The addition of BERT transformer architecture (with fine-tuning), as alternative to the above RNN architectures for sequence labeling, is currently work in progress. 

Note that all our annotation data for sequence labelling follows the [IOB2](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)) scheme.

### Examples

#### NER

##### Overview

We have reimplemented in DeLFT the main neural architectures for NER of the last two years and performed a reproducibility analysis of the these systems with comparable evaluation criterias. Unfortunaltely, in publications, systems are usually compared directly with reported results obtained in different settings, which can bias scores by more than 1.0 points and completely invalidate both comparison and interpretation of results.  

You can read more about our reproducibility study of neural NER in this [blog article](http://science-miner.com/a-reproducibility-study-on-neural-ner/).

All reported scores bellow are __f-score__ for the CoNLL-2003 NER dataset. We report first the f-score averaged over 10 training runs, and second the best f-score over these 10 training runs. All the DeLFT trained models are included in this repository. 

| Architecture  | Implementation | Glove only (avg / best)| Glove + valid. set (avg / best)| ELMo + Glove (avg / best)| ELMo + Glove + valid. set (avg / best)|
| --- | --- | --- | --- | --- | --- |
| BidLSTM-CRF   | DeLFT | __90.75__ / __91.35__  | 91.13 / 91.60 | __92.47__ / __92.71__ | __92.69__ / __93.09__ | 
|               | [(Lample and al., 2016)](https://arxiv.org/abs/1603.01360) | - / 90.94 |      |              |               | 
| BidLSTM-CNN-CRF | DeLFT | 90.73 / 91.07| 91.01 / 91.26 | 92.30 / 92.57| 92.67 / 93.04 |
|               | [(Ma & Hovy, 2016)](https://arxiv.org/abs/1603.01354) |  - / 91.21  | | | |
|               | [(Peters & al. 2018)](https://arxiv.org/abs/1802.05365) |  | | 92.22** / - | |
| BidLSTM-CNN   | DeLFT | 89.23 / 89.47  | 89.35 / 89.87 | 91.66 / 92.00 | 92.01 / 92.16 |
|               | [(Chiu & Nichols, 2016)](https://arxiv.org/abs/1511.08308) || __90.88***__ / - | | |
| BidGRU-CRF    | DeLFT | 90.38 / 90.72  | 90.28 / 90.69 | 92.03 / 92.44 | 92.43 / 92.71 |
|               | [(Peters & al. 2017)](https://arxiv.org/abs/1705.00108) |  | |  | 91.93* / - |



_*_ reported f-score using Senna word embeddings and not Glove.

** f-score is averaged over 5 training runs. 

*** reported f-score with Senna word embeddings (Collobert 50d) averaged over 10 runs, including case features and not including lexical features. DeLFT implementation of the same architecture includes the capitalization features too, but uses the more efficient GloVe 300d embeddings.


##### Command Line Interface

Different datasets and languages are supported. They can be specified by the command line parameters. The general usage of the CLI is as follow: 

```
usage: nerTagger.py [-h] [--fold-count FOLD_COUNT] [--lang LANG]
                    [--dataset-type DATASET_TYPE]
                    [--train-with-validation-set]
                    [--architecture ARCHITECTURE] [--use-ELMo] [--use-BERT]
                    [--data-path DATA_PATH] [--file-in FILE_IN]
                    [--file-out FILE_OUT]
                    action

Neural Named Entity Recognizers

positional arguments:
  action                one of [train, train_eval, eval, tag]

optional arguments:
  -h, --help            show this help message and exit
  --fold-count FOLD_COUNT
                        number of folds or re-runs to be used when training
  --lang LANG           language of the model as ISO 639-1 code
  --dataset-type DATASET_TYPE
                        dataset to be used for training the model
  --train-with-validation-set
                        Use the validation set for training together with the
                        training set
  --architecture ARCHITECTURE
                        type of model architecture to be used, one of
                        [BidLSTM_CRF, BidLSTM_CNN, BidLSTM_CNN_CRF, BidGRU-
                        CRF]
  --use-ELMo            Use ELMo contextual embeddings
  --use-BERT            Use BERT extracted features (embeddings)
  --data-path DATA_PATH
                        path to the corpus of documents for training (only use
                        currently with Ontonotes corpus in orginal XML format)
  --file-in FILE_IN     path to a text file to annotate
  --file-out FILE_OUT   path for outputting the resulting JSON NER anotations
  --embedding EMBEDDING
                        The desired pre-trained word embeddings using their
                        descriptions in the file embedding-registry.json. Be
                        sure to use here the same name as in the registry
                        ('glove-840B', 'fasttext-crawl', 'word2vec'), and that
                        the path in the registry to the embedding file is
                        correct on your system.
```

More explanations and examples are presented in the following sections. 

##### CONLL 2003

DeLFT comes with various pre-trained models with the CoNLL-2003 NER dataset.

By default, the BidLSTM-CRF architecture is used. With this available model, glove-840B word embeddings, and optimisation of hyperparameters, the current f1 score on CoNLL 2003 _testb_ set is __91.35__ (best run over 10 training, using _train_ set for training and _testa_ for validation), as compared to the 90.94 reported in [1], or __90.75__ when averaged over 10 training. Best model f1 score becomes __91.60__ when using both _train_ and _testa_ (validation set) for training (best run over 10 training), as it is done by (Chiu & Nichols, 2016) or some recent works like (Peters and al., 2017).  

Using BidLSTM-CRF model with ELMo embeddings, following [5] and some parameter optimisations and [warm-up](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md#notes-on-statefulness-and-non-determinism), make the predictions around 30 times slower but improve the f1 score on CoNLL 2003 currently to __92.47__ (averaged over 10 training, __92.71__ for best model, using _train_ set for training and _testa_ for validation), or __92.69__ (averaged over 10 training, __93.09__ best model) when training with the validation set (as in the paper Peters and al., 2017).

For re-training a model, the CoNLL-2003 NER dataset (`eng.train`, `eng.testa`, `eng.testb`) must be present under `data/sequenceLabelling/CoNLL-2003/` in IOB2 tagging sceheme (look [here](https://github.com/Franck-Dernoncourt/NeuroNER/tree/master/data/conll2003/en) for instance ;) and [here](https://github.com/kermitt2/delft/tree/master/utilities). The CONLL 2003 dataset (English) is the default dataset and English is the default language, but you can also indicate it explicitly as parameter with `--dataset-type conll2003` and specifying explicitly the language `--lang en`.

For training and evaluating following the traditional approach (training with the train set without validation set, and evaluating on test set), use:

> python3 nerTagger.py --dataset-type conll2003 train_eval

To use ELMo contextual embeddings, add the parameter `--use-ELMo`. This will slow down considerably (30 times) the first epoch of the training, then the contextual embeddings will be cached and the rest of the training will be similar to usual embeddings in term of training time. Alternatively add `--use-BERT` to use BERT extracted features as contextual embeddings to the RNN architecture. 

> python3 nerTagger.py --dataset-type conll2003 --use-ELMo train_eval

Some recent works like (Chiu & Nichols, 2016) and (Peters and al., 2017) also train with the validation set, leading obviously to a better accuracy (still they compare their scores with scores previously reported trained differently, which is arguably a bit unfair - this aspect is mentioned in (Ma & Hovy, 2016)). To train with both train and validation sets, use the parameter `--train-with-validation-set`:

> python3 nerTagger.py --dataset-type conll2003 --train-with-validation-set train_eval

Note that, by default, the BidLSTM-CRF model is used. (Documentation on selecting other models and setting hyperparameters to be included here !)

For evaluating against CoNLL 2003 testb set with the existing model:

> python3 nerTagger.py --dataset-type conll2003 eval

```text
    Evaluation on test set:
        f1 (micro): 91.35
                 precision    recall  f1-score   support

            ORG     0.8795    0.9007    0.8899      1661
            PER     0.9647    0.9623    0.9635      1617
           MISC     0.8261    0.8120    0.8190       702
            LOC     0.9260    0.9305    0.9282      1668

    avg / total     0.9109    0.9161    0.9135      5648

```

If the model has been trained also with the validation set (`--train-with-validation-set`), similarly to (Chiu & Nichols, 2016) or (Peters and al., 2017), results are significantly better:

```text
    Evaluation on test set:
        f1 (micro): 91.60
                 precision    recall    f1-score    support

            LOC     0.9219    0.9418    0.9318      1668
           MISC     0.8277    0.8077    0.8176       702
            PER     0.9594    0.9635    0.9614      1617
            ORG     0.9029    0.8904    0.8966      1661

    avg / total     0.9158    0.9163    0.9160      5648
```

Using ELMo with the best model obtained over 10 training (not using the validation set for training, only for early stop):

```text
    Evaluation on test set:
        f1 (micro): 92.71
                      precision    recall  f1-score   support

                 PER     0.9787    0.9672    0.9729      1617
                 LOC     0.9368    0.9418    0.9393      1668
                MISC     0.8237    0.8319    0.8278       702
                 ORG     0.9072    0.9181    0.9126      1661

    all (micro avg.)     0.9257    0.9285    0.9271      5648

```

Using ELMo and training with the validation set gives a f-score of 93.09 (best model), 92.69 averaged over 10 runs (the best model is provided under `data/models/sequenceLabelling/ner-en-conll2003-BidLSTM_CRF/with_validation_set/`).

For training with all the available data:

> python3 nerTagger.py --dataset-type conll2003 train

To take into account the strong impact of random seed, you need to train multiple times with the n-folds options. The model will be trained n times with different seed values but with the same sets if the evaluation set is provided. The evaluation will then give the average scores over these n models (against test set) and for the best model which will be saved. For 10 times training for instance, use:

> python3 nerTagger.py --dataset-type conll2003 --fold-count 10 train_eval

After training a model, for tagging some text, for instance in a file `data/test/test.ner.en.txt` (), use the command:

> python3 nerTagger.py --dataset-type conll2003 --file-in data/test/test.ner.en.txt tag

Note that, currently, the input text file must contain one sentence per line, so the text must be presegmented into sentences. To obtain the JSON annotations in a text file instead than in the standard output, use the parameter `--file-out`. Predictions work at around 7400 tokens per second for the BidLSTM_CRF architecture with a GeForce GTX 1080 Ti. 

This produces a JSON output with entities, scores and character offsets like this:

```json
{
    "runtime": 0.34,
    "texts": [
        {
            "text": "The University of California has found that 40 percent of its students suffer food insecurity. At four state universities in Illinois, that number is 35 percent.",
            "entities": [
                {
                    "text": "University of California",
                    "endOffset": 32,
                    "score": 1.0,
                    "class": "ORG",
                    "beginOffset": 4
                },
                {
                    "text": "Illinois",
                    "endOffset": 134,
                    "score": 1.0,
                    "class": "LOC",
                    "beginOffset": 125
                }
            ]
        },
        {
            "text": "President Obama is not speaking anymore from the White House.",
            "entities": [
                {
                    "text": "Obama",
                    "endOffset": 18,
                    "score": 1.0,
                    "class": "PER",
                    "beginOffset": 10
                },
                {
                    "text": "White House",
                    "endOffset": 61,
                    "score": 1.0,
                    "class": "LOC",
                    "beginOffset": 49
                }
            ]
        }
    ],
    "software": "DeLFT",
    "date": "2018-05-02T12:24:55.529301",
    "model": "ner"
}

```

If you have trained the model with ELMo, you need to indicate to use ELMo-based model when annotating with the parameter `--use-ELMo` (note that the runtime impact is important as compared to traditional embeddings): 

> python3 nerTagger.py --dataset-type conll2003 --use-ELMo --file-in data/test/test.ner.en.txt tag

For English NER tagging, the default static embeddings is Glove (`glove-840B`). Other static embeddings can be specified with the parameter `--embedding`, for instance:

> python3 nerTagger.py --dataset-type conll2003 --embedding word2vec train_eval


##### Ontonotes 5.0 CONLL 2012

DeLFT comes with pre-trained models with the [Ontonotes 5.0 CoNLL-2012 NER dataset](http://cemantix.org/data/ontonotes.html). As dataset-type identifier, use `conll2012`. All the options valid for CoNLL-2003 NER dataset are usable for this dataset. Default static embeddings for Ontonotes are `fasttext-crawl`, which can be changed with parameter `--embedding`.

With the default BidLSTM-CRF architecture, FastText embeddings and without any parameter tuning, f1 score is __86.65__ averaged over these 10 trainings, with best run at  __87.01__ (provided model) when trained with the train set strictly. 

With ELMo, f-score is __88.66__ averaged over these 10 trainings, and with best best run at __89.01__.

For re-training, the assembled Ontonotes datasets following CoNLL-2012 must be available and converted into IOB2 tagging scheme, see [here](https://github.com/kermitt2/delft/tree/master/utilities) for more details. To train and evaluate following the traditional approach (training with the train set without validation set, and evaluating on test set), use:

> python3 nerTagger.py --dataset-type conll2012 train_eval

```text
Evaluation on test set:
	f1 (micro): 87.01
                  precision    recall  f1-score   support

            DATE     0.8029    0.8695    0.8349      1602
        CARDINAL     0.8130    0.8139    0.8135       935
          PERSON     0.9061    0.9371    0.9214      1988
             GPE     0.9617    0.9411    0.9513      2240
             ORG     0.8799    0.8568    0.8682      1795
           MONEY     0.8903    0.8790    0.8846       314
            NORP     0.9226    0.9501    0.9361       841
         ORDINAL     0.7873    0.8923    0.8365       195
            TIME     0.5772    0.6698    0.6201       212
     WORK_OF_ART     0.6000    0.5060    0.5490       166
             LOC     0.7340    0.7709    0.7520       179
           EVENT     0.5000    0.5556    0.5263        63
         PRODUCT     0.6528    0.6184    0.6351        76
         PERCENT     0.8717    0.8567    0.8642       349
        QUANTITY     0.7155    0.7905    0.7511       105
             FAC     0.7167    0.6370    0.6745       135
        LANGUAGE     0.8462    0.5000    0.6286        22
             LAW     0.7308    0.4750    0.5758        40

all (micro avg.)     0.8647    0.8755    0.8701     11257
```

With ELMo embeddings (using the default hyper-parameters, except the batch size which is increased to better learn the less frequent classes):

```text
Evaluation on test set:
  f1 (micro): 89.01
                  precision    recall  f1-score   support

             LAW     0.7188    0.5750    0.6389        40
         PERCENT     0.8946    0.8997    0.8971       349
           EVENT     0.6212    0.6508    0.6357        63
        CARDINAL     0.8616    0.7722    0.8144       935
        QUANTITY     0.7838    0.8286    0.8056       105
            NORP     0.9232    0.9572    0.9399       841
             LOC     0.7459    0.7709    0.7582       179
            DATE     0.8629    0.8252    0.8437      1602
        LANGUAGE     0.8750    0.6364    0.7368        22
             GPE     0.9637    0.9607    0.9622      2240
         ORDINAL     0.8145    0.9231    0.8654       195
             ORG     0.9033    0.8903    0.8967      1795
           MONEY     0.8851    0.9076    0.8962       314
             FAC     0.8257    0.6667    0.7377       135
            TIME     0.6592    0.6934    0.6759       212
          PERSON     0.9350    0.9477    0.9413      1988
     WORK_OF_ART     0.6467    0.7169    0.6800       166
         PRODUCT     0.6867    0.7500    0.7170        76

all (micro avg.)     0.8939    0.8864    0.8901     11257
```

For ten model training with average, worst and best model with ELMo embeddings, use:

> python3 nerTagger.py --dataset-type conll2012 --use-ELMo --fold-count 10 train_eval

##### French model (based on Le Monde corpus)

Note that Le Monde corpus is subject to copyrights and is limited to research usage only. This is the default French model, so it will be used by simply indicating the language as parameter: `--lang fr`, but you can also indicate explicitly the dataset with `--dataset-type lemonde`. Default static embeddings for French language models are `wiki.fr`, which can be changed with parameter `--embedding`.

Similarly as before, for training and evaluating use:

> python3 nerTagger.py --lang fr train_eval

In practice, we need to repeat training and evaluation several times to neutralise random seed effects and to average scores, here ten times:

> python3 nerTagger.py --lang fr --fold-count 10 train_eval

The performance is as follow, with a f-score of __91.01__ averaged over 10 training:

```text
average over 10 folds
  macro f1 = 0.9100881012386587
  macro precision = 0.9048633201198737
  macro recall = 0.9153907496012759 

** Worst ** model scores - 

                  precision    recall  f1-score   support

      <location>     0.9467    0.9647    0.9556       368
   <institution>     0.8621    0.8333    0.8475        30
      <artifact>     1.0000    0.5000    0.6667         4
  <organisation>     0.9146    0.8089    0.8585       225
        <person>     0.9264    0.9522    0.9391       251
      <business>     0.8463    0.8936    0.8693       376

all (micro avg.)     0.9040    0.9083    0.9061      1254

** Best ** model scores - 

                  precision    recall  f1-score   support

      <location>     0.9439    0.9592    0.9515       368
   <institution>     0.8667    0.8667    0.8667        30
      <artifact>     1.0000    0.5000    0.6667         4
  <organisation>     0.8813    0.8578    0.8694       225
        <person>     0.9453    0.9641    0.9546       251
      <business>     0.8706    0.9122    0.8909       376

all (micro avg.)     0.9090    0.9242    0.9166      1254
```

With frELMo:

> python3 nerTagger.py --lang fr --fold-count 10 --use-ELMo train_eval

```text
average over 10 folds
    macro f1 = 0.9209397554337976
    macro precision = 0.91949107960079
    macro recall = 0.9224082934609251 

** Worst ** model scores - 

                  precision    recall  f1-score   support

  <organisation>     0.8704    0.8356    0.8526       225
        <person>     0.9344    0.9641    0.9490       251
      <artifact>     1.0000    0.5000    0.6667         4
      <location>     0.9173    0.9647    0.9404       368
   <institution>     0.8889    0.8000    0.8421        30
      <business>     0.9130    0.8936    0.9032       376

all (micro avg.)     0.9110    0.9147    0.9129      1254

** Best ** model scores - 

                  precision    recall  f1-score   support

  <organisation>     0.9061    0.8578    0.8813       225
        <person>     0.9416    0.9641    0.9528       251
      <artifact>     1.0000    0.5000    0.6667         4
      <location>     0.9570    0.9674    0.9622       368
   <institution>     0.8889    0.8000    0.8421        30
      <business>     0.9016    0.9255    0.9134       376

all (micro avg.)     0.9268    0.9290    0.9279      1254
```

For training with all the dataset without evaluation:

> python3 nerTagger.py --lang fr train

and for annotating some examples:

> python3 nerTagger.py --lang fr --file-in data/test/test.ner.fr.txt tag

```json
{
    "date": "2018-06-11T21:25:03.321818",
    "runtime": 0.511,
    "software": "DeLFT",
    "model": "ner-fr-lemonde",
    "texts": [
        {
            "entities": [
                {
                    "beginOffset": 5,
                    "endOffset": 13,
                    "score": 1.0,
                    "text": "Allemagne",
                    "class": "<location>"
                },
                {
                    "beginOffset": 57,
                    "endOffset": 68,
                    "score": 1.0,
                    "text": "Donald Trump",
                    "class": "<person>"
                }
            ],
            "text": "Or l’Allemagne pourrait préférer la retenue, de peur que Donald Trump ne surtaxe prochainement les automobiles étrangères."
        }
    ]
}

```

<p align="center">
    <img src="https://abstrusegoose.com/strips/muggle_problems.png">
</p>

This above work is licensed under a [Creative Commons Attribution-Noncommercial 3.0 United States License](http://creativecommons.org/licenses/by-nc/3.0/us/). 

#### GROBID models

DeLFT supports [GROBID](https://github.com/kermitt2/grobid) training data (originally for CRF) and GROBID feature matrix to be labelled. Default static embeddings for GROBID models are `glove-840B`, which can be changed with parameter `--embedding`. 

Train a model:

> python3 grobidTagger.py *name-of-model* train

where *name-of-model* is one of GROBID model (_date_, _affiliation-address_, _citation_, _header_, _name-citation_, _name-header_, ...), for instance:

> python3 grobidTagger.py date train

To segment the training data and eval on 10%:

> python3 grobidTagger.py *name-of-model* train_eval

For instance for the _date_ model:

> python3 grobidTagger.py date train_eval

```text
        Evaluation:
        f1 (micro): 96.41
                 precision    recall  f1-score   support

        <month>     0.9667    0.9831    0.9748        59
         <year>     1.0000    0.9844    0.9921        64
          <day>     0.9091    0.9524    0.9302        42

    avg / total     0.9641    0.9758    0.9699       165
```

For applying a model on some examples:

> python3 grobidTagger.py date tag

```json
{
    "runtime": 0.509,
    "software": "DeLFT",
    "model": "grobid-date",
    "date": "2018-05-23T14:18:15.833959",
    "texts": [
        {
            "entities": [
                {
                    "score": 1.0,
                    "endOffset": 6,
                    "class": "<month>",
                    "beginOffset": 0,
                    "text": "January"
                },
                {
                    "score": 1.0,
                    "endOffset": 11,
                    "class": "<year>",
                    "beginOffset": 8,
                    "text": "2006"
                }
            ],
            "text": "January 2006"
        },
        {
            "entities": [
                {
                    "score": 1.0,
                    "endOffset": 4,
                    "class": "<month>",
                    "beginOffset": 0,
                    "text": "March"
                },
                {
                    "score": 1.0,
                    "endOffset": 13,
                    "class": "<day>",
                    "beginOffset": 10,
                    "text": "27th"
                },
                {
                    "score": 1.0,
                    "endOffset": 19,
                    "class": "<year>",
                    "beginOffset": 16,
                    "text": "2001"
                }
            ],
            "text": "March the 27th, 2001"
        }
    ]
}
```

Similarly to the NER models, to use ELMo contextual embeddings, add the parameter `--use-ELMo`, e.g.:

> python3 grobidTagger.py citation --use-ELMo train_eval

Add the parameter `--use-BERT` to use BERT extracted features as contextual embeddings for the RNN architecture. 

Similarly to the NER models, for n-fold training (action `train_eval` only), specify the value of `n` with the parameter `--fold-count`, e.g.:

> python3 grobidTagger.py citation --fold-count=10 train_eval

By default the Grobid data to be used are the ones available under the `data/sequenceLabelling/grobid` subdirectory, but a Grobid data file can be provided by the parameter `--input`: 

> python3 grobidTagger.py *name-of-model* train --input *path-to-the-grobid-data-file-to-be-used-for-training*

or 

> python3 grobidTagger.py *name-of-model* train_Eval --input *path-to-the-grobid-data-file-to-be-used-for-training_and_eval_with_random_split*

The evaluation of a model with a specific Grobid data file can be performed using the `eval` action and specifying the data file with `--input`: 

> python3 grobidTagger.py citation eval --input *path-to-the-grobid-data-file-to-be-used-for-evaluation*


#### Insult recognition

A small experimental model for recognising insults and threats in texts, based on the Wikipedia comment from the Kaggle _Wikipedia Toxic Comments_ dataset, English only. This uses a small dataset labelled manually.

For training:

> python3 insultTagger.py train

By default training uses the whole train set.

Example of a small tagging test:

> python3 insultTagger.py tag

will produced (__socially offensive language warning!__) result like this:

```json
{
    "runtime": 0.969,
    "texts": [
        {
            "entities": [],
            "text": "This is a gentle test."
        },
        {
            "entities": [
                {
                    "score": 1.0,
                    "endOffset": 20,
                    "class": "<insult>",
                    "beginOffset": 9,
                    "text": "moronic wimp"
                },
                {
                    "score": 1.0,
                    "endOffset": 56,
                    "class": "<threat>",
                    "beginOffset": 54,
                    "text": "die"
                }
            ],
            "text": "you're a moronic wimp who is too lazy to do research! die in hell !!"
        }
    ],
    "software": "DeLFT",
    "date": "2018-05-14T17:22:01.804050",
    "model": "insult"
}
```

#### Creating your own model

As long your task is a sequence labelling of text, adding a new corpus and create an additional model should be straightfoward. If you want to build a model named `toto` based on labelled data in one of the supported format (CoNLL, TEI or GROBID CRF), create the subdirectory `data/sequenceLabelling/toto` and copy your training data under it.  

(To be completed)

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

also available (via TensorFlow): 

* `bert` or `scibert`: BERT (Bidirectional Encoder Representations from Transformers) architecture (classification corresponds to a fine tuning)

Note: by default the first 300 tokens of the text to be classified are used, which is largely enough for any _short text_ classification tasks and works fine with low profile GPU (for instance GeForce GTX 1050 Ti with 4 GB memory). For taking into account a larger portion of the text, modify the config model parameter `maxlen`. However, using more than 1000 tokens for instance requires a modern GPU with enough memory (e.g. 10 GB).

For all these RNN architectures, it is possible to use ELMo contextual embeddings (`--use-ELMo`) or BERT extracted features as embeddings (`--use-BERT`). The integration of BERT as an additional non-RNN architecture is done via TensorFlow, we do not mix Keras and TensorFlow layers. 

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

#### Citation classification

We use the dataset developed and presented by A. Athar in the following article:

[7] Awais Athar. "Sentiment Analysis of Citations using Sentence Structure-Based Features". Proceedings of the ACL 2011 Student Session, 81-87, 2011. http://www.aclweb.org/anthology/P11-3015

For a given scientific article, the task is to estimate if the occurrence of a bibliographical citation is positive, neutral or negative given its citation context. Note that the dataset, similarly to the Toxic Comment classification, is highly unbalanced (86% of the citations are neutral).

In this example, we formulate the problem as a 3 class regression (`negative`. `neutral`, `positive`). To train the model:

> python3 citationClassifier.py train

with n-folds:

> python3 citationClassifier.py train --fold-count 10

Training and evalation (ratio):

> python3 citationClassifier.py train_eval

<!-- which should produce the following evaluation (using the 2-layers Bidirectional GRU model `gru`):

eval before data generator
```
Evaluation on 896 instances:

Class: negative
    accuracy at 0.5 = 0.9665178571428571
    f-1 at 0.5 = 0.9665178571428571
    log-loss = 0.10193770380479757
    roc auc = 0.9085232470270055

Class: neutral
    accuracy at 0.5 = 0.8995535714285714
    f-1 at 0.5 = 0.8995535714285714
    log-loss = 0.2584601024897698
    roc auc = 0.8914776135848872

Class: positive
    accuracy at 0.5 = 0.9252232142857143
    f-1 at 0.5 = 0.9252232142857143
    log-loss = 0.20726886795593405
    roc auc = 0.8892779640954823

Macro-average:
    average accuracy at 0.5 = 0.9304315476190476
    average f-1 at 0.5 = 0.9304315476190476
    average log-loss = 0.18922222475016715
    average roc auc = 0.8964262749024584

Micro-average:
    average accuracy at 0.5 = 0.9304315476190482
    average f-1 at 0.5 = 0.9304315476190482
    average log-loss = 0.18922222475016712
    average roc auc = 0.9319196428571429
```    


```text
Evaluation on 896 instances:

Class: negative
    accuracy at 0.5 = 0.9654017857142857
    f-1 at 0.5 = 0.9654017857142857
    log-loss = 0.1056664130630102
    roc auc = 0.898580121703854

Class: neutral
    accuracy at 0.5 = 0.8939732142857143
    f-1 at 0.5 = 0.8939732142857143
    log-loss = 0.25354114470640177
    roc auc = 0.88643347739321

Class: positive
    accuracy at 0.5 = 0.9185267857142857
    f-1 at 0.5 = 0.9185267857142856
    log-loss = 0.1980544119553914
    roc auc = 0.8930591175116723

Macro-average:
    average accuracy at 0.5 = 0.9259672619047619
    average f-1 at 0.5 = 0.9259672619047619
    average log-loss = 0.18575398990826777
    average roc auc = 0.8926909055362455

Micro-average:
    average accuracy at 0.5 = 0.9259672619047624
    average f-1 at 0.5 = 0.9259672619047624
    average log-loss = 0.18575398990826741
    average roc auc = 0.9296875


```

In [7], based on a SVM (linear kernel) and custom features, the author reports a F-score of 0.898 for micro-average and 0.764 for macro-average. As we can observe, a non-linear deep learning approach, even without any feature engineering nor tuning, is very robust for an unbalanced dataset and provides higher accuracy.
-->

To classify a set of citation contexts:

> python3 citationClassifier.py classify

which will produce some JSON output like this:

```json
{
    "model": "citations",
    "date": "2018-05-13T16:06:12.995944",
    "software": "DeLFT",
    "classifications": [
        {
            "negative": 0.001178970211185515,
            "text": "One successful strategy [15] computes the set-similarity involving (multi-word) keyphrases about the mentions and the entities, collected from the KG.",
            "neutral": 0.187219500541687,
            "positive": 0.8640883564949036
        },
        {
            "negative": 0.4590276777744293,
            "text": "Unfortunately, fewer than half of the OCs in the DAML02 OC catalog (Dias et al. 2002) are suitable for use with the isochrone-fitting method because of the lack of a prominent main sequence, in addition to an absence of radial velocity and proper-motion data.",
            "neutral": 0.3570767939090729,
            "positive": 0.18021513521671295
        },
        {
            "negative": 0.0726129561662674,
            "text": "However, we found that the pairwise approach LambdaMART [41] achieved the best performance on our datasets among most learning to rank algorithms.",
            "neutral": 0.12469841539859772,
            "positive": 0.8224021196365356
        }
    ],
    "runtime": 1.202
}

```

## TODO

__Embeddings__:

* use/experiment more with OOV mechanisms

* train decent French embeddings (ELMo)

__Models__:

* add BERT transformer architecture (non just the extracted features as embeddings as now)

* test Theano as alternative backend (waiting for Apache MXNet...)

* augment word vectors with features, in particular layout features generated by GROBID

* review/rewrite the current Linear Chain CRF layer that we are using, this Keras CRF implementation is (i) a runtime bottleneck, we could try to use Cython for improving runtime and (ii) the viterbi decoding is incomplete, it does not outputing final decoded label scores and it can't output n-best. 

__NER__:

* complete the benchmark with OntoNotes 5 - other languages

* align the CoNLL corpus tokenisation (CoNLL corpus is "pre-tokenised", but we might not want to follow this particular tokenisation)

__Production stack__:

* improve runtime

__Build more models and examples__...

* model for entity disambiguation

* dependency parser

## Acknowledgments

* Keras CRF implementation by Philipp Gross

* The evaluations for sequence labelling are based on a modified version of https://github.com/chakki-works/seqeval

* The preprocessor of the sequence labelling part is derived from https://github.com/Hironsan/anago/

* [ELMo](https://allennlp.org/elmo) contextual embeddings are developed by the [AllenNLP](https://allennlp.org) team and we use the TensorFlow library [bilm-tf](https://github.com/allenai/bilm-tf) for integrating them into DeLFT.

## License and contact

Distributed under [Apache 2.0 license](http://www.apache.org/licenses/LICENSE-2.0). The dependencies used in the project are either themselves also distributed under Apache 2.0 license or distributed under a compatible license.

Contact: Patrice Lopez (<patrice.lopez@science-miner.com>)
