<img align="right" width="150" height="150" src="doc/cat-delft-small.jpg">

[![Build Status](https://travis-ci.org/kermitt2/delft.svg?branch=master)](https://travis-ci.org/kermitt2/delft)
[![PyPI version](https://badge.fury.io/py/delft.svg)](https://badge.fury.io/py/delft)
[![SWH](https://archive.softwareheritage.org/badge/origin/https://github.com/kermitt2/delft/)](https://archive.softwareheritage.org/browse/origin/https://github.com/kermitt2/delft/)
[![License](http://img.shields.io/:license-apache-blue.svg)](http://www.apache.org/licenses/LICENSE-2.0.html)



# DeLFT

__DeLFT__ (**De**ep **L**earning **F**ramework for **T**ext) is a Keras and TensorFlow framework for text processing, focusing on sequence labelling (e.g. named entity tagging, information extraction) and text classification (e.g. comment classification). This library re-implements standard state-of-the-art Deep Learning architectures relevant to text processing tasks.  

DeLFT has three main purposes: 

1. __Usefulness__, by targeting the most common textual content used by humans to communicate, which is not just simple text as considered usually by existing Deep Learning works in NLP, but _rich text_ where tokens are associated to layout information (font. style, etc.), positions in structured documents, and possibly other lexical or symbolic contextual information. Such rich text is also usually coming from large documents like PDF or HTML, and not just text segments like sentences or paragraphs.

2. __Reproducibility and benchmarking__, by implementing several state-of-the-art algorithms for both sequence labelling and text classification tasks, including the usage of ELMo contextualised embeddings and BERT transformer architecture, offering the capacity to validate reported results and to benchmark several methods under the same conditions and criteria.

3. __Production level__, by offering optimzed performance, robustness and integration possibilities, which can support better engineering decisions and successful production-level applications. 

Some key elements include: 

* Reduction of the size of RNN models, in particular by removing word embeddings from them. For instance, the model for the toxic comment classifier went down from a size of 230 MB with embeddings to 1.8 MB. In practice the size of all the models of DeLFT is less than 2 MB, except for Ontonotes 5.0 NER model which is 4.7 MB.

* Implementation of a generic support of features. 

* Usage of dynamic data generator so that the training data do not need to stand completely in memory.

* Efficiently loading and management of an unlimited volume of pre-trained embeddings.

* A comprehensive evaluation framework with the standard metrics for sequence labeling and classification tasks, including n-fold cross validation. 

A native Java integration of the library has been realized in [GROBID](https://github.com/kermitt2/grobid) via [JEP](https://github.com/ninia/jep).

DeLFT has been tested with python 3.5 and 3.6, Keras 2.2 and Tensorflow 1.7+ as backend. As always, GPU(s) are required for decent training time: a GeForce GTX 1050 Ti for instance is absolutely fine without ELMo contextual embeddings. Using ELMo or BERT Base model is fine with a GeForce GTX 1080 Ti.

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

DeLFT uses tensorflow 1.12 as backend, and will exploit your available GPU with the condition that CUDA (>=8.0) is properly installed. 

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

The first time DeLFT starts and accesses pre-trained embeddings, these embeddings are serialised and stored in a LMDB database, a very efficient embedded database using memory-mapped file (already used in the Machine Learning world by Caffe and Torch for managing large training data). The next time these embeddings will be accessed, they will be immediately available.

Our approach solves the bottleneck problem pointed for instance [here](https://spenai.org/bravepineapple/faster_em/) in a much better way than quantising+compression or pruning. After being compiled and stored at the first access, any volume of embeddings vectors can be used immediately without any loading, with a negligible usage of memory, without any accuracy loss and with a negligible impact on runtime when using SSD. In practice, we can exploit for instance embeddings for dozen languages simultaneously, without any memory and runtime issues - a requirement for any ambitious industrial deployment of a neural NLP system. 

For instance, in a traditional approach `glove-840B` takes around 2 minutes to load and 4GB in memory. Managed with LMDB, after a first load time of around 4 minutes, `glove-840B` can be accessed immediately and takes only a couple MB in memory, for an impact on runtime negligible (around 1% slower) for any further command line calls.

By default, the LMDB databases are stored under the subdirectory `data/db`. The size of a database is roughly equivalent to the size of the original uncompressed embeddings file. To modify this path, edit the file `embedding-registry.json` and change the value of the attribute `embedding-lmdb-path`.

To get FastText .bin format support please uncomment the package `fasttextmirror==0.8.22` in `requirements.txt` or `requirements-gpu.txt` according to your system's configuration. Please note that the **.bin format is not supported on Windows platforms**. Installing the FastText .bin format support introduces the following additional dependencies:

* (gcc-4.8 or newer) or (clang-3.3 or newer)
* [Python](https://www.python.org/) version 2.7 or >=3.4
* [pybind11](https://github.com/pybind/pybind11)

While FastText .bin format are supported by DeLFT (including using ngrams for OOV words), this format will be loaded entirely in memory and does not take advantage of our memory-efficient management of embeddings.

> I have plenty of memory on my machine, I don't care about load time because I need to grab a coffee every ten minutes, I only process one language at the time, so I am not interested in taking advantage of the LMDB emebedding management !

Ok, ok, then set the `embedding-lmdb-path` value to `"None"` in the file `embedding-registry.json`, the embeddings will be loaded in memory as immutable data, like in the usual Keras scripts.

## Sequence Labelling

### Available models

The following DL architectures are supported by DeLFT:

* __BidLSTM-CRF__ with words and characters input following:

&nbsp;&nbsp;&nbsp;&nbsp; [1] Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer. "Neural Architectures for Named Entity Recognition". Proceedings of NAACL 2016. https://arxiv.org/abs/1603.01360

* __BidLSTM_CRF_FEATURES__ same as above, with generic feature channel (feature matrix can be provided in the usual CRF++/Wapiti/YamCha format).

* __BidLSTM-CNN__ with words, characters and custom casing features input, see:

&nbsp;&nbsp;&nbsp;&nbsp; [2] Jason P. C. Chiu, Eric Nichols. "Named Entity Recognition with Bidirectional LSTM-CNNs". 2016. https://arxiv.org/abs/1511.08308

* __BidLSTM-CNN-CRF__ with words, characters and custom casing features input following:

&nbsp;&nbsp;&nbsp;&nbsp; [3] Xuezhe Ma and Eduard Hovy. "End-to-end Sequence Labelling via Bi-directional LSTM-CNNs-CRF". 2016. https://arxiv.org/abs/1603.01354

* __BidGRU-CRF__, similar to: 

&nbsp;&nbsp;&nbsp;&nbsp; [4] Matthew E. Peters, Waleed Ammar, Chandra Bhagavatula, Russell Power. "Semi-supervised sequence tagging with bidirectional language models". 2017. https://arxiv.org/pdf/1705.00108  

* __BERT__ transformer architecture, with fine-tuning and a CRF as activation layer, adapted to sequence labeling. Any pre-trained TensorFlow BERT models can be used (e.g. SciBERT or BioBERT for scientific and medical texts). 

&nbsp;&nbsp;&nbsp;&nbsp; [6] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova, BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. 2018. https://arxiv.org/abs/1810.04805

In addition, the following contextual embeddings can be used in combination to the RNN architectures: 

* [__ELMo__](https://allennlp.org/elmo) contextualised embeddings, which lead to the state of the art (92.22% F1 on CoNLL2003 NER dataset, averaged over five runs), when combined with _BidLSTM-CRF_ with , see:

&nbsp;&nbsp;&nbsp;&nbsp; [5] Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer. "Deep contextualized word representations". 2018. https://arxiv.org/abs/1802.05365

* __BERT__ feature extraction to be used as contextual embeddings (as ELMo alternative), as explained in section 5.4 of: 

&nbsp;&nbsp;&nbsp;&nbsp; [6] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova, BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. 2018. https://arxiv.org/abs/1810.04805

Note that all our annotation data for sequence labelling follows the [IOB2](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)) scheme and we did not find any advantages to add alternative labelling scheme after experiments.

### Examples

#### NER

##### Overview

We have reimplemented in DeLFT the main neural architectures for NER of the last four years and performed a reproducibility analysis of the these systems with comparable evaluation criterias. Unfortunaltely, in publications, systems are usually compared directly with reported results obtained in different settings, which can bias scores by more than 1.0 point and completely invalidate both comparison and interpretation of results.  

You can read more about our reproducibility study of neural NER in this [blog article](http://science-miner.com/a-reproducibility-study-on-neural-ner/). This effort is similar to the work of [(Yang and Zhang, 2018)](https://arxiv.org/pdf/1806.04470.pdf) (see also [NCRFpp](https://github.com/jiesutd/NCRFpp)) but has also been extended to BERT for a fair comparison of RNN for sequence labeling, and can also be related to the motivations of [(Pressel et al., 2018)](http://aclweb.org/anthology/W18-2506) [MEAD](https://github.com/dpressel/mead-baseline). 

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

Results with BERT fine-tuning, including a final CRF activation layer, instead of a softmax (a CRF activation layer improves f-score in average by +0.30 for sequence labelling task): 

| Architecture  | Implementation | f-score |
| --- | --- | --- | 
| bert-base-en    | DeLFT | 90.9 |  
| bert-base-en+CRF    | DeLFT | 91.2 |  
| bert-base-en        | [(Devlin & al. 2018)](https://arxiv.org/abs/1810.04805) | 92.4 |

For DeLFT, the average is obtained with 10 training runs (see [full results](https://github.com/kermitt2/delft/pull/78#issuecomment-569493805)) and for (Devlin & al. 2018) averaged with 5 runs. As noted [here](https://github.com/google-research/bert/issues/223), the original CoNLL-2003 NER results with BERT reported by the Google Research paper are not reproducible, and the score obtained by DeLFT is very similar to those obtained by all the systems having reproduced this experiment (the original paper probably reported token-level metrics instead of the usual entity-level metrics, giving in our humble opinion a misleading conclusion about the performance of transformers for sequence labelling tasks). 

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
                        ['BidLSTM_CRF', 'BidLSTM_CRF_FEATURES', 'BidLSTM_CNN_CRF', 
                        'BidLSTM_CNN_CRF', 'BidGRU_CRF', 'BidLSTM_CNN', 
                        'BidLSTM_CRF_CASING', 'bert-base-en', 'bert-base-en', 
                        'scibert', 'biobert']
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

DeLFT comes with various trained models for the CoNLL-2003 NER dataset.

By default, the BidLSTM-CRF architecture is used. With this available model, glove-840B word embeddings, and optimisation of hyperparameters, the current f1 score on CoNLL 2003 _testb_ set is __91.35__ (best run over 10 training, using _train_ set for training and _testa_ for validation), as compared to the 90.94 reported in [1], or __90.75__ when averaged over 10 training. Best model f1 score becomes __91.60__ when using both _train_ and _testa_ (validation set) for training (best run over 10 training), as it is done by (Chiu & Nichols, 2016) or some recent works like (Peters and al., 2017).  

Using BidLSTM-CRF model with ELMo embeddings, following [5] and some parameter optimisations and [warm-up](https://github.com/allenai/allennlp/blob/master/docs/tutorials/how_to/elmo.md#notes-on-statefulness-and-non-determinism), make the predictions around 30 times slower but improve the f1 score on CoNLL 2003 currently to __92.47__ (averaged over 10 training, __92.71__ for best model, using _train_ set for training and _testa_ for validation), or __92.69__ (averaged over 10 training, __93.09__ best model) when training with the validation set (as in the paper Peters and al., 2017).

For re-training a model, the CoNLL-2003 NER dataset (`eng.train`, `eng.testa`, `eng.testb`) must be present under `data/sequenceLabelling/CoNLL-2003/` in IOB2 tagging sceheme (look [here](https://github.com/Franck-Dernoncourt/NeuroNER/tree/4cbfc3a1b4c4a5242e1cfbaea48d6f7e972e8881/data/conll2003/en) for instance ;) and [here](https://github.com/kermitt2/delft/tree/master/delft/utilities). The CONLL 2003 dataset (English) is the default dataset and English is the default language, but you can also indicate it explicitly as parameter with `--dataset-type conll2003` and specifying explicitly the language `--lang en`.

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

Using BERT architecture for sequence labelling (pre-trained transformer with fine-tuning), for instance here the `bert-base-en`, cased, pre-trained model, use:

> python3 nerTagger.py --architecture bert-base-en --dataset-type conll2003 --fold-count 10 train_eval

```text
average over 10 folds
            precision    recall  f1-score   support

       ORG     0.8804    0.9114    0.8957      1661
      MISC     0.7823    0.8189    0.8002       702
       PER     0.9633    0.9576    0.9605      1617
       LOC     0.9290    0.9316    0.9303      1668

  macro f1 = 0.9120
  macro precision = 0.9050
  macro recall = 0.9191

```

For training with all the available data:

> python3 nerTagger.py --dataset-type conll2003 train

To take into account the strong impact of random seed, you need to train multiple times with the n-folds options. The model will be trained n times with different seed values but with the same sets if the evaluation set is provided. The evaluation will then give the average scores over these n models (against test set) and for the best model which will be saved. For 10 times training for instance, use:

> python3 nerTagger.py --dataset-type conll2003 --fold-count 10 train_eval

After training a model, for tagging some text, for instance in a file `data/test/test.ner.en.txt` (), use the command:

> python3 nerTagger.py --dataset-type conll2003 --file-in data/test/test.ner.en.txt tag

For instance for tagging the text with a specific architecture: 

> python3 nerTagger.py --dataset-type conll2003 --file-in data/test/test.ner.en.txt --architecture bert-base-en tag

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

For re-training, the assembled Ontonotes datasets following CoNLL-2012 must be available and converted into IOB2 tagging scheme, see [here](https://github.com/kermitt2/delft/tree/master/delft/utilities) for more details. To train and evaluate following the traditional approach (training with the train set without validation set, and evaluating on test set), use:

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

Note that Le Monde corpus is subject to copyrights and is limited to research usage only, it is usually referred to as "corpus FTB". The corpus file `ftb6_ALL.EN.docs.relinked.xml` must be located under `delft/data/sequenceLabelling/leMonde/`. This is the default French model, so it will be used by simply indicating the language as parameter: `--lang fr`, but you can also indicate explicitly the dataset with `--dataset-type ftb`. Default static embeddings for French language models are `wiki.fr`, which can be changed with parameter `--embedding`.

Similarly as before, for training and evaluating use:

> python3 nerTagger.py --lang fr --dataset-type ftb train_eval

In practice, we need to repeat training and evaluation several times to neutralise random seed effects and to average scores, here ten times:

> python3 nerTagger.py --lang fr --dataset-type ftb --fold-count 10 train_eval

The performance is as follow, for the BiLSTM-CRF architecture and fasttext `wiki.fr` embeddings, with a f-score of __91.01__ averaged over 10 training:

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

> python3 nerTagger.py --lang fr --dataset-type ftb --fold-count 10 --use-ELMo train_eval

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

For historical reason, we can also consider a particular split of the FTB corpus into train, dev and set set and with a forced tokenization (like the old CoNLL 2013 NER), that was used in previous work for comparison. Obviously the evaluation is dependent to this particular set and the n-fold cross validation is a much better practice and should be prefered (as well as a format that do not force a tokenization). For using the forced split FTB (using the files `ftb6_dev.conll`, `ftb6_test.conll` and `ftb6_train.conll` located under `delft/data/sequenceLabelling/leMonde/`), use as parameter `--dataset-type ftb_force_split`:

> python3 nerTagger.py --lang fr --dataset-type ftb_force_split --fold-count 10 train_eval

which gives for the BiLSTM-CRF architecture and fasttext `wiki.fr` embeddings, a f-score of __86.37__ averaged over 10 training:

```
average over 10 folds
                    precision    recall  f1-score   support

      Organization     0.8410    0.7431    0.7888       311
            Person     0.9086    0.9327    0.9204       205
          Location     0.9219    0.9144    0.9181       347
           Company     0.8140    0.8603    0.8364       290
  FictionCharacter     0.0000    0.0000    0.0000         2
           Product     1.0000    1.0000    1.0000         3
               POI     0.0000    0.0000    0.0000         0
           company     0.0000    0.0000    0.0000         0

  macro f1 = 0.8637
  macro precision = 0.8708
  macro recall = 0.8567 


** Worst ** model scores -
                  precision    recall  f1-score   support

    Organization     0.8132    0.7138    0.7603       311
        Location     0.9152    0.9020    0.9086       347
         Company     0.7926    0.8172    0.8048       290
          Person     0.9095    0.9317    0.9205       205
         Product     1.0000    1.0000    1.0000         3
FictionCharacter     0.0000    0.0000    0.0000         2

all (micro avg.)     0.8571    0.8342    0.8455      1158


** Best ** model scores -
                  precision    recall  f1-score   support

    Organization     0.8542    0.7910    0.8214       311
        Location     0.9226    0.9280    0.9253       347
         Company     0.8212    0.8552    0.8378       290
          Person     0.9095    0.9317    0.9205       205
         Product     1.0000    1.0000    1.0000         3
FictionCharacter     0.0000    0.0000    0.0000         2

all (micro avg.)     0.8767    0.8722    0.8745      1158
```

With frELMo:

> python3 nerTagger.py --lang fr --dataset-type ftb_force_split --fold-count 10 --use-ELMo train_eval

```
average over 10 folds
                    precision    recall  f1-score   support

      Organization     0.8605    0.7752    0.8155       311
            Person     0.9227    0.9371    0.9298       205
          Location     0.9281    0.9432    0.9356       347
           Company     0.8401    0.8779    0.8585       290
  FictionCharacter     0.1000    0.0500    0.0667         2
           Product     0.8750    1.0000    0.9286         3
               POI     0.0000    0.0000    0.0000         0
           company     0.0000    0.0000    0.0000         0

  macro f1 = 0.8831
  macro precision = 0.8870
  macro recall = 0.8793 


** Worst ** model scores -
                  precision    recall  f1-score   support

        Location     0.9366    0.9366    0.9366       347
    Organization     0.8309    0.7428    0.7844       311
          Person     0.9268    0.9268    0.9268       205
         Company     0.8179    0.8828    0.8491       290
         Product     0.7500    1.0000    0.8571         3
FictionCharacter     0.0000    0.0000    0.0000         2

all (micro avg.)     0.8762    0.8679    0.8720      1158


** Best ** model scores -
                  precision    recall  f1-score   support

        Location     0.9220    0.9539    0.9377       347
    Organization     0.8777    0.7846    0.8285       311
          Person     0.9187    0.9366    0.9275       205
         Company     0.8444    0.9172    0.8793       290
         Product     1.0000    1.0000    1.0000         3
FictionCharacter     0.0000    0.0000    0.0000         2

all (micro avg.)     0.8900    0.8946    0.8923      1158
```

For the `ftb_force_split` dataset, similarly as for CoNLL 2013, you can use the `train_with_validation_set` parameter to add the validation set in the training data. The above results are all obtained without using `train_with_validation_set` (which is the common approach).

Finally, for training with all the dataset without evaluation (e.g. for production):

> python3 nerTagger.py --lang fr --dataset-type ftb train

and for annotating some examples:

> python3 nerTagger.py --lang fr --dataset-type ftb --file-in data/test/test.ner.fr.txt tag

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

As usual, the architecture to be used for the indicated model can be specified with the `--architecture` parameter:

> python3 grobidTagger.py citation train_eval --architecture BidLSTM_CRF_FEATURES

With the architectures having a feature channel, the categorial features (as generated by GROBID) will be automatically selected (typically the layout and lexical class features). The models not having a feature channel will only use the tokens as input (as the usual Deep Learning models for text). 

Similarly to the NER models, to use ELMo contextual embeddings, add the parameter `--use-ELMo`, e.g.:

> python3 grobidTagger.py citation --use-ELMo train_eval

Add the parameter `--use-BERT` to use BERT extracted features as contextual embeddings for the RNN architecture. 

Similarly to the NER models, for n-fold training (action `train_eval` only), specify the value of `n` with the parameter `--fold-count`, e.g.:

> python3 grobidTagger.py citation --fold-count=10 train_eval

By default the Grobid data to be used are the ones available under the `data/sequenceLabelling/grobid` subdirectory, but a Grobid data file can be provided by the parameter `--input`: 

> python3 grobidTagger.py *name-of-model* train --input *path-to-the-grobid-data-file-to-be-used-for-training*

or 

> python3 grobidTagger.py *name-of-model* train_eval --input *path-to-the-grobid-data-file-to-be-used-for-training_and_eval_with_random_split*

The evaluation of a model with a specific Grobid data file can be performed using the `eval` action and specifying the data file with `--input`: 

> python3 grobidTagger.py citation eval --input *path-to-the-grobid-data-file-to-be-used-for-evaluation*


The evaluation of a model can be performed calling 

> python3 grobidTagger.py citation eval --input evaluation_data


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

Training and evalation (ratio) with 10-folds:

> python3 citationClassifier.py train_eval --fold-count 10

which should produce the following evaluation (using the 2-layers Bidirectional GRU model `gru`):

```
Evaluation on 896 instances:
                   precision        recall       f-score       support
      negative        0.1494        0.4483        0.2241            29
       neutral        0.9653        0.8058        0.8784           793
      positive        0.3333        0.6622        0.4434            74
```

Similarly as other scripts, use `--architecture` to specify an alternative DL architecture, for instance SciBERT:

> python3 citationClassifier.py train_eval --architecture scibert

```
Evaluation on 896 instances:
                   precision        recall       f-score       support
      negative        0.1712        0.6552        0.2714            29
       neutral        0.9740        0.8020        0.8797           793
      positive        0.4015        0.7162        0.5146            74
```

To classify a set of citation contexts with default model (2-layers Bidirectional GRU model `gru`):

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

__Models__:

* The integration of FLAIR contextual embeddings (branch `flair` and `flair2`) raised several issues and we did not manage to reproduce the results from the full FLAIR implementation. We should experiment with https://github.com/kensho-technologies/bubs, a Keras/TensorFlow reimplementation of the Flair Contextualized Embeddings.

* Try to migrate to TF 2.0 and tf.keras

* Review/rewrite the current Linear Chain CRF layer that we are using, this Keras CRF implementation is (i) a runtime bottleneck, we could try to use Cython for improving runtime and (ii) the viterbi decoding is incomplete, it does not outputing final decoded label scores and it can't output n-best. 

* Port everything to Apache MXNet? :)

__NER__:

* complete the benchmark with OntoNotes 5 - other languages

* align the CoNLL corpus tokenisation (CoNLL corpus is "pre-tokenised", but we might not want to follow this particular tokenisation)

__Production__:

* automatic download of embeddings on demand

* improve runtime

__Build more models and examples__...

* Model for entity disambiguation (deeptype for entity-fishing)

* Relation extractions (in particular with medical texts)

Note that we are focusing on sequence labelling/information extraction and text classification tasks, which are our main applications, and not on text understanding and machine translation which are the object of already many other Open Source frameworks. 

## Acknowledgments

* Keras CRF implementation by Philipp Gross

* The evaluations for sequence labelling are based on a modified version of https://github.com/chakki-works/seqeval

* The preprocessor of the sequence labelling part is derived from https://github.com/Hironsan/anago/

* [ELMo](https://allennlp.org/elmo) contextual embeddings are developed by the [AllenNLP](https://allennlp.org) team and we use the TensorFlow library [bilm-tf](https://github.com/allenai/bilm-tf) for integrating them into DeLFT.

* [BERT](https://github.com/google-research/bert) transformer original implementation by Google Research, which has been adapted for text classification and sequence labelling in DeLFT.

* [FastPredict](https://github.com/marcsto/rl/blob/master/src/fast_predict2.py) from by Marc Stogaitis, adapted to our BERT usages. 


## License and contact

Distributed under [Apache 2.0 license](http://www.apache.org/licenses/LICENSE-2.0). The dependencies used in the project are either themselves also distributed under Apache 2.0 license or distributed under a compatible license.

Contact: Patrice Lopez (<patrice.lopez@science-miner.com>)

## How to cite

If you want to this work, please refer to the present GitHub project, together with the [Software Heritage](https://www.softwareheritage.org/) project-level permanent identifier. For example, with BibTeX:

```bibtex
@misc{DeLFT,
    title = {DeLFT},
    howpublished = {\url{https://github.com/kermitt2/delft}},
    publisher = {GitHub},
    year = {2018--2020},
    archivePrefix = {swh},
    eprint = {1:dir:54eb292e1c0af764e27dd179596f64679e44d06e}
}
```
