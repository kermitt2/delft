## Sequence Labelling

### Available models

The following DL architectures are supported by DeLFT:

* __BidLSTM_CRF__ with words and characters input following:

&nbsp;&nbsp;&nbsp;&nbsp; [1] Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer. "Neural Architectures for Named Entity Recognition". Proceedings of NAACL 2016. https://arxiv.org/abs/1603.01360

* __BidLSTM_CRF_FEATURES__ same as above, with generic feature channel (feature matrix can be provided in the usual CRF++/Wapiti/YamCha format).

* __BidLSTM_CNN__ with words, characters and custom casing features input, see:

&nbsp;&nbsp;&nbsp;&nbsp; [2] Jason P. C. Chiu, Eric Nichols. "Named Entity Recognition with Bidirectional LSTM-CNNs". 2016. https://arxiv.org/abs/1511.08308

* __BidLSTM_CNN_CRF__ with words, characters and custom casing features input following:

&nbsp;&nbsp;&nbsp;&nbsp; [3] Xuezhe Ma and Eduard Hovy. "End-to-end Sequence Labelling via Bi-directional LSTM-CNNs-CRF". 2016. https://arxiv.org/abs/1603.01354

* __BidGRU_CRF__, similar to: 

&nbsp;&nbsp;&nbsp;&nbsp; [4] Matthew E. Peters, Waleed Ammar, Chandra Bhagavatula, Russell Power. "Semi-supervised sequence tagging with bidirectional language models". 2017. https://arxiv.org/pdf/1705.00108  

* __BERT__ transformer architecture, with fine-tuning, adapted to sequence labeling. Any pre-trained BERT models can be used (e.g. DistilBERT, SciBERT or BioBERT for scientific and medical texts). 

&nbsp;&nbsp;&nbsp;&nbsp; [6] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova, BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. 2018. https://arxiv.org/abs/1810.04805

* __BERT_CRF__ transformer architecture, for fine-tuning and a CRF as final activation layer. Any pre-trained TensorFlow BERT models can be used (e.g. DistilBERT, SciBERT or BioBERT for scientific and medical texts). 

* __BERT_CRF_CHAR__ transformer architecture, for fine-tuning, with a character input channel and a CRF as final activation layer. The character input channel initializes character embeddings, which are then concatenated with BERT embeddings, followed by a bidirectional LSTM prior to the CRF layer.
Any pre-trained TensorFlow BERT models can be used. 

* __BERT_CRF_FEATURES__ transformer architecture, for fine-tuning, with a generic feature channel (feature matrix can be provided in the usual CRF++/Wapiti/YamCha format) and a CRF as final activation layer. Any pre-trained TensorFlow BERT models can be used. 

* __BERT_CRF_CHAR_FEATURES__ transformer architecture, for fine-tuning, with a character input channel, a generic feature channel and a CRF as final activation layer. Any pre-trained TensorFlow BERT models can be used. 

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
usage: delft/application/nerTagger.py [-h] [--fold-count FOLD_COUNT] [--lang LANG]
                    [--dataset-type DATASET_TYPE]
                    [--train-with-validation-set]
                    [--architecture ARCHITECTURE] 
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
  --data-path DATA_PATH
                        path to the corpus of documents for training (only use
                        currently with Ontonotes corpus in orginal XML format)
  --file-in FILE_IN     path to a text file to annotate
  --file-out FILE_OUT   path for outputting the resulting JSON NER anotations
  --embedding EMBEDDING
                        The desired pre-trained word embeddings using their
                        descriptions in the file delft/resources-registry.json. Be
                        sure to use here the same name as in the registry
                        ('glove-840B', 'fasttext-crawl', 'word2vec'), and that
                        the path in the registry to the embedding file is
                        correct on your system.
```


#### Creating your own model

As long your task is a sequence labelling of text, adding a new corpus and create an additional model should be straightfoward. If you want to build a model named `toto` based on labelled data in one of the supported format (CoNLL, TEI or GROBID CRF), create the subdirectory `data/sequenceLabelling/toto` and copy your training data under it.  

(To be completed)
