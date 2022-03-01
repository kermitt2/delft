## Sequence Labelling

### Available models

The following DL architectures are supported by DeLFT:

* __BidLSTM_CRF__ with words and characters input following:

```
[1] Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer. "Neural Architectures for Named Entity Recognition". Proceedings of NAACL 2016. https://arxiv.org/abs/1603.01360
```

* __BidLSTM_CRF_FEATURES__ same as above, with generic feature channel (feature matrix can be provided in the usual CRF++/Wapiti/YamCha format).

* __BidLSTM_CNN__ with words, characters and custom casing features input, see:

```
[2] Jason P. C. Chiu, Eric Nichols. "Named Entity Recognition with Bidirectional LSTM-CNNs". 2016. https://arxiv.org/abs/1511.08308
```

* __BidLSTM_CNN_CRF__ with words, characters and custom casing features input following:


```
[3] Xuezhe Ma and Eduard Hovy. "End-to-end Sequence Labelling via Bi-directional LSTM-CNNs-CRF". 2016. https://arxiv.org/abs/1603.01354
```

* __BidGRU_CRF__, similar to: 

```
[4] Matthew E. Peters, Waleed Ammar, Chandra Bhagavatula, Russell Power. "Semi-supervised sequence tagging with bidirectional language models". 2017. https://arxiv.org/pdf/1705.00108  
```

* __BERT__ transformer architecture, with fine-tuning, adapted to sequence labeling. Any pre-trained BERT models can be used (e.g. DistilBERT, SciBERT or BioBERT for scientific and medical texts). 

```
[6] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova, BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. 2018. https://arxiv.org/abs/1810.04805
```

* __BERT_CRF__ transformer architecture, for fine-tuning and a CRF as final activation layer. Any pre-trained TensorFlow BERT models can be used (e.g. DistilBERT, SciBERT or BioBERT for scientific and medical texts). 

* __BERT_CRF_CHAR__ transformer architecture, for fine-tuning, with a character input channel and a CRF as final activation layer. The character input channel initializes character embeddings, which are then concatenated with BERT embeddings, followed by a bidirectional LSTM prior to the CRF layer.
Any pre-trained TensorFlow BERT models can be used. 

* __BERT_CRF_FEATURES__ transformer architecture, for fine-tuning, with a generic feature channel (feature matrix can be provided in the usual CRF++/Wapiti/YamCha format) and a CRF as final activation layer. Any pre-trained TensorFlow BERT models can be used. 

* __BERT_CRF_CHAR_FEATURES__ transformer architecture, for fine-tuning, with a character input channel, a generic feature channel and a CRF as final activation layer. Any pre-trained TensorFlow BERT models can be used. 

Note that all our annotation data for sequence labelling follows the [IOB2](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)) scheme and we did not find any advantages to add alternative labelling scheme after experiments.

### Creating your own model

As long your task is a sequence labelling of text, adding a new corpus and create an additional model should be straightfoward. If you want to build a model named `toto` based on labelled data in one of the supported format (CoNLL, TEI or GROBID CRF), create the subdirectory `data/sequenceLabelling/toto` and copy your training data under it.  

(To be completed)
