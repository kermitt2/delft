## Sequence Labelling

> ⚠️ **ELMo support was removed in DeLFT 0.4.x.** The ELMo bullet below is kept for historical reference (DeLFT 0.3.x and earlier). From 0.4.x onwards, transformer-based architectures (`BERT_CRF`, `BERT_ChainCRF`, …) cover the same use case.

### Available models

The following DL architectures are supported by DeLFT:

* __BidLSTM_CRF__ (CRF implementation based on recent tensorflow addons) or __BidLSTM_ChainCRF__ (CRF implementation from previous DeLFT version updated to tensorflow 2) with words and characters input following:

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

All RNN models (LSTM/GRU/CNN) can further uses ELMo contextualized embeddings to improve results:

* [__ELMo__](https://allennlp.org/elmo) contextualised embeddings, see:

```
[7] Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer. "Deep contextualized word representations". 2018. https://arxiv.org/abs/1802.05365
```

Note that all our annotation data for sequence labelling follows the [IOB2](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)) scheme and we did not find any advantages to add alternative labelling scheme after experiments.


### Creating your own model

As long as your task is sequence labelling, adding a new corpus and creating an additional model is straightforward. If you want to build a model named `toto` based on labelled data in one of the supported formats (CoNLL, TEI or GROBID CRF), create the subdirectory `data/sequenceLabelling/toto` and copy your training data under it.

The fastest path is to copy one of the existing application scripts in `delft/applications/` as a starting template:

- `delft/applications/nerTagger.py` — for token classification on plain CoNLL-style data (good template when your input is just `token<TAB>label` lines).
- `delft/applications/grobidTagger.py` — for sequence labelling with extra layout / categorical features (GROBID CRF feature matrix format).

In both, the model is created and trained through the high-level `Sequence` wrapper (`delft.sequenceLabelling.Sequence`):

```python
from delft.sequenceLabelling import Sequence

model = Sequence(
    "toto",
    architecture="BidLSTM_CRF",   # or "BERT_CRF", etc.
    embeddings_name="glove-840B", # for RNN architectures
    # transformer_name="bert-base-cased",  # for BERT_* architectures
)
model.train(x_train, y_train, x_valid=x_dev, y_valid=y_dev)
model.save("data/models/sequenceLabelling/toto")
```

Use the loaders in `delft/sequenceLabelling/reader.py` to read your training data; pick the one matching your file format:

- `load_data_and_labels_conll` — CoNLL-style `token<TAB>label` files
- `load_data_and_labels_crf_file` — GROBID CRF feature-matrix files
- `load_data_and_labels_xml_file` — TEI XML
- `load_data_and_labels_json_offsets` — JSON with character offsets
- `load_data_and_labels_ontonotes` — Ontonotes 5.0 corpus

After training, the model can be applied with `model.tag(...)` or via a CLI wrapper modelled on `nerTagger.py`'s `train` / `train_eval` / `tag` actions.
