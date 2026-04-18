# Creating GROBID models with DeLFT

[GROBID](https://github.com/kermitt2/grobid) uses a cascade of sequence labeling models to parse complete documents. The particularity of these models is to use joint text and layout fatures to identify document structures more accurately. The script `delft/applications/grobidTagger.py` allows the creation of various GROBID models to be used by the GROBID services for parsing various structures such as document headers, references, affiliations, authors, dates, etc.

## General command line for training GROBID models in DeLFT

```
usage: grobidTagger.py [-h] [--fold-count FOLD_COUNT] [--architecture ARCHITECTURE] [--use-ELMo]
                       [--embedding EMBEDDING] [--transformer TRANSFORMER] [--output OUTPUT]
                       [--input INPUT] [--feature-indices FEATURE_INDICES]
                       [--max-sequence-length MAX_SEQUENCE_LENGTH] [--batch-size BATCH_SIZE]
                       [--tensorboard]
                       model {train,train_eval,eval,tag}

Trainer for GROBID models using the DeLFT library

positional arguments:
  model                 Name of the model.
  {train,train_eval,eval,tag}

optional arguments:
  -h, --help            show this help message and exit
  --fold-count FOLD_COUNT
                        Number of fold to use when evaluating with n-fold cross validation.
  --architecture ARCHITECTURE
                        Type of model architecture to be used, one of ['BidLSTM', 'BidLSTM_CRF',
                        'BidLSTM_ChainCRF', 'BidLSTM_CNN_CRF', 'BidLSTM_CNN_CRF', 'BidGRU_CRF',
                        'BidLSTM_CNN', 'BidLSTM_CRF_CASING', 'BidLSTM_CRF_FEATURES',
                        'BidLSTM_ChainCRF_FEATURES', 'BERT', 'BERT_CRF', 'BERT_ChainCRF',
                        'BERT_CRF_FEATURES', 'BERT_CRF_CHAR', 'BERT_CRF_CHAR_FEATURES']
  --use-ELMo            Use ELMo contextual embeddings
  --embedding EMBEDDING
                        The desired pre-trained word embeddings using their descriptions in the
                        file. For local loading, use delft/resources-registry.json. Be sure to
                        use here the same name as in the registry, e.g. ['glove-840B',
                        'fasttext-crawl', 'word2vec'] and that the path in the registry to the
                        embedding file is correct on your system.
  --transformer TRANSFORMER
                        The desired pre-trained transformer to be used in the selected
                        architecture. For local loading use, delft/resources-registry.json, and
                        be sure to use here the same name as in the registry, e.g. ['bert-base-
                        cased', 'bert-large-cased', 'allenai/scibert_scivocab_cased'] and that
                        the path in the registry to the model path is correct on your system.
                        HuggingFace transformers hub will be used otherwise to fetch the model,
                        see https://huggingface.co/models for model names
  --output OUTPUT       Directory where to save a trained model.
  --input INPUT         Grobid data file to be used for training (train action), for training
                        and evaluation (train_eval action) or just for evaluation (eval action).
  --max-sequence-length MAX_SEQUENCE_LENGTH
                        max-sequence-length parameter to be used.
  --batch-size BATCH_SIZE
                        batch-size parameter to be used.
```


## GROBID models

DeLFT supports [GROBID](https://github.com/kermitt2/grobid) training data (originally for CRF) and GROBID feature matrix to be labelled. Default static embeddings for GROBID models are `glove-840B`, which can be changed with parameter `--embedding`. 

Train a model with all available training data:

```sh
python3  *name-of-model* train --architecture *name-of-architecture*
```

where *name-of-model* is one of GROBID model (_date_, _affiliation-address_, _citation_, _header_, _name-citation_, _name-header_, ...), for instance:

and where *name-of-architecture* is one of `['BidLSTM', 'BidLSTM_CRF', 'BidLSTM_ChainCRF', 'BidLSTM_CNN_CRF', 'BidLSTM_CNN_CRF', 'BidGRU_CRF', 'BidLSTM_CNN', 'BidLSTM_CRF_CASING', 'BidLSTM_CRF_FEATURES', 'BidLSTM_ChainCRF_FEATURES', 'BERT', 'BERT_CRF', 'BERT_ChainCRF', 'BERT_CRF_FEATURES', 'BERT_CRF_CHAR', 'BERT_CRF_CHAR_FEATURES']`.


```sh
python3 delft/applications/grobidTagger.py date train --architecture BidLSTM_CRF
```

To segment the training data and eval on 10%, use the action `train_eval` instead of `train`:

```sh
python3 delft/applications/grobidTagger.py *name-of-model* train_eval --architecture *name-of-architecture*
```

For instance for the _date_ model:

```sh
python3 delft/applications/grobidTagger.py date train_eval --architecture BidLSTM_CRF
```

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

```sh
python3 delft/applications/grobidTagger.py date tag --architecture BidLSTM_CRF
```

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

As usual, depending of the architecture to be used you can indicate wither which embeddings whould be used for a RNN model (default is glove-840B):

```sh
python3 delft/applications/grobidTagger.py citation train_eval --architecture BidLSTM_CRF_FEATURES --embedding glove-840B
```

or the name of the transformer model you wish use in an architecture including a transformer layer:

```sh
python3 delft/applications/grobidTagger.py header train --architecture BERT_CRF --transformer allenai/scibert_scivocab_cased
```

With the architectures having a feature channel, the categorial features (as generated by GROBID) will be automatically selected (typically the layout and lexical class features). The models not having a feature channel will only use the tokens as input (as the usual Deep Learning models for text). 

Similarly to the NER models, for n-fold training (action `train_eval` only), specify the value of `n` with the parameter `--fold-count`, e.g.:

```sh
python3 delft/applications/grobidTagger.py citation train_eval --architecture BidLSTM_CRF_FEATURES --fold-count=10 
```

By default the Grobid data to be used are the ones available under the `data/sequenceLabelling/grobid` subdirectory, but a Grobid data file can be provided by the parameter `--input`: 

```sh
python3 delft/applications/grobidTagger.py *name-of-model* train --architecture *name-of-architecture* --input *path-to-the-grobid-data-file-to-be-used-for-training*
```

or 

```sh
python3 delft/applications/grobidTagger.py *name-of-model* train_eval --architecture *name-of-architecture* --input *path-to-the-grobid-data-file-to-be-used-for-training_and_eval_with_random_split*
```

The evaluation of a model with a specific Grobid data file can be performed using the `eval` action and specifying the data file with `--input`: 

```sh
python3 delft/applications/grobidTagger.py citation eval --architecture *name-of-architecture* --input *path-to-the-grobid-data-file-to-be-used-for-evaluation*
```
