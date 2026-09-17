# Creating GROBID models with DeLFT

[GROBID](https://github.com/kermitt2/grobid) uses a cascade of sequence labeling models to parse complete documents. The particularity of these models is to use joint text and layout fatures to identify document structures more accurately. The script `delft/applications/grobidTagger.py` allows the creation of various GROBID models to be used by the GROBID services for parsing various structures such as document headers, references, affiliations, authors, dates, etc.

## General command line for training GROBID models in DeLFT

```
usage: grobidTagger.py [-h] [--fold-count FOLD_COUNT]
                       [--architecture ARCHITECTURE] [--output OUTPUT]
                       [--embedding EMBEDDING] [--transformer TRANSFORMER]
                       [--input INPUT] [--incremental]
                       [--input-model INPUT_MODEL]
                       [--max-sequence-length MAX_SEQUENCE_LENGTH]
                       [--window-stride WINDOW_STRIDE]
                       [--batch-size BATCH_SIZE] [--patience PATIENCE]
                       [--learning-rate LEARNING_RATE] [--max-epoch MAX_EPOCH]
                       [--early-stop EARLY_STOP] [--multi-gpu]
                       [--num-workers NUM_WORKERS] [--suffix SUFFIX] [--wandb]
                       model {train,train_eval,eval,tag}

Trainer for GROBID models using the DeLFT library

positional arguments:
  model                 Name of the model.
  {train,train_eval,eval,tag}

options:
  -h, --help            show this help message and exit
  --fold-count FOLD_COUNT
                        Number of fold to use when evaluating with n-fold cross validation.
  --architecture ARCHITECTURE
                        Type of model architecture to be used, one of
                        ['BidLSTM', 'BidLSTM_CRF', 'BidLSTM_ChainCRF',
                        'BidLSTM_CNN_CRF', 'BidGRU_CRF',
                        'BidLSTM_CNN', 'BidLSTM_CRF_CASING',
                        'BidLSTM_CRF_FEATURES', 'BidLSTM_ChainCRF_FEATURES',
                        'BERT', 'BERT_FEATURES', 'BERT_CRF', 'BERT_ChainCRF',
                        'BERT_CRF_FEATURES', 'BERT_ChainCRF_FEATURES']
  --output OUTPUT       Directory where to save a trained model.
  --embedding EMBEDDING
                        The desired pre-trained word embeddings, see --help
                        and `delft/resources-registry.json` (e.g. 'glove-840B',
                        'fasttext-crawl', 'word2vec').
  --transformer TRANSFORMER
                        The desired pre-trained transformer to be used in the
                        selected architecture, either a local entry from
                        `delft/resources-registry.json` or a HuggingFace Hub
                        model id (e.g. 'bert-base-cased',
                        'allenai/scibert_scivocab_cased').
  --input INPUT         Grobid data file to be used for training (train action),
                        for training and evaluation (train_eval action) or just
                        for evaluation (eval action).
  --incremental         training is incremental, starting from existing model
                        if present.
  --input-model INPUT_MODEL
                        In case of incremental training, path to an existing
                        model to be used to start the training, instead of the
                        default one.
  --max-sequence-length MAX_SEQUENCE_LENGTH
                        max-sequence-length parameter to be used.
  --window-stride WINDOW_STRIDE
                        Cut the training sequences longer than
                        max-sequence-length into windows of that length, one
                        every window-stride (in tokens, or in sub-tokens with
                        a transformer), instead of truncating them. A stride
                        smaller than max-sequence-length makes the windows
                        overlap.
  --batch-size BATCH_SIZE
                        batch-size parameter to be used.
  --patience PATIENCE   patience, number of extra epochs to perform after the
                        best epoch before stopping a training.
  --learning-rate LEARNING_RATE
                        Initial learning rate.
  --max-epoch MAX_EPOCH
                        Maximum number of epochs for training.
  --early-stop EARLY_STOP
                        Force early training termination when metrics scores
                        are not improving after a number of epochs equals to
                        the patience parameter.
  --multi-gpu           Enable distributed computing across multiple GPUs (the
                        batch size needs to be set accordingly using
                        --batch-size).
  --num-workers NUM_WORKERS
                        Number of DataLoader worker processes (default:
                        min(4, cpu_count - 1) for train/eval, 0 for tagging;
                        use 0 for no multiprocessing at all).
  --suffix SUFFIX       Suffix appended to the model name, as in
                        grobid-header-BidLSTM_CRF-<suffix>, to keep several
                        models of the same task and architecture side by
                        side, e.g. one per embeddings, transformer or
                        hyper-parameter setting. Pass the same value to eval
                        and tag to select that model.
  --wandb               Enable the logging of the training using Weights and
                        Biases.
```


> Add `--wandb` to any `train` / `train_eval` / `eval` command to log the run to Weights & Biases. See [Experiment tracking (W&B)](wandb.md) for setup, project selection, and resuming a run for evaluation.

## Model names

A model is saved under `data/models/sequenceLabelling/` in a directory named after the task and the architecture, e.g. `grobid-header-BidLSTM_CRF_FEATURES`. That name says nothing about the embeddings, the transformer or the hyper-parameters, so training the `header` model with `BERT_CRF` on two transformers, or with `BidLSTM_CRF_FEATURES` on two static embeddings, writes twice to the same directory.

`--suffix` keeps such models side by side. It is appended to the name as is:

```sh
python3 delft/applications/grobidTagger.py header train --architecture BERT_CRF --transformer allenai/scibert_scivocab_cased --suffix scibert_scivocab_cased
python3 delft/applications/grobidTagger.py header train --architecture BERT_CRF --transformer answerdotai/ModernBERT-base --suffix ModernBERT-base
```

gives `grobid-header-BERT_CRF-scibert_scivocab_cased` and `grobid-header-BERT_CRF-ModernBERT-base`. The same `--suffix` given to `eval` and `tag` selects the model to load. A suffix holds letters, digits, `.`, `_` and `-`. It is a label and is never parsed: what a model was trained with is in its `config.json`.

Without `--suffix` the name is what it always was, which is also the name GROBID builds when it loads a model through DeLFT. To serve a model trained with a suffix from GROBID, copy or link its directory to the name without the suffix.

## GROBID models

DeLFT supports [GROBID](https://github.com/kermitt2/grobid) training data (originally for CRF) and GROBID feature matrix to be labelled. Default static embeddings for GROBID models are `glove-840B`, which can be changed with parameter `--embedding`. 

For an end-to-end comparison of architectures on the `header` model (F1-score and runtime on the PMC evaluation set), see [Header model evaluation summary](features_header_eval_summary.md).

Train a model with all available training data:

```sh
python3 delft/applications/grobidTagger.py *name-of-model* train --architecture *name-of-architecture*
```

where *name-of-model* is one of GROBID model (_date_, _affiliation-address_, _citation_, _header_, _name-citation_, _name-header_, ...),

and where *name-of-architecture* is one of `['BidLSTM', 'BidLSTM_CRF', 'BidLSTM_ChainCRF', 'BidLSTM_CNN', 'BidLSTM_CNN_CRF', 'BidGRU_CRF', 'BidLSTM_CRF_CASING', 'BidLSTM_CRF_FEATURES', 'BidLSTM_ChainCRF_FEATURES', 'BERT', 'BERT_CRF', 'BERT_ChainCRF', 'BERT_FEATURES', 'BERT_CRF_FEATURES', 'BERT_ChainCRF_FEATURES']` (see [Sequence Labeling](sequence_labeling.md#available-models)).

For instance, for the _date_ model:

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

### Training sequences longer than `--max-sequence-length`

A model takes at most `--max-sequence-length` tokens (sub-tokens with a transformer, where 512 sub-tokens can be less than 300 words). By default a longer training sequence is cut there and what follows is not trained on. With `--window-stride`, it is cut into windows of `--max-sequence-length` instead, one every `--window-stride`, and each window is a training example:

```sh
python3 delft/applications/grobidTagger.py fulltext train --architecture BERT_CRF --transformer allenai/scibert_scivocab_cased --max-sequence-length 512 --window-stride 256
```

The whole sequence is then trained on, and the model also sees sequences that start and end in the middle of a field, which is what it receives when the caller cuts long inputs before sending them for labelling. A stride equal to `--max-sequence-length` puts the windows side by side; a smaller one makes them overlap, for more examples per epoch. Only the training set is cut into windows: the validation and evaluation sets are truncated as before, so that scores stay comparable with and without the option.

From Python, the option is `Sequence(..., window_stride=256)`.

### Sequences longer than the model takes

A model labels at most `max_sequence_length` tokens of a sequence (sub-tokens with a transformer, where 512 sub-tokens can be less than 300 words). A longer sequence is truncated when it is labelled: the tokens after the cut are left out of the result, and a warning is logged. It is up to the caller to cut long sequences before sending them.

## Calling DeLFT from GROBID

GROBID embeds DeLFT in the JVM process through [JEP](https://github.com/ninia/jep): it instantiates a `Sequence`, loads the model once, then calls `tag()` for every sequence to be labelled.

In that setup the number of DataLoader worker *processes* matters. A DataLoader is built on every `tag()` call, so any value above 0 respawns worker processes per call, each one forking the host interpreter. Pass `nb_workers=0` to run everything in-process:

```python
from delft.sequenceLabelling import Sequence

model = Sequence("grobid-header-BidLSTM_CRF_FEATURES", nb_workers=0)
model.load()
annotations = model.tag(texts, "json", features=features)
```

`nb_workers` can also be given per call, which takes precedence over the constructor:

```python
annotations = model.tag(texts, "json", features=features, nb_workers=0)
```

When neither is set, tagging defaults to `nb_workers=0` already — the constructor default (`min(4, cpu_count - 1)`) only applies to training and evaluation, where the worker pool is spawned once and amortised over the whole run. `create_dataloader` additionally caps the requested count by dataset size, so small batches never spawn workers that would sit idle.

The same applies to text classification, where `Classifier.predict()` accepts `nb_workers` and the equivalent `use_main_thread_only=True`:

```python
result = classifier.predict(texts, "json", use_main_thread_only=True)
```

From the command line, `--num-workers 0` has the same effect:

```sh
python3 delft/applications/grobidTagger.py header tag --architecture BidLSTM_CRF_FEATURES --num-workers 0
```
