# Creating GROBID models with DeLFT

[GROBID](https://github.com/kermitt2/grobid) uses a cascade of sequence labeling models to parse complete documents. The particularity of these models is to use joint text and layout fatures to identify document structures more accurately. The script `delft/applications/grobidTagger.py` allows the creation of various GROBID models to be used by the GROBID services for parsing various structures such as document headers, references, affiliations, authors, dates, etc.

## General command line for training GROBID models in DeLFT

```
usage: grobidTagger.py [-h] [--fold-count FOLD_COUNT] [--seed SEED]
                       [--architecture ARCHITECTURE] [--output OUTPUT]
                       [--embedding EMBEDDING] [--transformer TRANSFORMER]
                       [--input INPUT] [--incremental]
                       [--input-model INPUT_MODEL]
                       [--max-sequence-length MAX_SEQUENCE_LENGTH]
                       [--features-indices FEATURES_INDICES]
                       [--features-vocabulary-size FEATURES_VOCABULARY_SIZE]
                       [--window-stride WINDOW_STRIDE]
                       [--text-features-indices TEXT_FEATURES_INDICES]
                       [--continuous-features-indices CONTINUOUS_FEATURES_INDICES]
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
  --seed SEED           Seed of the random number generators, to run a training
                        again with the same split of the data, the same
                        initial weights and the same order of the batches.
                        Default: not seeded, every run differs.
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
  --features-indices FEATURES_INDICES
                        Columns of the training file to use as features with
                        a FEATURES architecture, the token being column 0,
                        e.g. 9-25,28. Default: every column with at most
                        features-vocabulary-size distinct values.
  --features-vocabulary-size FEATURES_VOCABULARY_SIZE
                        Maximum number of distinct values of a feature column
                        (default: 12).
  --window-stride WINDOW_STRIDE
                        Cut the training sequences longer than
                        max-sequence-length into windows of that length, one
                        every window-stride (in tokens, or in sub-tokens with
                        a transformer), instead of truncating them. A stride
                        smaller than max-sequence-length makes the windows
                        overlap. The validation and evaluation sets are
                        then scored on whole sequences too.
  --text-features-indices TEXT_FEATURES_INDICES
                        Columns of the training file the text of a token is
                        taken from, the token being column 0. For the models
                        that label lines, 0,1 reads the first two tokens of a
                        line rather than the first one.
  --continuous-features-indices CONTINUOUS_FEATURES_INDICES
                        Columns of the training file that hold numbers, given
                        to a FEATURES architecture as numbers scaled to [0, 1]
                        rather than as categories, the token being column 0,
                        e.g. 20,21.
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


> Add `--seed 42` to a `train` / `train_eval` command to make it reproducible: the split of the data, the initial weights and the order of the batches are then the same from a run to the next, and so are the scores on a same machine. Without it every run draws its own, and two runs cannot be compared on the same evaluation set.

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

### Sequences longer than the model takes

A model labels at most `max_sequence_length` tokens of a sequence (sub-tokens with a transformer, where 512 sub-tokens can be less than 300 words). A longer sequence is truncated when it is labelled: the tokens after the cut are left out of the result, and a warning is logged. It is up to the caller to cut long sequences before sending them.

### Training files

The training files are CRF matrices, one token per line: the token, its features, and its label, which any run of spaces and tabs separates, as for Wapiti and CRF++. A model trained with features (a `*_FEATURES` architecture) needs them to evaluate and to label too: doing so without them is an error.

### Choosing the feature columns

The `*_FEATURES` architectures take the columns of the GROBID training file as categorical features, column 0 being the token. By default every column with at most 12 distinct values is used. This leaves out the lexical columns (token, prefixes, suffixes), and also any other column above the limit: in the current GROBID training files, column 25 of _segmentation_ (79 values) and the word shape of _affiliation-address_ (column 19, 133 values). The columns used, and those left out, are printed when the training starts:

```text
Features: using columns [9, 10, 11, 12, 13, 14, 15]
Features: left out columns [0, 1, 2, 3, 4, 5, 6, 7, 8], which have more than 12 distinct values (features_vocabulary_size)
```

To choose the columns, give them with `--features-indices`, as numbers and ranges:

```sh
python3 delft/applications/grobidTagger.py header train --architecture BidLSTM_CRF_FEATURES --features-indices 9-25,28
```

Exactly these columns are then used. If one of them has more distinct values than the limit, the training stops with an error naming it, rather than going on without the column; raise the limit with `--features-vocabulary-size` to keep it. From Python, these are `Sequence(..., features_indices=[...], features_vocabulary_size=...)`.

### Training sequences longer than `--max-sequence-length`

A model takes at most `--max-sequence-length` tokens (sub-tokens with a transformer, where 512 sub-tokens can be less than 300 words). By default a longer training sequence is cut there and what follows is not trained on. With `--window-stride`, it is cut into windows of `--max-sequence-length` instead, one every `--window-stride`, and each window is a training example:

```sh
python3 delft/applications/grobidTagger.py fulltext train --architecture BERT_CRF --transformer allenai/scibert_scivocab_cased --max-sequence-length 512 --window-stride 256
```

The whole sequence is then trained on, and the model also sees sequences that start and end in the middle of a field, which is what it receives when the caller cuts long inputs before sending them for labelling. A stride equal to `--max-sequence-length` puts the windows side by side; a smaller one makes them overlap, for more examples per epoch.

The validation and evaluation sets are then cut into windows too, but always side by side, and the windows of a sequence are put back together before scoring: each token is scored once, a field that spans two windows counts as one, and the scores cover the whole of every sequence rather than its beginning. The stride is saved with the model, so that the `eval` action evaluates a model the way it was trained. Scores obtained with and without the option are not comparable: they are not measured on the same tokens.

From Python, the option is `Sequence(..., window_stride=256)`.

### Models that label lines: reading more than the first token

In the _segmentation_ and _reference-segmenter_ models a position of a sequence is a line, not a token. GROBID gives the first two tokens of the line in the first two columns of the training file, and by default DeLFT takes the text of a position from the first column only: the model reads one word per line.

`--text-features-indices` lists the columns the text is taken from, column 0 being the token:

```sh
python3 delft/applications/grobidTagger.py segmentation train --architecture BidLSTM_CRF_FEATURES --text-features-indices 0,1
```

The text of a line is then its first two tokens: the characters of both are encoded, the word embeddings of the two are concatenated (the input of the model grows accordingly), and a transformer sub-tokenizes both, the label of the line staying on its first sub-token. The columns are saved with the model, so nothing is to be given to evaluate or to tag, except the features themselves, which `tag()` then requires. The tokens returned by `tag()` are the ones it was given.

Two tokens are longer than one: consider a higher `max_char_length` (30 by default) when creating the `Sequence`. From Python, the option is `Sequence(..., text_features_indices=[0, 1])`.

### Features that are numbers

The `*_FEATURES` architectures take the feature columns as categories: every distinct value gets its own embedding, and a column with more than 12 distinct values is left out. A column of numbers (a position, a length, a count) fits badly: with many values it is left out, and as categories `17` and `18` have nothing in common.

`--continuous-features-indices` lists columns to be given to the model as numbers, column 0 being the token:

```sh
python3 delft/applications/grobidTagger.py table train --architecture BidLSTM_CRF_FEATURES --continuous-features-indices 20,21
```

Each of these columns is scaled to [0, 1] with the minimum and maximum seen in the training data, and joins the encoded categorical features at the input of the model; it is not used as a category as well. When labelling, a value outside the range seen in training is clipped, and a value that is not a number counts as the minimum. The columns and their ranges are saved with the model. From Python, the option is `Sequence(..., continuous_features_indices=[20, 21])`.

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
