# Management of embeddings

The first time DeLFT starts and accesses pre-trained embeddings, these embeddings are serialised and stored in a LMDB database, a very efficient embedded database using memory-mapped file (already used in the Machine Learning world by Caffe and Torch for managing large training data). The next time these embeddings will be accessed, they will be immediately available.

Our approach solves the bottleneck problem pointed for instance [here](https://spenai.org/bravepineapple/faster_em/) in a much better way than quantising+compression or pruning. After being compiled and stored at the first access, any volume of embeddings vectors can be used immediately without any loading, with a negligible usage of memory, without any accuracy loss and with a negligible impact on runtime when using SSD. In practice, we can exploit for instance embeddings for dozen languages simultaneously, without any memory and runtime issues - a requirement for any ambitious industrial deployment of a neural NLP system. 

For instance, in a traditional approach `glove-840B` takes around 2 minutes to load and 4GB in memory. Managed with LMDB, after a first load time of around 4 minutes, `glove-840B` can be accessed immediately and takes only a couple MB in memory, for an impact on runtime negligible (around 1% slower) for any further command line calls.

By default, the LMDB databases are stored under the subdirectory `data/db`. The size of a database is roughly equivalent to the size of the original uncompressed embeddings file. To modify this path, edit the file `delft/resources-registry.json` and change the value of the attribute `embedding-lmdb-path`.

> I have plenty of memory on my machine, I don't care about load time because I need to grab a coffee every ten minutes, I only process one language at the time, so I am not interested in taking advantage of the LMDB emebedding management !

Ok, ok, then set the `embedding-lmdb-path` value to `"None"` in the file `delft/resources-registry.json`, the embeddings will be loaded in memory as immutable data.

## Getting the embeddings files

When the `path` of an embeddings entry of the registry is not a file, the embeddings are downloaded from its `url` before being compiled into LMDB. A file of the Hugging Face Hub is taken the way the transformers are, through the Hub client: `url` is then a `hf://` reference, `hf://[datasets/]owner/repository[@revision]/file`, or the URL the Hub serves the file at:

```json
{
    "name": "word2vec",
    "path": "/PATH/TO/GoogleNews-vectors-negative300.vec",
    "type": "w2v",
    "format": "vec",
    "lang": "en",
    "item": "word",
    "url": "hf://datasets/sciencialab/word2vec-google-news-negative-300/GoogleNews-vectors-negative300.vec.gz"
}
```

The file is kept in the usual HuggingFace cache (`HF_HOME`), shared with the transformers, so that building the database again does not download it again, and `HF_HUB_OFFLINE=1` works from that cache only. A private repository needs an access token, `HF_ACCESS_TOKEN` as for the transformers, or the one of `huggingface_hub` (`HF_TOKEN`, `hf auth login`). Any other `url` is downloaded under `embedding-download-path`, which is emptied once the embeddings are compiled.

## Modern static embeddings

The embeddings above (glove, word2vec, fasttext `.vec`) are word-level embedding files of several GB, distributed between 2014 and 2018. DeLFT also supports the recent generation of static embeddings, which are distilled from a transformer and distributed on the HuggingFace hub as a small embedding matrix plus a tokenizer:

- the [sentence-transformers static embeddings](https://huggingface.co/blog/static-embeddings), `static-retrieval-mrl-en-v1` and `static-similarity-mrl-multilingual-v1`,
- the [Model2Vec](https://github.com/MinishLab/model2vec) *potion* models, `potion-base-8M`, `potion-base-32M` and `potion-multilingual-128M`.

They are one to two orders of magnitude smaller than glove (30MB to 500MB instead of 4GB), they load in a couple of seconds, they report clearly better quality on similarity and retrieval benchmarks, and - because the vocabulary is made of sub-word units - **there is no out-of-vocabulary word at all**, which the older embeddings handle with a zero vector.

The following names are available in `delft/resources-registry.json` and can be used wherever a static embedding name is expected, for instance:

```sh
python3 delft/applications/nerTagger.py --dataset-type conll2003 train_eval --architecture BidLSTM_CRF --embedding potion-base-8M
```

| name | model | dimensions | languages |
| --- | --- | --- | --- |
| `static-retrieval-mrl-en` | `sentence-transformers/static-retrieval-mrl-en-v1` | 1024 | en |
| `static-retrieval-mrl-en-256` | `sentence-transformers/static-retrieval-mrl-en-v1` | 256 | en |
| `static-similarity-mrl-multilingual` | `sentence-transformers/static-similarity-mrl-multilingual-v1` | 1024 | multilingual |
| `static-similarity-mrl-multilingual-256` | `sentence-transformers/static-similarity-mrl-multilingual-v1` | 256 | multilingual |
| `potion-base-8M` | `minishlab/potion-base-8M` | 256 | en |
| `potion-base-32M` | `minishlab/potion-base-32M` | 512 | en |
| `potion-multilingual-128M` | `minishlab/potion-multilingual-128M` | 256 | multilingual |

The model files are downloaded on first usage and kept in the usual HuggingFace cache (`HF_HOME`), so nothing is compiled into LMDB for these embeddings, and `embedding-lmdb-path` has no effect on them. Set `HF_HUB_OFFLINE=1` to work from the cache only.

Any other static embedding model can be used without editing the registry, by giving its hub identifier or the path to a local copy directly as embedding name:

```sh
python3 delft/applications/grobidTagger.py citation train --architecture BidLSTM_CRF --embedding minishlab/potion-base-32M
```

### How word vectors are produced

A word vector is obtained the way these models embed a text: the word is tokenized into sub-word units and the vectors of these units are averaged (then L2-normalized when the model asks for it). The result is memoized, so the tokenization cost is paid once per distinct word of the corpus.

An entry of the embeddings registry describing such a model looks like this:

```json
{
    "name": "potion-base-8M",
    "model": "minishlab/potion-base-8M",
    "type": "model2vec",
    "format": "static-transformer",
    "lang": "en",
    "item": "word"
}
```

`format` (or `type`) set to `static-transformer` is what routes the embeddings to this backend, and `model` is the hub identifier or the path of a local copy of the model. Two optional attributes are available:

- `dimensions`: the `static-retrieval-mrl-*` and `static-similarity-mrl-*` models are matryoshka (MRL) models, whose vectors can be truncated to fewer dimensions at a very small quality cost. Truncating reduces the size of the RNN input layer accordingly, and hence the training and inference time - this is what the `-256` variants above do.
- `normalize`: `true` L2-normalizes every word vector after pooling, `false` leaves the vectors as the model gives them, and `"global"` divides the whole vocabulary once by the mean norm of its units, so that the average unit has a norm of 1 while a unit keeps its size relative to the others. It defaults to what the model itself declares (`normalize` in its `config.json`, or a `Normalize` module in its `modules.json`). The registry entries of the sentence-transformers models set it to `true`, because their raw vectors have norms of a few hundreds, a scale a recurrent layer does not train well on; but in these models the norm of a unit is also how much it weighs in the pooling they were trained for, which normalizing every word throws away. The `static-retrieval-mrl-en-raw` and `static-retrieval-mrl-en-scaled` entries are the same model with `false` and `"global"`, to compare.

## Contextual embeddings from a frozen transformer

The hidden states of a transformer (BERT, SciBERT, RoBERTa, ...) can be used as word embeddings by all the RNN architectures (`BidLSTM_CRF`, `BidLSTM_CRF_FEATURES`, `BidLSTM_ChainCRF`, ...), the same way ELMo embeddings were used in the past:

```sh
python3 delft/applications/grobidTagger.py citation train --architecture BidLSTM_CRF --embedding scibert-contextual
```

This is different from the `BERT*` architectures: the transformer is **frozen**, it is a feature extractor and not a part of the trained model. The model that is trained and saved is the usual RNN one, with its usual settings (Adam, learning rate of 0.001, no limit of 512 sub-word units on the sequence length), and `--transformer` must not be set.

| name | model | dimensions | language |
|---|---|---|---|
| `scibert-contextual` | `allenai/scibert_scivocab_cased` | 768 | en |
| `bert-base-cased-contextual` | `google-bert/bert-base-cased` | 768 | en |

### How word vectors are produced

- A sequence is tokenized into sub-word units and sent to the transformer. The vector of a word is the mean of the last four hidden layers of its first sub-word unit, so that a sequence of `n` tokens gives `n` vectors and the character, feature and label channels stay aligned, as with static embeddings. Tokens are given as they are written: there is no lowercasing nor number normalization, the transformer has its own handling of those.
- A sequence longer than what the transformer accepts is covered by overlapping windows, each sub-word unit taking its vector from the window where it is the furthest from an edge, i.e. where it has the most context on both sides. The usual long `max_sequence_length` of the RNN architectures can thus be kept (e.g. for the GROBID header model).
- As the transformer is frozen, the vectors of a sequence never change. They are computed once, before the first epoch, in the process that owns the GPU, and cached in LMDB as `float16` under `<embedding-lmdb-path>/contextual/<model>-<hash of the settings>`. The following epochs, n-fold trainings and later runs on the same corpus read the cache, and cost what a training with static embeddings costs. DataLoader workers never load the transformer.
- The vectors of texts to be tagged are computed on the fly and are not written in the cache.

The cache is made of LMDB *shards*: a training that finds sequences missing from the cache writes them in a shard of its own, which is published (atomic rename) once complete and never modified afterwards. Several trainings can therefore share the same cache at the same time, from different nodes and on a network file system, as the tasks of a SLURM job array do. Two trainings starting at the same time on the same corpus will both embed it, which is only a waste of time and space: when using `scripts/train_distributed_array.sh`, submit one architecture per model first (see the header of the script). With a multi-GPU training (`--multi-gpu`), the sequences to embed are shared out between the GPUs.

The cache needs `2 x dimensions` bytes per token, i.e. around 1.5 GB per million tokens for a 768 dimensions model.

An entry of the embeddings registry describing such embeddings looks like this:

```json
{
    "name": "scibert-contextual",
    "model": "allenai/scibert_scivocab_cased",
    "type": "contextual-embedding",
    "format": "contextual-transformer",
    "layers": [-4, -3, -2, -1],
    "layer-pooling": "mean",
    "subword-pooling": "first",
    "lang": "en",
    "item": "word"
}
```

A transformer that is not in the registry can be used with the default settings by prefixing its hub identifier, or the path of a local copy, with `contextual:`:

```sh
python3 delft/applications/grobidTagger.py citation train --architecture BidLSTM_CRF --embedding contextual:michiyasunaga/LinkBERT-base
```

`format` set to `contextual-transformer` is what routes the embeddings to this backend, and `model` is the hub identifier or the path of a local copy of the transformer. The optional attributes are:

- `layers`: the hidden layers to use, as indices in the hidden states of the transformer (default: the last four).
- `layer-pooling`: `mean` (default), `sum` or `concat`. `concat` multiplies the dimensions, the size of the RNN input layer and the size of the cache by the number of layers.
- `subword-pooling`: `first` (default), `last` or `mean`.
- `window` and `stride`: size of a window in sub-word units (default 512, bound by what the transformer accepts) and step between two windows (default 256).
- `batch-size`: number of windows sent to the transformer at once (default 32).
- `cache`: set to `false` to keep the vectors in memory only, and `cache-path` to store the cache somewhere else.

## Stacked embeddings

Several embeddings can be used together, the vector of a word being the concatenation of the vectors given by each of them. The typical usage is a static embedding next to a contextual one, the way ELMo was used together with glove in the past: the static vector brings what is known about the word whatever the sentence, the contextual one brings the reading of the word in this sentence.

The embeddings to stack are given in the embedding name, separated by `+`:

```sh
python3 delft/applications/grobidTagger.py citation train --architecture BidLSTM_CRF --embedding glove-840B+scibert-contextual
```

Each part is anything that is accepted as an embedding name: a name of the registry, the hub identifier or the path of a static embedding model, or a transformer with the `contextual:` prefix (`glove-840B+contextual:michiyasunaga/LinkBERT-base`). There can be more than two of them, and the vectors are concatenated in the order of the name. The size of the RNN input layer is the sum of the sizes, e.g. 300 + 768 for the example above.

Each embedding keeps the behaviour it has when used alone:

- tokens are looked up with number normalization in the static embeddings, and given as they are written to the transformer,
- each one keeps its own LMDB database or cache. In particular, the cache of contextual embeddings is the same whether they are used alone or in a stack: training with `glove-840B+scibert-contextual` after `scibert-contextual` on the same corpus does not embed the corpus again.

A stack can also be given a name in the embeddings registry:

```json
{
    "name": "glove-scibert",
    "format": "stacked",
    "embeddings": ["glove-840B", "scibert-contextual"],
    "lang": "en"
}
```

A stack cannot be a part of another stack: list all the embeddings instead. The name of the embeddings is saved with the model, and the same embeddings are needed to use it.

## Upgrading LMDB caches from 0.3.x to 0.4.x

Starting with DeLFT 0.4.x, embedding vectors are stored in LMDB as raw `float32` bytes instead of the legacy serialized-object format used in 0.3.x. This makes the cache directly readable from other languages (used by [GROBID](https://github.com/kermitt2/grobid) via JEP) and improves load performance.

Existing LMDB caches built by DeLFT 0.3.x are **not directly readable** by 0.4.x. You can either rebuild them by re-loading the source embeddings, or convert in place with the bundled utility:

```sh
python -m delft.utilities.convert_lmdb_embeddings --input <old-lmdb-path> --output <new-lmdb-path>
```

After the conversion, point `embedding-lmdb-path` (in `delft/resources-registry.json`) at the new path and the embedding loader will use the converted cache transparently.
