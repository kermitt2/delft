# Management of embeddings

The first time DeLFT starts and accesses pre-trained embeddings, these embeddings are serialised and stored in a LMDB database, a very efficient embedded database using memory-mapped file (already used in the Machine Learning world by Caffe and Torch for managing large training data). The next time these embeddings will be accessed, they will be immediately available.

Our approach solves the bottleneck problem pointed for instance [here](https://spenai.org/bravepineapple/faster_em/) in a much better way than quantising+compression or pruning. After being compiled and stored at the first access, any volume of embeddings vectors can be used immediately without any loading, with a negligible usage of memory, without any accuracy loss and with a negligible impact on runtime when using SSD. In practice, we can exploit for instance embeddings for dozen languages simultaneously, without any memory and runtime issues - a requirement for any ambitious industrial deployment of a neural NLP system. 

For instance, in a traditional approach `glove-840B` takes around 2 minutes to load and 4GB in memory. Managed with LMDB, after a first load time of around 4 minutes, `glove-840B` can be accessed immediately and takes only a couple MB in memory, for an impact on runtime negligible (around 1% slower) for any further command line calls.

By default, the LMDB databases are stored under the subdirectory `data/db`. The size of a database is roughly equivalent to the size of the original uncompressed embeddings file. To modify this path, edit the file `delft/resources-registry.json` and change the value of the attribute `embedding-lmdb-path`.

> I have plenty of memory on my machine, I don't care about load time because I need to grab a coffee every ten minutes, I only process one language at the time, so I am not interested in taking advantage of the LMDB emebedding management !

Ok, ok, then set the `embedding-lmdb-path` value to `"None"` in the file `delft/resources-registry.json`, the embeddings will be loaded in memory as immutable data.

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
- `normalize`: whether word vectors are L2-normalized after pooling. It defaults to what the model itself declares (`normalize` in its `config.json`, or a `Normalize` module in its `modules.json`). The registry entries of the sentence-transformers models set it to `true`, because their raw vectors have norms of a few hundreds, a scale a recurrent layer does not train well on.

## Upgrading LMDB caches from 0.3.x to 0.4.x

Starting with DeLFT 0.4.x, embedding vectors are stored in LMDB as raw `float32` bytes instead of the legacy serialized-object format used in 0.3.x. This makes the cache directly readable from other languages (used by [GROBID](https://github.com/kermitt2/grobid) via JEP) and improves load performance.

Existing LMDB caches built by DeLFT 0.3.x are **not directly readable** by 0.4.x. You can either rebuild them by re-loading the source embeddings, or convert in place with the bundled utility:

```sh
python -m delft.utilities.convert_lmdb_embeddings --input <old-lmdb-path> --output <new-lmdb-path>
```

After the conversion, point `embedding-lmdb-path` (in `delft/resources-registry.json`) at the new path and the embedding loader will use the converted cache transparently.
