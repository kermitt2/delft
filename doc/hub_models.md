# Models on the Hugging Face Hub

DeLFT can load its models from the [Hugging Face Hub](https://huggingface.co) as well as from disk. A model on the Hub is the directory DeLFT saves, under the same name, so getting one is a copy with nothing to rename.

## Where a model is

A model is a folder of either

- a **model repository**, one per task, holding the models of that task. It is versioned, and is the place for the models a release depends on:

    ```
    hf://lfoppiano/grobid-model-header/grobid-header-BidLSTM_CRF_FEATURES
    hf://lfoppiano/grobid-model-header@v1.1.0/grobid-header-BidLSTM_CRF_FEATURES
    ```

- a **[bucket](https://huggingface.co/docs/hub/storage-buckets)**, the mutable, unversioned storage of the Hub, mirroring a models directory. It is the place for the models of experiments:

    ```
    hf://buckets/lfoppiano/delft-models/grobid-header-BidLSTM_CRF_FEATURES
    ```

This is the `hf://` syntax of `huggingface_hub`, the revision (`@v1.1.0`, a branch, a tag or a commit) being optional and for repositories only.

The folder is named as [model names](grobid.md#model-names) says: `grobid-{task}-{architecture}`, with an optional `-{suffix}`. A repository of a task therefore holds a folder per architecture and variant of it, and only the folder that is asked for is downloaded: the models with a transformer inside weigh hundreds of megabytes each.

## Loading a model

Nothing changes in the way a model is loaded by name. When it is not in the models directory, DeLFT downloads it there first, then loads it as usual:

```python
from delft.sequenceLabelling import Sequence

model = Sequence("grobid-header-BidLSTM_CRF_FEATURES")
model.load()  # downloaded to data/models/sequenceLabelling/ if it is not there
```

This also holds for the `eval` and `tag` actions of the applications, and for GROBID, which loads models this way.

### From a given place

`load()` also takes the place of the model instead of the models directory, whatever the registry says. This is the way to load a given revision, from a private repository with a `token`, or for an application embedding DeLFT, which has the place of its models and no DeLFT directory layout:

```python
model = Sequence("grobid-header-BidLSTM_CRF_FEATURES", nb_workers=0)

# a repository or a bucket: the model is looked for there under its name
model.load("hf://lfoppiano/grobid-model-header@v1.1.0", cache_dir="~/.cache/my-application/models")

# the model itself, whatever the name given to the Sequence
model.load("hf://lfoppiano/grobid-model-header@v1.1.0/grobid-header-BidLSTM_CRF_FEATURES")

# the same, as the address bar of a browser gives it
model.load("https://huggingface.co/lfoppiano/grobid-model-header/tree/v1.1.0/grobid-header-BidLSTM_CRF_FEATURES")

# an archive of the model directory, anywhere over HTTP
model.load("https://example.org/models/grobid-header-BidLSTM_CRF_FEATURES.zip")

# the URL of a folder: {name of the model}.zip is looked for there
model.load("https://example.org/models/")

# the directory of a model, whatever its name
model.load("/path/to/a/model")

annotations = model.tag(texts, "json", features=features)
```

The `eval` and `tag` actions of the applications take the same places with `--input-model`, for instance to evaluate a model of an experiment that is in a bucket:

```sh
python -m delft.applications.grobidTagger header eval --architecture BidLSTM_CRF --suffix my-experiment \
    --input data/header.train --input-model hf://buckets/lfoppiano/delft-models
```

An archive is a `.zip`, `.tar.gz`, `.tgz` or `.tar` file named after the model, holding the files of the model at its root or in a single folder. `token`, when given, is sent as a bearer token.

What is downloaded goes to `cache_dir`, else to the directory the `DELFT_MODELS_DIR` environment variable names, else to `~/.cache/delft/models`. `Classifier.load` does the same for text classification. Private repositories of the Hub are read with the token of `huggingface-cli login`, or the `token` argument.

### What is on disk wins, except for another revision

A model that was trained or copied into the models directory is never touched, whatever the Hub holds under that name.

A model that was downloaded records where from, in a `.hub-source` file. It is downloaded again when it is asked from somewhere else, another revision or another URL in particular, so that changing a pinned revision does not silently keep the model already there. A reference without a revision follows a branch that moves, and a URL a file that can change: use `--force` (below), or `resolve_model(..., force=True)`, to get what was published since.

## Where DeLFT looks

The `models-hub` section of `delft/resources-registry.json` tells where the models are:

```json
"models-hub": {
    "repo": "lfoppiano/grobid-model-{short_name}",
    "revision": "v1.1.0",
    "bucket": "lfoppiano/delft-models",
    "overrides": {
        "datasets": "lfoppiano/delft-model-datasets"
    }
}
```

| Key | Meaning |
|---|---|
| `repo` | Repository of the models whose name starts with `grobid-`, `{short_name}` being the task: `grobid-header-BidLSTM_CRF` is looked for in `lfoppiano/grobid-model-header`. |
| `revision` | Branch, tag or commit of the repositories to take the models from. Pin it for reproducible installs. `null` follows the default branch. |
| `bucket` | Looked in when the repository does not have the model. `null` for none. |
| `overrides` | Repository of the models `repo` does not apply to, or to take from somewhere else, by their name or by the start of it up to a dash. The longest start wins. |

Every key is optional. Without the section, or with `repo` and `bucket` set to `null`, DeLFT never goes to the Hub.

## Listing and downloading models

```sh
# the models of a task
python -m delft.applications.hub_models list header

# one model
python -m delft.applications.hub_models pull header --architecture BidLSTM_CRF_FEATURES --suffix ""

# every model of an architecture, whatever the suffix
python -m delft.applications.hub_models pull header --architecture BERT_CRF

# by pattern, from a given repository and revision
python -m delft.applications.hub_models pull hf://lfoppiano/grobid-model-header@v1.1.0 --match "*potion*"

# everything
python -m delft.applications.hub_models pull header --all
```

A task stands for its repository as the registry tells it, and with `--bucket` for the models of that task in the bucket. `pull` downloads to `data/models/sequenceLabelling` unless `--output` names another directory, skips the models already there unless `--force` is given, and refuses to download a whole repository without `--all`.

`--architecture BidLSTM_CRF` takes the model without a suffix and the ones with one, and leaves `BidLSTM_CRF_FEATURES` alone, which a pattern cannot do. `--suffix ""` selects the model without a suffix.

The same is available from Python, in `delft.utilities.hub_models`: `resolve_model`, `list_models`, `select_models` and `pull_models`.

## Weights format

Models are published with their weights in the [safetensors](sequence_labeling.md) format, `model.safetensors`, rather than as a pickled state dict: the Hub flags pickled weights as unsafe, since loading them runs whatever code they contain. DeLFT loads either format without being told which.
