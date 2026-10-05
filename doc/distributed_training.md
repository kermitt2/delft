# Training on a cluster (SLURM)

DeLFT ships three shell scripts under [`scripts/`](https://github.com/kermitt2/delft/tree/master/scripts)
for training on a GPU cluster:

1. **`train_distributed_array.sh`**, a SLURM submitter run on the login node. One `sbatch`
   call submits a whole set of trainings as a throttled job array: the standard GROBID
   matrices, the license classifier, or a hyper-parameter sweep on one model.
2. **`test_distributed_array.sh`**, the same kind of submitter for a test matrix: every
   application with every architecture, trained, evaluated, trained with n folds and used,
   see [Testing everything](#testing-everything).
3. **`train_distributed.sh`**, a single-node, multi-GPU launcher: a thin wrapper around
   `torchrun` that you run *inside* an existing GPU allocation.

> **Note on paths.** The submitter defaults to a specific cluster account: an enroot
> container image, a checkout under `/netscratch/lfoppiano/...`, a list of partitions. All of
> it is overridable from the environment, see
> [Adapting to your own cluster](#adapting-to-your-own-cluster).

## Prerequisites

The submitter assumes an [enroot](https://github.com/NVIDIA/enroot)/Pyxis container setup:

- An enroot image containing the PyTorch runtime, by default at
  `/netscratch/lfoppiano/enroot/delft-pytorch.sqsh`.
- A checkout of DeLFT, the one you submit from, with a virtual environment in `.venv/` (so
  the entrypoint is `.venv/bin/python`).
- The training data laid out under `data/sequenceLabelling/grobid/` (see
  [GROBID models](grobid.md)).

## The array submitter

```sh
./scripts/train_distributed_array.sh {train|train-eval|bert|bert-eval|license} [EMBEDDING]
./scripts/train_distributed_array.sh sweep MODEL [--flag value[,value...] ...]
```

The script is both the submitter and the payload of the array: on the login node it computes
the matrix and calls `sbatch` once, with the array index range and the concurrency throttle;
in each array task it maps `SLURM_ARRAY_TASK_ID` back onto one experiment and `exec`s the
training. Every task gets one GPU. Set `DRY_RUN=true` to print the command of every task,
with its index, instead of submitting anything: what it prints is exactly what the tasks
run, since the submitter hands them the settings it resolved (models, embedding, suffix,
sweep flags) as arguments rather than through the environment.

### Standard profiles

The four BidLSTM architectures are `BidLSTM_CRF`, `BidLSTM_CRF_FEATURES`, `BidLSTM_ChainCRF`
and `BidLSTM_ChainCRF_FEATURES`. The five transformers are SciBERT (cased and uncased),
ModernBERT, DeBERTa-v3 and LinkBERT, all with the `BERT_CRF` architecture.

| Profile | Matrix | Tasks | Command |
|---------|--------|-------|---------|
| `train` | 11 GROBID models × 4 BidLSTM architectures | 44 | `grobidTagger <model> train` |
| `train-eval` | 12 models (+ `fulltext`) × 4 BidLSTM architectures | 48 | `grobidTagger <model> train_eval --wandb` |
| `bert` | 11 models × 5 transformers | 55 | `grobidTagger <model> train --architecture BERT_CRF` |
| `bert-eval` | 15 models × 5 transformers | 75 | `grobidTagger <model> train_eval --architecture BERT_CRF --wandb` |
| `license` | license classifier × `gru` | 1 | `licenseClassifier train` |

The static-embedding profiles (`train`, `train-eval`, `license`) take an optional embedding
name after the profile and default to `glove-840B`. `none` trains without pre-trained word
embeddings: on the character features alone for the GROBID models, and with word embeddings
learned from the training texts for the license classifier. The `header` and `citation`
models get `--num-workers 6`, plus `--max-sequence-length 3000` in the `train` profile.

The models of the `bert` profiles are named after their transformer, with `--suffix`:
`grobid-header-BERT_CRF-scibert_scivocab_cased`, `grobid-header-BERT_CRF-ModernBERT-base`
(see [GROBID models](grobid.md)). Two transformers of a run with the same name under different
owners are named with their owner, `owner-a-model` and `owner-b-model`, so that they do not
save the same model. The models of the static profiles keep the release names,
`grobid-header-BidLSTM_CRF_FEATURES`.

A profile is adjusted from the environment:

| Variable | Effect |
|----------|--------|
| `MODELS` | Space-separated subset of the models of the profile: `MODELS="header citation"` trains two models × 4 architectures. |
| `ARCHITECTURES` | Space-separated architectures in place of the four BidLSTM ones (`train`, `train-eval`) or of `gru` (`license`). |
| `TRANSFORMERS` | Space-separated transformers in place of the five default ones (`bert`, `bert-eval`). |
| `INCREMENTAL=true` | Add `--incremental` to every task, to continue training the models already saved. |
| `SUFFIX` | Appended to the name of every model of the run, after the transformer for the `bert` profiles: `SUFFIX=v2` gives `grobid-date-BidLSTM_CRF-v2`. Use it to train a second set without overwriting the first. |

### Sweeping the hyper-parameters of one model

```sh
./scripts/train_distributed_array.sh sweep header \
    --architecture BidLSTM_CRF_FEATURES,BidLSTM_ChainCRF_FEATURES \
    --batch-size 4,8,16,32 --max-epoch 50,100 --patience 5,10,15 --learning-rate 1e-3
```

Every `--flag` after the model name is a flag of `grobidTagger`. A flag with several
comma-separated values is a dimension of the sweep; the tasks are the cartesian product of the
dimensions (2 × 4 × 2 × 3 = 48 above), the first dimension given varying slowest. A flag with
one value is passed to every task as it is. A boolean flag of the tagger (`--incremental`,
`--multi-gpu`, `--wandb`) is given alone, without a value.

The action is `train_eval` (`ACTION=train` for a plain training) and `--wandb` is added
unless `WANDB=false`, so that the runs can be compared in Weights & Biases.

Each task names its model with a `--suffix` built from its swept values, so that no two tasks
overwrite each other and the models can be told apart afterwards:

- a swept `--transformer` contributes the last part of its name (`ModernBERT-base`), or its
  whole name when another swept transformer ends the same way, and a swept
  `--embedding` its value, or `no-embedding` for the value `none`, which trains that task
  without word embeddings;
- any other swept flag contributes the flag without dashes followed by the value:
  `batchsize8`, `maxepoch50`, `patience10`, `learningrate1e-5`, `earlystopfalse`;
- a swept `--architecture` contributes nothing, the architecture being part of the model
  name already;
- the parts are joined with `-`, `SUFFIX` is appended when set, and any character a model name
  cannot hold is replaced by `-`.

The example above saves, among others,
`grobid-header-BidLSTM_CRF_FEATURES-batchsize8-maxepoch50-patience10`. To sweep a transformer
and its learning rate on a plain training:

```sh
DRY_RUN=true ACTION=train ./scripts/train_distributed_array.sh sweep citation \
    --architecture BERT_CRF \
    --transformer allenai/scibert_scivocab_cased,answerdotai/ModernBERT-base \
    --learning-rate 1e-5,3e-5 --num-workers 6
```

which prints the four commands, with suffixes such as `ModernBERT-base-learningrate3e-5`.
Drop `DRY_RUN=true` to submit. The same `--suffix` given to `grobidTagger <model> eval` or
`tag` selects one of the models.

### Submission settings

| Variable | Default | Effect |
|----------|---------|--------|
| `MAX_PARALLEL_JOBS` | `4` | Maximum number of array tasks running at once (the `%<n>` array throttle). |
| `ARRAY_SPEC` | `0-<N-1>` | SLURM array index spec, e.g. `3,7,12-15`: re-run the tasks that failed instead of the whole matrix. The indices are those `DRY_RUN=true` prints. |
| `DRY_RUN` | `false` | Print the command of every task and exit without submitting. |
| `LOG_DIR` | `~/slurm_logs/<profile>[_<model>]_array_<timestamp>` | Where the per-task logs go, one `<array-job-id>_<task-id>.log` per task, stdout and stderr together. |
| `PYTHON_BIN` | `.venv/bin/python` | Interpreter used inside the container. |

### Examples

```sh
# Train every GROBID model × every BidLSTM architecture, 4 jobs at a time
./scripts/train_distributed_array.sh train

# The same with another embedding, 8 jobs at a time, without overwriting the glove models
MAX_PARALLEL_JOBS=8 SUFFIX=fasttext ./scripts/train_distributed_array.sh train fasttext-crawl

# Train and evaluate every model without word embeddings, then with glove, to compare
SUFFIX=char-only ./scripts/train_distributed_array.sh train-eval none
./scripts/train_distributed_array.sh train-eval glove-840B

# Train and evaluate the citation model alone, with the four architectures
MODELS=citation ./scripts/train_distributed_array.sh train-eval

# Continue the training of every model already saved
INCREMENTAL=true ./scripts/train_distributed_array.sh train

# Check what a profile would run, without submitting anything
DRY_RUN=true ./scripts/train_distributed_array.sh bert

# Re-run only the tasks that failed
ARRAY_SPEC=3,7,12-15 ./scripts/train_distributed_array.sh bert-eval

# The license classifier with two architectures
ARCHITECTURES="gru lstm" ./scripts/train_distributed_array.sh license

# Compare no embeddings, glove and potion on the citation model
./scripts/train_distributed_array.sh sweep citation --architecture BidLSTM_CRF_FEATURES \
    --embedding none,glove-840B,potion-base-8M --num-workers 6

# Sweep the batch size and the learning rate of the date model
./scripts/train_distributed_array.sh sweep date --architecture BidLSTM_CRF \
    --batch-size 8,16,32 --learning-rate 1e-3,5e-4
```

## Testing everything

```sh
./scripts/test_distributed_array.sh {smoke|full} [GROUP ...]
./scripts/test_distributed_array.sh report LOG_DIR
```

`test_distributed_array.sh` submits one array that exercises every application:

- `smoke` limits every training to 3 epochs and the n-fold trainings to 2 folds. It tells
  whether everything runs, not how well.
- `full` trains with the epochs of every application and 5 folds, and gives scores to look at.

A **group** is an application: `grobid` (one row per GROBID model with a training file),
`ner`, `insult`, `dataset`, `citation`, `dataseer`, `license`, `software`, `software-context`
and `toxic`. All of them run unless some are named after the profile. A group whose training
data is not in the checkout is left out, with a note on the standard error.

A **task** is one row with one architecture, and runs up to five steps one after the other:

| Step | Taggers | Classifiers |
|------|---------|-------------|
| `train_eval` | `train_eval` | `train_eval` |
| `train` | `train` | `train` |
| `eval` | `eval` on the training file (`grobid`), on the test set (`ner`) | |
| `tag` | `tag` (`grobid` models with sample texts, `insult`, `dataset`) | `classify` |
| `nfold` | `train_eval --fold-count N` | `train_eval --fold-count N` |

An application runs the steps it has, an architecture with features is not used to tag, and
`toxic` is not run with `bert`, which does not do multi-label classification. A task goes on after a
failed step, apart from `eval` and `tag` which need the model of `train`, and fails when any
of its steps did.

The matrix is rows × architectures: the 15 sequence labelling architectures for `grobid`,
those without features for the other taggers, the 11 text classification ones for the
classifiers. The embeddings and the transformers are not a dimension. They **rotate** over
the tasks: an architecture without a transformer gets one of `none`, `glove-840B`,
`potion-base-8M`, `static-retrieval-mrl-en`, `scibert-contextual`,
`contextual:<most-embed-sci>` and the stack `glove-840B+scibert-contextual`, a BERT one gets one of the five transformers of the `bert`
profiles, and the choice shifts by one from a row to the next, so that every embedding meets
every architecture as the rows go by. `PRODUCT=true` runs every architecture with every one
of them instead, which is about six times as many tasks.

Every task works in a directory of its own, `data/test-runs/<run>/task_<index>`, where the
data of the checkout is linked and the models are saved. A run therefore never replaces the
models of `data/models`, two tasks never write the same model, and the directory is removed
when the task ends (`KEEP_MODELS=true` keeps it). The embeddings databases and the cache of
the contextual vectors, under `data/db`, are shared with the checkout.

| Variable | Effect |
|----------|--------|
| `MODELS` | GROBID models, in place of all those with a training file. |
| `ARCHITECTURES` | Sequence labelling architectures, in place of all of them. |
| `CLASSIFIER_ARCHITECTURES` | Text classification architectures, in place of all of them. |
| `EMBEDDINGS` | Embeddings to rotate over, `none` standing for no pre-trained ones. |
| `TRANSFORMERS` | Transformers to rotate over. |
| `STEPS` | Steps to run, in place of `train_eval nfold train eval tag`. |
| `MAX_EPOCH` | Epochs of every training: 3 for `smoke`, the default of the application for `full`. |
| `FOLD_COUNT` | Folds of the `nfold` step: 2 for `smoke`, 5 for `full`. |
| `PRODUCT=true` | Every embedding and transformer with every architecture. |
| `KEEP_MODELS=true` | Keep the working directories and the models in them. |
| `WORK_ROOT` | Where the working directories go, in place of `data/test-runs/<run>`. |

The [submission settings](#submission-settings) and the
[cluster settings](#adapting-to-your-own-cluster) are those of the training submitter, apart
from the time limit of a task: one hour for `smoke`, 23 hours for `full`, unless `TIME_LIMIT`
is set. A task stopped by the limit has no result in the report.

```sh
# What a smoke run would do: every task with the command of each of its steps
DRY_RUN=true ./scripts/test_distributed_array.sh smoke

# Everything, 3 epochs, 8 tasks at a time
MAX_PARALLEL_JOBS=8 ./scripts/test_distributed_array.sh smoke

# The classifiers only
./scripts/test_distributed_array.sh smoke citation license software software-context toxic

# Two GROBID models, the architectures without a transformer, training and n folds only
MODELS="date header" ARCHITECTURES="BidLSTM_CRF BidLSTM_CRF_FEATURES BidLSTM_ChainCRF" \
    STEPS="train_eval nfold" ./scripts/test_distributed_array.sh full grobid

# Which tasks passed, which failed and at which step
./scripts/test_distributed_array.sh report ~/slurm_logs/test_smoke_20261003_101500
```

The log of a task starts with the checkout and the commit whose code it runs. The submitter prints the `report` command of its run. The report lists the tasks that failed
with their failed steps and their log, those that have no result (not run yet, still running,
or stopped by the time limit), and exits with an error unless every task passed. The indices
it prints are those to give to `ARRAY_SPEC` to run these tasks again: add `LOG_DIR=<the same
directory>` so that the report covers both runs.

Several tasks of a row need the same contextual vectors, and each computes them when they run
at the same time on a corpus that is not cached yet. To compute them once, run the tasks with
`scibert-contextual` of every row first (`DRY_RUN=true` gives their indices), then the rest.

## Running & monitoring

```sh
# Your queued/running jobs
squeue -u $USER

# The tasks of a submitted array, and why any of them failed
squeue -j <array-job-id>
sacct -j <array-job-id> --format=JobID,JobName%40,State,Elapsed,ExitCode

# Live log of one task
tail -f ~/slurm_logs/*_array_*/<array-job-id>_<task-id>.log
```

The submitter prints the array job ID and the log directory when it has queued the array. The
first line of every task log is the command the task runs.

## Single-node multi-GPU training

`train_distributed.sh` wraps `torchrun` for data-parallel training on a single node, inside an
allocation you already hold (for instance after `srun --gpus=4 --pty bash`):

```sh
./scripts/train_distributed.sh [NUM_GPUS] <python command ...>

# Auto-detect GPUs
./scripts/train_distributed.sh python -m delft.applications.grobidTagger name-header train --multi-gpu

# Force 2 GPUs
./scripts/train_distributed.sh 2 python -m delft.applications.grobidTagger name-header train --multi-gpu
```

- If the first argument is a number it is used as `--nproc_per_node`; otherwise the GPU count is
  detected with `nvidia-smi -L`.
- The training command must be passed the `--multi-gpu` flag so DeLFT enables distributed mode.

Data parallelism splits every batch across the GPUs of one training: it shortens one long
training, it does not let a model that does not fit on one GPU fit on several. For many
independent trainings, the array submitter uses the cluster better.

## Adapting to your own cluster

The cluster settings of the submitter are environment variables with defaults, all at the top
of `scripts/train_distributed_array.sh`:

| Variable | Default | Meaning |
|----------|---------|---------|
| `CONTAINER_IMAGE` | `/netscratch/lfoppiano/enroot/delft-pytorch.sqsh` | enroot image to run inside. Set it empty (`CONTAINER_IMAGE=`) to drop the three `--container-*` options on a site without enroot/Pyxis, and run against your own module or conda environment. |
| `CONTAINER_WORKDIR` | the checkout the script is in | working directory of the tasks. The tasks run the code of the checkout you submit from, whichever of several checkouts it is. |
| `CONTAINER_MOUNTS` | `/netscratch:/netscratch,$HOME:$HOME` | host paths mounted into the container |
| `PARTITIONS` | `RTX3090,RTXA6000,RTXB6000,L40S` | candidate partitions, the first available is used |
| `CPUS_PER_TASK` | `6` | CPU cores per task. Without it SLURM gives a task one core, on which the data loading workers starve the GPU: a BidLSTM training then shows a few percent of GPU use. |
| `MEMORY` | `100G` | host memory per task |
| `TIME_LIMIT` | `1-00:00` | wall-clock limit per task (days-hours:minutes) |
| `SBATCH_EXTRA` | | any further `sbatch` options, as one string: `SBATCH_EXTRA="--account=abc --qos=long"` |

Every task also gets `--gpus=1 --nodes=1 --export=ALL`, the latter forwarding these variables
and the others of this page to the tasks. Set the variables in your shell profile, or in a
small wrapper script, rather than editing the submitter.
