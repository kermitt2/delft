# Training on a cluster (SLURM)

DeLFT ships two shell scripts under [`scripts/`](https://github.com/kermitt2/delft/tree/master/scripts)
for training on a GPU cluster:

1. **`train_distributed_array.sh`**, a SLURM submitter run on the login node. One `sbatch`
   call submits a whole set of trainings as a throttled job array: the standard GROBID
   matrices, the license classifier, or a hyper-parameter sweep on one model.
2. **`train_distributed.sh`**, a single-node, multi-GPU launcher: a thin wrapper around
   `torchrun` that you run *inside* an existing GPU allocation.

> **Note on paths.** The submitter defaults to a specific cluster account: an enroot
> container image, a checkout under `/netscratch/lfoppiano/...`, a list of partitions. All of
> it is overridable from the environment, see
> [Adapting to your own cluster](#adapting-to-your-own-cluster).

## Prerequisites

The submitter assumes an [enroot](https://github.com/NVIDIA/enroot)/Pyxis container setup:

- An enroot image containing the PyTorch runtime, by default at
  `/netscratch/lfoppiano/enroot/delft-pytorch.sqsh`.
- A checkout of DeLFT, by default at `/netscratch/lfoppiano/delft/delft-pytorch2`, with a
  virtual environment in `.venv/` (so the entrypoint is `.venv/bin/python`).
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
name after the profile and default to `glove-840B`; `none` trains without word embeddings,
on the character features alone. The `header` and `citation` models get `--num-workers 6`,
plus `--max-sequence-length 3000` in the `train` profile.

The models of the `bert` profiles are named after their transformer, with `--suffix`:
`grobid-header-BERT_CRF-scibert_scivocab_cased`, `grobid-header-BERT_CRF-ModernBERT-base`
(see [GROBID models](grobid.md)). The models of the static profiles keep the release names,
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

- a swept `--transformer` contributes the last part of its name (`ModernBERT-base`), a swept
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
| `CONTAINER_WORKDIR` | `/netscratch/lfoppiano/delft/delft-pytorch2` | working directory of the tasks: the DeLFT checkout |
| `CONTAINER_MOUNTS` | `/netscratch:/netscratch,$HOME:$HOME` | host paths mounted into the container |
| `PARTITIONS` | `RTX3090,RTXA6000,RTXB6000,L40S` | candidate partitions, the first available is used |
| `CPUS_PER_TASK` | `6` | CPU cores per task. Without it SLURM gives a task one core, on which the data loading workers starve the GPU: a BidLSTM training then shows a few percent of GPU use. |
| `MEMORY` | `100G` | host memory per task |
| `TIME_LIMIT` | `3-00:00` | wall-clock limit per task (days-hours:minutes) |
| `SBATCH_EXTRA` | | any further `sbatch` options, as one string: `SBATCH_EXTRA="--account=abc --qos=long"` |

Every task also gets `--gpus=1 --nodes=1 --export=ALL`, the latter forwarding these variables
and the others of this page to the tasks. Set the variables in your shell profile, or in a
small wrapper script, rather than editing the submitter.
