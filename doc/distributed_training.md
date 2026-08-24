# Training on a cluster (SLURM)

DeLFT ships a set of shell scripts under [`scripts/`](https://github.com/kermitt2/delft/tree/master/scripts)
to train and evaluate the GROBID models (and the license classifier) on a SLURM cluster. They
fall into two families:

1. **A single-node, multi-GPU launcher** — `train_distributed.sh`, a thin wrapper around
   `torchrun` that you run *inside* an existing GPU allocation.
2. **SLURM job submitters** — scripts that you run on the login node to submit training jobs to
   the queue with `sbatch`, either as one throttled job array (`train_distributed_array.sh`) or
   as one `sbatch --wrap` job per experiment.

> **Note on paths.** These scripts were written for a specific cluster account and contain
> hard-coded paths (`/netscratch/lfoppiano/...`, the container image, the checkout directory).
> They are documented here **as-is**, as a reference setup. See
> [Adapting to your own cluster](#adapting-to-your-own-cluster) for what to change.

## Prerequisites

The SLURM submitters assume an [enroot](https://github.com/NVIDIA/enroot)/Pyxis container setup:

- An enroot image containing the PyTorch runtime, at
  `/netscratch/lfoppiano/enroot/delft-pytorch.sqsh`.
- A checkout of DeLFT at `/netscratch/lfoppiano/delft/delft-pytorch2`, with a virtual
  environment in `.venv/` (so the entrypoint is `.venv/bin/python`).
- The training data laid out under `data/sequenceLabelling/grobid/` (see
  [GROBID models](grobid.md)).

## Shared SLURM configuration

Every submitter passes the same block of SLURM options (collected in `SBATCH_OPTS` in the
per-experiment submitters, spelled out on the `sbatch` call in the array submitter):

| Option | Value | Meaning |
|--------|-------|---------|
| `--container-image` | `/netscratch/lfoppiano/enroot/delft-pytorch.sqsh` | enroot image to run inside |
| `--container-workdir` | `/netscratch/lfoppiano/delft/delft-pytorch2` | working directory inside the container (the DeLFT checkout) |
| `--container-mounts` | `/netscratch:/netscratch,$HOME:$HOME` | host paths mounted into the container |
| `--export` | `ALL` | forward the submitting environment into the job |
| `--mem` | `100G` | host memory per job |
| `-p` | `RTX3090,RTXA6000,RTXB6000,L40S` | candidate partitions (first available is used) |
| `--gpus` | `1` | one GPU per job |
| `--nodes` | `1` | single node |
| `--time` | `3-00:00` | wall-clock limit (3 days; **1 day** for the `header` array profile) |

The Python entrypoint is `.venv/bin/python -m delft.applications.grobidTagger` for the GROBID
models, and `.venv/bin/python -m delft.applications.licenseClassifier` for the license
classifier.

Each submitter writes its logs to a timestamped directory under
`~/slurm_logs/<run-name>_<YYYYMMDD_HHMMSS>/`, one `*.log` file per job (combined stdout/stderr).
The array submitter names them `<array-job-id>_<task-id>.log`.

## Environment variables

All submitters accept these overrides:

| Variable | Default | Effect |
|----------|---------|--------|
| `MAX_PARALLEL_JOBS` | `4` | Maximum number of jobs running at once. The array submitter enforces it with the `%<n>` array throttle; the per-experiment submitters poll `squeue` and wait before submitting more (`wait_for_capacity`). |
| `WAIT_INTERVAL` | `30` | Seconds between capacity checks in the per-experiment submitters. |
| `WAIT_FOR_COMPLETION` | `false` | Run the optional incremental wave in `train_eval_distributed_all_models.sh`. |

`train_distributed_array.sh` accepts a few more:

| Variable | Default | Effect |
|----------|---------|--------|
| `DRY_RUN` | `false` | Print the command for every task in the matrix and exit without submitting. |
| `ARRAY_SPEC` | `0-<N-1>` | SLURM array index spec, e.g. `3,7,12-15` — use it to re-run the tasks that failed instead of the whole matrix. |
| `LOG_DIR` | `~/slurm_logs/<profile>_array_<timestamp>` | Where the per-task logs go. |
| `PYTHON_BIN` | `.venv/bin/python` | Interpreter used inside the container. |
| `LEARNING_RATE` | `1e-3` | `header` profile only. |

Example: `MAX_PARALLEL_JOBS=8 ./scripts/train_distributed_all_models.sh`

## Script reference

The four BidLSTM architectures referenced below are
`BidLSTM_CRF`, `BidLSTM_CRF_FEATURES`, `BidLSTM_ChainCRF`, and `BidLSTM_ChainCRF_FEATURES`.
The five transformers used by the BERT scripts are SciBERT (cased and uncased), ModernBERT,
DeBERTa-v3, and LinkBERT.

### Launcher (run inside an allocation)

| Script | What it runs | Notes |
|--------|--------------|-------|
| `train_distributed.sh` | `torchrun --standalone --nproc_per_node=<N> <command>` | Single-node, multi-GPU. `<N>` is the first argument if it is a number, otherwise auto-detected from `nvidia-smi`. See [below](#single-node-multi-gpu-training). |

### Per-experiment submitters (`sbatch --wrap`, throttled)

These submit many jobs to the queue at once, keeping at most `MAX_PARALLEL_JOBS` in flight.

| Script | Models | Architecture(s) | Command / notes |
|--------|--------|-----------------|-----------------|
| `train_distributed_all_models.sh` | 11 GROBID models | 4 BidLSTM architectures | `train`; `header` & `citation` add `--num-workers 6 --max-sequence-length 3000` |
| `train_distributed_all_models_incremental.sh` | 11 GROBID models | 4 BidLSTM architectures | `train --incremental`; `header` adds the long-sequence flags |
| `train_distributed_citation.sh` | `citation` only | 4 BidLSTM architectures | `train --num-workers 6 --max-sequence-length 3000` |
| `train_eval_distributed_all_models.sh` | 11 models + `fulltext` | 4 BidLSTM architectures | `train_eval --wandb`; `header` & `citation` add `--num-workers 6`. With `WAIT_FOR_COMPLETION=true`, runs a second `--incremental` wave |
| `train_distributed_license_classifier.sh` | license classifier | `gru` | single `licenseClassifier train` job |

### SLURM array submitter

`train_distributed_array.sh` replaces the removed sequential launchers with a single throttled
SLURM array: one `sbatch` call, one array task per experiment. Pick the matrix with a profile
name; the script maps `SLURM_ARRAY_TASK_ID` back onto a model/architecture combination.

```sh
./scripts/train_distributed_array.sh {train|train-eval|bert|bert-eval|header} [EMBEDDING]
```

| Profile | Experiment matrix | Tasks | Command |
|---------|-------------------|-------|---------|
| `train` | 11 models × 4 BidLSTM architectures | 44 | `train` |
| `train-eval` | 12 models × 4 BidLSTM architectures | 48 | `train_eval --wandb` |
| `bert` | 11 models × 5 transformers | 55 | `train --architecture BERT_CRF` |
| `bert-eval` | 15 models × 5 transformers | 75 | `train_eval --architecture BERT_CRF --wandb` |
| `header` | 2 architectures × 4 batch sizes × 2 max-epochs × 4 early-stopping settings | 64 | `train_eval --wandb`, `--time=1-00:00` |

As in the per-experiment submitters, `header` and `citation` add `--num-workers 6`, plus
`--max-sequence-length 3000` in the `train` profile.

Static-embedding launchers accept an optional embedding name and default to `glove-840B`. The
array launcher takes it after the profile; the dedicated submitters take it as their first
argument. The `bert` and `bert-eval` profiles use their selected transformer and do not pass
static embeddings.

## Running & monitoring

Run the launcher inside a GPU allocation:

```sh
# Auto-detect GPUs
./scripts/train_distributed.sh python -m delft.applications.grobidTagger name-header train --multi-gpu

# Force 2 GPUs
./scripts/train_distributed.sh 2 python -m delft.applications.grobidTagger name-header train --multi-gpu
```

Submit batches from the login node:

```sh
# Train every GROBID model × every BidLSTM architecture, 4 jobs at a time
./scripts/train_distributed_array.sh train

# Check what a profile would run, without submitting anything
DRY_RUN=true ./scripts/train_distributed_array.sh bert

# Override the default glove-840B embedding
./scripts/train_distributed_array.sh train fasttext-crawl

# BERT train/eval matrix, with at most 8 concurrent array tasks
MAX_PARALLEL_JOBS=8 ./scripts/train_distributed_array.sh bert-eval

# Re-run only the tasks that failed
ARRAY_SPEC=3,7,12-15 ./scripts/train_distributed_array.sh bert-eval

# One sbatch job per experiment instead of an array
MAX_PARALLEL_JOBS=8 ./scripts/train_distributed_all_models.sh

# Train + evaluate, then run an incremental second wave when the first finishes
WAIT_FOR_COMPLETION=true ./scripts/train_eval_distributed_all_models.sh
```

Monitor and inspect:

```sh
# Your queued/running jobs
squeue -u $USER

# The tasks of a submitted array, and why any of them failed
squeue -j <array-job-id>
sacct -j <array-job-id> --format=JobID,JobName%40,State,Elapsed,ExitCode

# Live logs for the most recent run
tail -f ~/slurm_logs/train_distributed_*/<job-name>_<jobid>.log
tail -f ~/slurm_logs/*_array_*/<array-job-id>_<task-id>.log
```

Each submitter prints its job (or array) ID and the log directory when it finishes queuing.

## Single-node multi-GPU training

`train_distributed.sh` wraps `torchrun` for data-parallel training on a single node:

```sh
./scripts/train_distributed.sh [NUM_GPUS] <python command ...>
```

- If the first argument is a number it is used as `--nproc_per_node`; otherwise the GPU count is
  detected with `nvidia-smi -L`.
- The training command must be passed the `--multi-gpu` flag so DeLFT enables distributed mode.

This is the launcher referenced by the multi-GPU example in `CLAUDE.md`.

## Adapting to your own cluster

To reuse these scripts on a different account or cluster, change:

- The three hard-coded paths in the `SBATCH_OPTS` block (or the `sbatch` call in
  `train_distributed_array.sh`): `--container-image`, `--container-workdir`, and the `$HOME`
  mount if your home is elsewhere.
- The `-p` partition list to match your cluster's GPU partitions.
- `--mem`, `--gpus`, and `--time` to your job's needs and your cluster's limits.
- `PYTHON_CMD` if your interpreter is not at `.venv/bin/python` (in the array submitter, set the
  `PYTHON_BIN` environment variable instead).

If your site does not use enroot/Pyxis container flags, drop the `--container-*` options and run
the `sbatch` commands against your own module/conda environment instead.
