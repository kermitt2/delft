# Training on a cluster (SLURM)

DeLFT ships a set of shell scripts under [`scripts/`](https://github.com/kermitt2/delft/tree/master/scripts)
to train and evaluate the GROBID models (and the license classifier) on a SLURM cluster. They
fall into two families:

1. **A single-node, multi-GPU launcher** — `train_distributed.sh`, a thin wrapper around
   `torchrun` that you run *inside* an existing GPU allocation.
2. **SLURM job submitters** — scripts that you run on the login node to submit one or many
   training jobs to the queue, either sequentially (`srun`) or in parallel with throttling
   (`sbatch`).

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

Every submitter defines the same block of SLURM options (`SBATCH_OPTS` for the parallel
scripts, `SRUN_OPTS` for the sequential ones):

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
| `--time` | `3-00:00` | wall-clock limit (3 days; **1 day** for `train_eval_header_model.sh`) |

The Python entrypoint is `.venv/bin/python -m delft.applications.grobidTagger` for the GROBID
models, and `.venv/bin/python -m delft.applications.licenseClassifier` for the license
classifier.

Each submitter writes its logs to a timestamped directory under
`~/slurm_logs/<run-name>_<YYYYMMDD_HHMMSS>/`, one `*.log` file per job (combined stdout/stderr).

## Environment variables

The `sbatch`-based submitters accept these overrides:

| Variable | Default | Effect |
|----------|---------|--------|
| `MAX_PARALLEL_JOBS` | `4` | Maximum number of jobs kept running/pending at once. The script polls `squeue` and waits before submitting more (`wait_for_capacity`). |
| `WAIT_INTERVAL` | `30` | Seconds to sleep between capacity checks. |
| `WAIT_FOR_COMPLETION` | `false` | Only `train_eval_distributed_all_models.sh`: when `true`, wait for the first wave to finish, then launch a second `--incremental` wave. |

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

### Sequential submitters (`srun`, blocking)

These loop over models and train them **one at a time** — the script blocks on each `srun`
until that training run finishes.

| Script | Models | Architecture(s) | Command |
|--------|--------|-----------------|---------|
| `train_all_models.sh` | 11 GROBID models | 4 BidLSTM architectures | `train` |
| `train_eval_all_models.sh` | 11 models + `fulltext` | 4 BidLSTM architectures | `train_eval --wandb` |
| `train_all_models_bert.sh` | 11 GROBID models | `BERT_CRF` × 5 transformers | `train` |
| `train_eval_all_models_bert.sh` | larger list (adds `quantities`, `units`, `values`, `fulltext`) | `BERT_CRF` × 5 transformers | `train_eval --wandb` |

### Parallel submitters (`sbatch --wrap`, throttled)

These submit many jobs to the queue at once, keeping at most `MAX_PARALLEL_JOBS` in flight.

| Script | Models | Architecture(s) | Command / notes |
|--------|--------|-----------------|-----------------|
| `train_distributed_all_models.sh` | 11 GROBID models | 4 BidLSTM architectures | `train`; `header` & `citation` add `--num-workers 6 --max-sequence-length 3000` |
| `train_distributed_all_models_incremental.sh` | 11 GROBID models | 4 BidLSTM architectures | `train --incremental`; `header` adds the long-sequence flags |
| `train_distributed_citation.sh` | `citation` only | 4 BidLSTM architectures | `train --num-workers 6 --max-sequence-length 3000` |
| `train_eval_distributed_all_models.sh` | 11 models + `fulltext` | 4 BidLSTM architectures | `train_eval --wandb`; `header` & `citation` add `--num-workers 6`. With `WAIT_FOR_COMPLETION=true`, runs a second `--incremental` wave |
| `train_distributed_license_classifier.sh` | license classifier | `gru` | single `licenseClassifier train` job |
| `train_eval_header_model.sh` | `header` only | `BidLSTM_CRF_FEATURES`, `BidLSTM_ChainCRF_FEATURES` | hyperparameter grid: batch size × max-epoch × patience × early-stop, `train_eval --wandb`, `--time=1-00:00` |

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
./scripts/train_distributed_all_models.sh

# Same, but allow 8 concurrent jobs
MAX_PARALLEL_JOBS=8 ./scripts/train_distributed_all_models.sh

# Train + evaluate, then run an incremental second wave when the first finishes
WAIT_FOR_COMPLETION=true ./scripts/train_eval_distributed_all_models.sh
```

Monitor and inspect:

```sh
# Your queued/running jobs
squeue -u $USER

# Live logs for the most recent run
tail -f ~/slurm_logs/train_distributed_*/<job-name>_<jobid>.log
```

Each submitter prints the list of submitted job IDs and the log directory when it finishes
queuing.

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

- The three hard-coded paths in the `SBATCH_OPTS`/`SRUN_OPTS` block: `--container-image`,
  `--container-workdir`, and the `$HOME` mount if your home is elsewhere.
- The `-p` partition list to match your cluster's GPU partitions.
- `--mem`, `--gpus`, and `--time` to your job's needs and your cluster's limits.
- `PYTHON_CMD` if your interpreter is not at `.venv/bin/python`.

If your site does not use enroot/Pyxis container flags, drop the `--container-*` options and run
the `srun`/`sbatch` commands against your own module/conda environment instead.
