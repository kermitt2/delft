# Experiment tracking with Weights & Biases

DeLFT can log training and evaluation runs to [Weights & Biases](https://wandb.ai)
(W&B). Tracking is **opt-in** via the `--wandb` flag and is supported by the sequence
labelling applications (`grobidTagger`, `nerTagger`, …) and the text classifiers.

## Setup

W&B is **not** part of the base install. It ships with the `[dev]` extras:

```sh
uv pip install -e ".[dev]"            # or ".[gpu,dev]" on a CUDA box
# ad-hoc alternative:
pip install wandb
```

Authenticate by exporting your API key before launching a run (DeLFT reads it from the
environment; a `.env` file is also honoured):

```sh
export WANDB_API_KEY=<your-key>
```

If `wandb` is not installed, or `WANDB_API_KEY` is unset, DeLFT prints a warning and
**silently continues without tracking** — training is never blocked by a missing W&B
setup.

## Usage

Add `--wandb` to any `train` or `train_eval` command:

```sh
python -m delft.applications.grobidTagger header train_eval \
    --architecture BERT_CRF --wandb
```

The run is named after the model, and the following hyper-parameters are recorded as the
run config: model name, architecture, transformer / embeddings name, embedding size,
batch size, learning rate, max epoch, patience, early-stop, and max sequence length.

During training the validation `f1` is logged each epoch (with a `max` summary). During
evaluation, `eval_f1`, `eval_precision` and `eval_recall` are logged, together with a
per-label **Evaluation scores** table.

### Choosing the project

The W&B project is resolved in this order:

1. `--wandb-project <name>` (CLI flag)
2. the `WANDB_PROJECT` environment variable
3. wandb's own default (`uncategorized`)

```sh
python -m delft.applications.grobidTagger header train_eval --wandb \
    --wandb-project delft-grobid
```

### Resuming a run for evaluation

To attach evaluation metrics to the **same** run produced by an earlier training (rather
than creating a new one), pass that run's ID to a standalone `eval` action:

```sh
python -m delft.applications.grobidTagger header eval \
    --wandb --wandb-run-id <run-id>
```

`--wandb-run-id` is only meaningful with the `eval` action; it resumes the existing run
(`resume="must"`) so the eval metrics and the per-label table land alongside the original
training curves.

## Run directory

wandb's default run-log directory is `./wandb/`, which on a subsequent run shadows the
`import wandb` package as an empty namespace package. To avoid this, DeLFT writes run logs
to `./.wandb/` (dot-prefixed) instead. Set `WANDB_DIR` to override:

```sh
export WANDB_DIR=/path/to/wandb-logs
```

## On a cluster

The SLURM submitter scripts under [`scripts/`](https://github.com/kermitt2/delft/tree/master/scripts)
already pass `--wandb` for their `train_eval` runs — see
[Training on a cluster (SLURM)](distributed_training.md). Make sure `WANDB_API_KEY` is
exported into the job environment (the scripts use `--export=ALL`, so exporting it on the
submitting shell is enough).

## Environment variables

| Variable | Effect |
|----------|--------|
| `WANDB_API_KEY` | Required to enable tracking; if unset, W&B logging is disabled with a warning. |
| `WANDB_PROJECT` | Default project name when `--wandb-project` is not given. |
| `WANDB_DIR` | Override the run-log directory (defaults to `./.wandb/`). |
