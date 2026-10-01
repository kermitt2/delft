#!/bin/bash
# Run scripts/benchmark_crf_loss.py on the GPUs of a SLURM cluster.
#
# Usage:
#   benchmark_crf_loss.sh all   [benchmark options...]
#   benchmark_crf_loss.sh srun  [benchmark options...]
#   benchmark_crf_loss.sh array [benchmark options...]
#
# `all` is the whole measurement in one command, whatever the checkout is on: it fetches,
# checks the delft to time (LOSS_REF) out into a worktree of its own beside the checkout,
# puts the benchmark in it, runs it on a GPU with the ladder of shapes, keeps the output
# and the records in LOG_DIR, and removes the worktree. The checkout itself is not touched.
# It does not even have to hold this script:
#
#   git fetch origin && bash <(git show origin/feature/crf-loss-benchmark:scripts/benchmark_crf_loss.sh) all
#
# `srun` runs the benchmark once, interactively, on the delft of the checkout as it is, on a
# GPU of whichever of the PARTITIONS is free, and prints its table to the terminal. `array`
# submits one batch job per partition, so that every kind of GPU gets its own table, written
# to LOG_DIR/<partition>.log. A SLURM job array shares one partition request between its
# tasks, hence one job per partition instead.
#
# The benchmark options are passed on as they are, e.g. `--reps 10 --profile-steps 5`,
# `--ladder` or `--shape citation:200,100,37,20`; see `benchmark_crf_loss.py --help`.
# Without any, `all` runs `--ladder --arms pytorch-crf,A,delft`.
#
# Environment overrides:
#   PARTITIONS         comma-separated partitions
#   CONTAINER_IMAGE    enroot image; empty to run without the --container-* options
#   CONTAINER_WORKDIR  working directory of the jobs (default: the checkout of this script)
#   CONTAINER_MOUNTS   host paths mounted into the container
#   CPUS_PER_TASK      CPU cores per job (default: 6)
#   MEMORY             host memory per job (default: 16G)
#   TIME_LIMIT         wall-clock limit per job (default: 00:30:00)
#   SBATCH_EXTRA       any further srun/sbatch options, as one string
#   PYTHON_BIN         interpreter to use (default: .venv/bin/python, of the checkout)
#   LOG_DIR            all, array: where the output goes (default: ~/slurm_logs/crf_loss_<timestamp>)
#   DRY_RUN            when "true", print the commands instead of running them
#
# For `all` only:
#   LOSS_REF           the delft to time (default: origin/feature/faster-crf-loss)
#   BENCHMARK_REF      take benchmark_crf_loss.py from this ref rather than from beside this
#                      script (default, when it is not beside it: origin/feature/crf-loss-benchmark)
#   FETCH              when "false", do not `git fetch origin` first
#   WORKTREE           where the worktree goes; it has to be under one of the CONTAINER_MOUNTS
#                      (default: a hidden directory beside the checkout)
#   KEEP_WORKTREE      when "true", leave the worktree in place afterwards

set -euo pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
# the checkout this script is part of, or the one it is run from when it is not in one
CHECKOUT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || git rev-parse --show-toplevel 2>/dev/null || true)
# not under git at all: the directory above this script, which is enough for srun and array
[[ -n "$CHECKOUT" ]] || CHECKOUT=$(cd "$SCRIPT_DIR/.." 2>/dev/null && pwd || pwd)

CONTAINER_IMAGE=${CONTAINER_IMAGE-/netscratch/lfoppiano/enroot/delft-pytorch.sqsh}
# the jobs run in the checkout, unless told otherwise
CONTAINER_WORKDIR=${CONTAINER_WORKDIR:-$CHECKOUT}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-"/netscratch:/netscratch,$HOME:$HOME"}
PARTITIONS=${PARTITIONS:-RTX3090,RTXA6000,RTXB6000,L40S}
CPUS_PER_TASK=${CPUS_PER_TASK:-6}
MEMORY=${MEMORY:-16G}
TIME_LIMIT=${TIME_LIMIT:-00:30:00}
SBATCH_EXTRA=${SBATCH_EXTRA:-}
PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
DRY_RUN=${DRY_RUN:-false}

usage() {
    echo "Usage: $0 {all|srun|array} [benchmark options...]" >&2
    exit 2
}

MODE=${1:-}
[[ "$MODE" == all || "$MODE" == srun || "$MODE" == array ]] || usage
shift

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=${LOG_DIR:-"${HOME}/slurm_logs/crf_loss_$TIMESTAMP"}

if [[ "$MODE" == all ]]; then
    git -C "$CHECKOUT" rev-parse --git-dir >/dev/null 2>&1 || {
        echo "all needs a git checkout of delft, which $CHECKOUT is not: run it from one" >&2
        exit 2
    }
    LOSS_REF=${LOSS_REF:-origin/feature/faster-crf-loss}
    FETCH=${FETCH:-true}
    KEEP_WORKTREE=${KEEP_WORKTREE:-false}
    # without its symbolic links, which is how the benchmark reports the delft it timed
    WORKTREE=$(realpath -m "${WORKTREE:-"$(dirname "$CHECKOUT")/.$(basename "$CHECKOUT")-crf-loss-$TIMESTAMP"}")
    BENCHMARK_SOURCE=$SCRIPT_DIR/benchmark_crf_loss.py
    if [[ -n "${BENCHMARK_REF:-}" || ! -f "$BENCHMARK_SOURCE" ]]; then
        BENCHMARK_REF=${BENCHMARK_REF:-origin/feature/crf-loss-benchmark}
        BENCHMARK_SOURCE=
    fi
    [[ $# -gt 0 ]] || set -- --ladder --arms pytorch-crf,A,delft
    [[ " $* " == *" --json "* ]] || set -- "$@" --json "$LOG_DIR/crf_loss.json"
    # the job runs in the worktree, which has no environment of its own: the interpreter of
    # the checkout, by its full path. The benchmark puts the delft beside it ahead of the
    # one that interpreter has installed.
    if [[ "$PYTHON_BIN" != /* ]]; then
        if [[ ! -x "$CHECKOUT/$PYTHON_BIN" && "$DRY_RUN" != true ]]; then
            echo "No interpreter at $CHECKOUT/$PYTHON_BIN: set PYTHON_BIN" >&2
            exit 2
        fi
        PYTHON_BIN=$CHECKOUT/$PYTHON_BIN
    fi
    CONTAINER_WORKDIR=$WORKTREE
    # the table as it is measured, not when the job ends
    export PYTHONUNBUFFERED=1
fi

# the benchmark, relative to the working directory of the job
CMD=("$PYTHON_BIN" scripts/benchmark_crf_loss.py --device cuda "$@")

COMMON_OPTS=()
if [[ -n "$CONTAINER_IMAGE" ]]; then
    COMMON_OPTS+=(--container-image="$CONTAINER_IMAGE"
                  --container-workdir="$CONTAINER_WORKDIR"
                  --container-mounts="$CONTAINER_MOUNTS")
else
    COMMON_OPTS+=(--chdir="$CONTAINER_WORKDIR")
fi
# shellcheck disable=SC2206  # SBATCH_EXTRA is a string of options to split on spaces
COMMON_OPTS+=(--export=ALL
              --cpus-per-task="$CPUS_PER_TASK"
              --mem="$MEMORY"
              --gpus=1
              --nodes=1
              --time="$TIME_LIMIT"
              $SBATCH_EXTRA)

# Print a command, quoted so that it can be pasted back into a shell.
print_command() {
    printf '%q ' "$@"
    printf '\n'
}

# Run a command, or only print it under DRY_RUN.
step() {
    if [[ "$DRY_RUN" == true ]]; then
        print_command "$@"
    else
        echo ">>> $*"
        "$@"
    fi
}

SRUN=(srun "${COMMON_OPTS[@]}" -p "$PARTITIONS" --job-name=delft_crf_loss "${CMD[@]}")

if [[ "$MODE" == srun ]]; then
    if [[ "$DRY_RUN" == true ]]; then
        print_command "${SRUN[@]}"
        exit 0
    fi
    exec "${SRUN[@]}"
fi

if [[ "$MODE" == all ]]; then
    remove_worktree() {
        if [[ "$KEEP_WORKTREE" == true ]]; then
            echo "Worktree kept: $WORKTREE"
        elif [[ "$DRY_RUN" == true || -e "$WORKTREE" ]]; then
            step git -C "$CHECKOUT" worktree remove --force "$WORKTREE"
        fi
    }
    trap remove_worktree EXIT

    [[ "$FETCH" != true ]] || step git -C "$CHECKOUT" fetch origin
    step git -C "$CHECKOUT" worktree add --detach "$WORKTREE" "$LOSS_REF"
    if [[ -n "$BENCHMARK_SOURCE" ]]; then
        step cp "$BENCHMARK_SOURCE" "$WORKTREE/scripts/benchmark_crf_loss.py"
    elif [[ "$DRY_RUN" == true ]]; then
        echo "git -C $CHECKOUT show $BENCHMARK_REF:scripts/benchmark_crf_loss.py > $WORKTREE/scripts/benchmark_crf_loss.py"
    else
        echo ">>> benchmark_crf_loss.py of $BENCHMARK_REF"
        git -C "$CHECKOUT" show "$BENCHMARK_REF:scripts/benchmark_crf_loss.py" >"$WORKTREE/scripts/benchmark_crf_loss.py"
    fi

    if [[ "$DRY_RUN" == true ]]; then
        echo "$(print_command "${SRUN[@]}")| tee $LOG_DIR/crf_loss.log"
        exit 0
    fi

    mkdir -p "$LOG_DIR"
    LOG=$LOG_DIR/crf_loss.log
    echo ">>> timing delft at $(git -C "$WORKTREE" log -1 --format='%h %s')"
    echo ">>> ${SRUN[*]}"
    status=0
    "${SRUN[@]}" 2>&1 | tee "$LOG" || status=$?

    echo
    if [[ "$status" != 0 ]]; then
        echo "The benchmark failed, with status $status" >&2
    elif grep -qF "delft=not asked for" "$LOG"; then
        :
    elif ! grep -qF "delft=$WORKTREE/" "$LOG"; then
        echo "WARNING: the delft arm did not time the worktree's delft; see the delft= entry of the header line" >&2
    elif ! grep -qF "(its own loss)" "$LOG"; then
        echo "WARNING: the delft at $LOSS_REF hands its loss to pytorch-crf, it does not compute it itself" >&2
    fi
    echo "Output:  $LOG"
    [[ ! -f "$LOG_DIR/crf_loss.json" ]] || echo "Records: $LOG_DIR/crf_loss.json"
    exit "$status"
fi

[[ "$DRY_RUN" == true ]] || mkdir -p "$LOG_DIR"

IFS=, read -r -a PARTITION_LIST <<<"$PARTITIONS"
JOB_IDS=()
for partition in "${PARTITION_LIST[@]}"; do
    # the records of each partition beside its log, unless --json was given
    JSON_OPTS=()
    [[ " $* " == *" --json "* ]] || JSON_OPTS=(--json "$LOG_DIR/$partition.json")
    SUBMIT=(sbatch --parsable "${COMMON_OPTS[@]}" -p "$partition"
            --job-name="delft_crf_loss_$partition"
            --output="$LOG_DIR/$partition.log"
            --error="$LOG_DIR/$partition.log"
            --wrap="$(print_command "${CMD[@]}" "${JSON_OPTS[@]}")")
    if [[ "$DRY_RUN" == true ]]; then
        print_command "${SUBMIT[@]}"
        continue
    fi
    job_id=$("${SUBMIT[@]}" | cut -d';' -f1)
    JOB_IDS+=("$job_id")
    echo "Submitted $job_id on $partition"
done

if [[ "$DRY_RUN" != true ]]; then
    joined=$(IFS=,; echo "${JOB_IDS[*]}")
    echo "Logs:    $LOG_DIR"
    echo "Monitor: squeue -j $joined"
fi
