#!/bin/bash
# Run scripts/benchmark_crf_loss.py on the GPUs of a SLURM cluster.
#
# Usage:
#   benchmark_crf_loss.sh srun  [benchmark options...]
#   benchmark_crf_loss.sh array [benchmark options...]
#
# `srun` runs the benchmark once, interactively, on a GPU of whichever of the PARTITIONS is
# free, and prints its table to the terminal. `array` submits one batch job per partition, so
# that every kind of GPU gets its own table, written to LOG_DIR/<partition>.log. A SLURM job
# array shares one partition request between its tasks, hence one job per partition instead.
#
# The benchmark options are passed on as they are, e.g. `--reps 10 --profile-steps 5`
# or `--shape citation:200,100,37,20`; see `benchmark_crf_loss.py --help`.
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
#   PYTHON_BIN         interpreter to use (default: .venv/bin/python)
#   LOG_DIR            array only: log directory (default: ~/slurm_logs/crf_loss_<timestamp>)
#   DRY_RUN            when "true", print the srun/sbatch commands instead of running them

set -euo pipefail

CONTAINER_IMAGE=${CONTAINER_IMAGE-/netscratch/lfoppiano/enroot/delft-pytorch.sqsh}
# the jobs run in the checkout this script is part of, unless told otherwise
CONTAINER_WORKDIR=${CONTAINER_WORKDIR:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-"/netscratch:/netscratch,$HOME:$HOME"}
PARTITIONS=${PARTITIONS:-RTX3090,RTXA6000,RTXB6000,L40S}
CPUS_PER_TASK=${CPUS_PER_TASK:-6}
MEMORY=${MEMORY:-16G}
TIME_LIMIT=${TIME_LIMIT:-00:30:00}
SBATCH_EXTRA=${SBATCH_EXTRA:-}
PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
DRY_RUN=${DRY_RUN:-false}

usage() {
    echo "Usage: $0 {srun|array} [benchmark options...]" >&2
    exit 2
}

MODE=${1:-}
[[ "$MODE" == srun || "$MODE" == array ]] || usage
shift

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

if [[ "$MODE" == srun ]]; then
    RUN=(srun "${COMMON_OPTS[@]}" -p "$PARTITIONS" --job-name=delft_crf_loss "${CMD[@]}")
    if [[ "$DRY_RUN" == true ]]; then
        print_command "${RUN[@]}"
        exit 0
    fi
    exec "${RUN[@]}"
fi

LOG_DIR=${LOG_DIR:-"${HOME}/slurm_logs/crf_loss_$(date +%Y%m%d_%H%M%S)"}
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
