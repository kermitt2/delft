#!/bin/bash
# Run scripts/benchmark_crf_loss.py on the GPUs of a SLURM cluster.
#
# Usage:
#   benchmark_crf_loss.sh all   [benchmark options...]
#   benchmark_crf_loss.sh srun  [benchmark options...]
#   benchmark_crf_loss.sh array [benchmark options...]
#   benchmark_crf_loss.sh train  [grobidTagger options...]
#   benchmark_crf_loss.sh report [LOG_DIR]
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
# `train` measures what the loss is worth in a training, which the benchmark cannot: it
# trains the same models twice with grobidTagger, once with delft as it is (BASE_REF, the
# baseline) and once with the same delft and the CRF layer of LOSS_REF (the candidate),
# everything else being equal: same seed, hence same split of the data, same initial weights
# and same order of the batches. It submits one job array, one task per model, architecture,
# embeddings and seed; a task runs its two trainings one after the other, so that they get
# the same GPU and the same node. Neither delft is checked out: their `delft/` packages are
# copied under LOG_DIR/code, with the resources registry of the checkout, and the trainings
# run in the checkout, where the training data and the embeddings are. The models are saved
# under names of their own and removed when the training ends: the result is the logs,
# LOG_DIR/runs/<model>.<architecture>.<embeddings>.seed<seed>.<arm>.log, every line of which
# starts with the time it was printed. The options after `train` are passed on to both
# trainings, e.g. `--max-epoch 10`.
#
# `report` reads these logs, of the LOG_DIR given or else of the latest `train`, and prints
# for every pair of trainings the seconds an epoch takes (the median, the first epoch left
# out, validation included), the best F1 on the validation set and the F1 of the evaluation,
# then for every training that failed its exit status, how far it got and the end of its errors.
# It can be run while the trainings are going. The two trainings of a pair are not the same
# to the bit, as their losses differ by rounding: F1 scores a few tenths of a point apart
# are what two runs give, SEEDS with several seeds tells by how much.
#
# Environment overrides:
#   PARTITIONS         comma-separated partitions
#   CONTAINER_IMAGE    enroot image; empty to run without the --container-* options
#   CONTAINER_WORKDIR  working directory of the jobs (default: the checkout of this script)
#   CONTAINER_MOUNTS   host paths mounted into the container
#   CPUS_PER_TASK      CPU cores per job (default: 6)
#   MEMORY             host memory per job (default: 16G; train: 100G)
#   TIME_LIMIT         wall-clock limit per job (default: 00:30:00; train: 1-00:00, for the
#                      two trainings of a task)
#   SBATCH_EXTRA       any further srun/sbatch options, as one string
#   PYTHON_BIN         interpreter to use (default: .venv/bin/python, of the checkout)
#   LOG_DIR            all, array, train: where the output goes (default:
#                      ~/slurm_logs/crf_loss_<timestamp>; train: ~/slurm_logs/crf_loss_train_<timestamp>)
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
#
# For `train` only:
#   MODELS             space-separated GROBID models (default: citation header
#                      affiliation-address reference-segmenter)
#   EMBEDDINGS         space-separated embeddings, `none` for no pre-trained word embeddings
#                      (default: glove-840B none)
#   ARCHITECTURES      space-separated architectures (default: BidLSTM_CRF_FEATURES)
#   SEEDS              space-separated seeds (default: 42)
#   ACTION             the tagger action (default: train_eval)
#   ARMS               the trainings of a task, in the order they run (default: baseline candidate)
#   BASE_REF           the delft both arms are made of (default: origin/dev)
#   LOSS_REF           the delft the candidate takes its CRF layer from (default:
#                      origin/feature/faster-crf-loss)
#   REGISTRY           the resources registry both arms use (default: the one of the checkout)
#   MAX_PARALLEL_JOBS  maximum number of tasks running at once (default: 4)
#   KEEP_MODELS        when "true", leave the trained models in data/models/sequenceLabelling
#   FETCH              when "false", do not `git fetch origin` first

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
SBATCH_EXTRA=${SBATCH_EXTRA:-}
PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
DRY_RUN=${DRY_RUN:-false}

usage() {
    cat >&2 <<USAGE
Usage: $0 {all|srun|array} [benchmark options...]
       $0 train [grobidTagger options...]
       $0 report [LOG_DIR]
USAGE
    exit 2
}

MODE=${1:-}
case "$MODE" in
    all | srun | array | train | report) shift ;;
    *) usage ;;
esac

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [[ "$MODE" == train ]]; then
    # two whole trainings per job, with the embeddings in memory
    MEMORY=${MEMORY:-100G}
    TIME_LIMIT=${TIME_LIMIT:-1-00:00}
    LOG_DIR=${LOG_DIR:-"${HOME}/slurm_logs/crf_loss_train_$TIMESTAMP"}
fi
MEMORY=${MEMORY:-16G}
TIME_LIMIT=${TIME_LIMIT:-00:30:00}
LOG_DIR=${LOG_DIR:-"${HOME}/slurm_logs/crf_loss_$TIMESTAMP"}

# How to call this script again: by its path, or from the branch when it is not in a file.
if [[ -f "$SCRIPT_DIR/benchmark_crf_loss.sh" ]]; then
    SELF="bash $SCRIPT_DIR/benchmark_crf_loss.sh"
else
    SELF="bash <(git show origin/feature/crf-loss-benchmark:scripts/benchmark_crf_loss.sh)"
fi

# The interpreter by its full path: the one of the checkout, unless PYTHON_BIN is a path already.
absolute_python() {
    [[ "$PYTHON_BIN" != /* ]] || return 0
    if [[ ! -x "$CHECKOUT/$PYTHON_BIN" && "$DRY_RUN" != true ]]; then
        echo "No interpreter at $CHECKOUT/$PYTHON_BIN: set PYTHON_BIN" >&2
        exit 2
    fi
    PYTHON_BIN=$CHECKOUT/$PYTHON_BIN
}

# What the log of a training says, as tab-separated fields: its state, the GPU, the number of
# epochs, the median seconds of an epoch after the first, the best F1 on the validation set
# and the F1 of the evaluation; "-" for what it does not say.
summarise_log() {
    if [[ ! -f "$1" ]]; then
        printf 'missing\t-\t0\t-\t-\t-\n'
        return
    fi
    awk '
        $1 == "#" {
            line = substr($0, 3)
            equals = index(line, "=")
            if (equals) meta[substr(line, 1, equals - 1)] = substr(line, equals + 1)
            next
        }
        $2 == "Epoch" && $3 ~ /^[0-9]+:$/ && $4 ~ /^loss=/ {
            stamps[++epochs] = $1
            for (i = 5; i <= NF; i++) {
                if ($i !~ /^val_f1=/) continue
                validated = 1
                if (substr($i, 8) + 0 > best) best = substr($i, 8) + 0
            }
            next
        }
        $2 == "all" && $3 == "(micro" { f1 = $7 }
        END {
            state = "running"
            if ("exit" in meta) state = meta["exit"] == "0" ? "done" : "failed"
            for (i = 2; i <= epochs; i++) seconds[++n] = stamps[i] - stamps[i - 1]
            for (i = 2; i <= n; i++) {
                value = seconds[i]
                for (j = i - 1; j >= 1 && seconds[j] > value; j--) seconds[j + 1] = seconds[j]
                seconds[j + 1] = value
            }
            median = "-"
            if (n) median = n % 2 ? seconds[(n + 1) / 2] : (seconds[n / 2] + seconds[n / 2 + 1]) / 2
            printf "%s\t%s\t%d\t%s\t%s\t%s\n", state, ("gpu" in meta ? meta["gpu"] : "-"), epochs, median,
                (validated ? best + 0 : "-"), (f1 == "" ? "-" : f1)
        }' "$1"
}

# The table of a comparison: one line per pair of trainings.
report() {
    local run_dir=$1 name names
    local b_state b_gpu b_epochs b_seconds b_valid b_f1 c_state c_gpu c_epochs c_seconds c_valid c_f1
    [[ ! -f "$run_dir/run.txt" ]] || { cat "$run_dir/run.txt"; echo; }
    names=$(find "$run_dir/runs" -maxdepth 1 -name '*.log' -printf '%f\n' | sed -E 's/\.(baseline|candidate)\.log$//' | sort -u)
    if [[ -z "$names" ]]; then
        echo "No training has started yet."
        return
    fi
    printf '%-58s %-15s %9s  %19s %8s  %17s  %17s  %s\n' \
        "" "" " epochs" "seconds per epoch" "" "best valid. F1" "evaluation F1" ""
    printf '%-58s %-15s %4s %4s  %9s %9s %8s  %8s %8s  %8s %8s  %s\n' \
        "model.architecture.embeddings.seed" "state" "base" "cand" "base" "cand" "speed-up" \
        "base" "cand" "base" "cand" "GPU"
    while IFS= read -r name; do
        IFS=$'\t' read -r b_state b_gpu b_epochs b_seconds b_valid b_f1 < <(summarise_log "$run_dir/runs/$name.baseline.log")
        IFS=$'\t' read -r c_state c_gpu c_epochs c_seconds c_valid c_f1 < <(summarise_log "$run_dir/runs/$name.candidate.log")
        awk -v name="$name" -v state="$b_state/$c_state" -v b_gpu="$b_gpu" -v c_gpu="$c_gpu" \
            -v b_epochs="$b_epochs" -v c_epochs="$c_epochs" -v b_seconds="$b_seconds" -v c_seconds="$c_seconds" \
            -v b_valid="$b_valid" -v c_valid="$c_valid" -v b_f1="$b_f1" -v c_f1="$c_f1" '
            function number(value, format) { return value == "-" ? "-" : sprintf(format, value) }
            BEGIN {
                speed_up = "-"
                if (b_seconds != "-" && c_seconds != "-" && c_seconds > 0) speed_up = sprintf("%.2fx", b_seconds / c_seconds)
                gpu = b_gpu
                if (b_gpu == "-") gpu = c_gpu
                else if (c_gpu != "-" && c_gpu != b_gpu) gpu = b_gpu " / " c_gpu
                printf "%-58s %-15s %4d %4d  %9s %9s %8s  %8s %8s  %8s %8s  %s\n", name, state, b_epochs, c_epochs,
                    number(b_seconds, "%.1f"), number(c_seconds, "%.1f"), speed_up,
                    number(b_valid, "%.4f"), number(c_valid, "%.4f"), number(b_f1, "%.4f"), number(c_f1, "%.4f"), gpu
            }'
    done <<<"$names"
    report_failures "$run_dir"
}

# Why the trainings that failed did: their exit status, how far they got and the end of what
# they wrote to their standard error.
report_failures() {
    local run_dir=$1 log status
    while IFS= read -r log; do
        status=$(sed -n 's/^# exit=//p' "$log" | tail -1)
        [[ -n "$status" && "$status" != 0 ]] || continue
        echo
        awk -v name="$(basename "$log" .log)" -v status="$status" '
            $1 == "#" && $2 ~ /^host=/ { host = substr($2, 6) }
            $1 == "#" && $2 ~ /^job=/ { job = substr($2, 5) }
            $2 == "Epoch" && $3 ~ /^[0-9]+:$/ && $4 ~ /^loss=/ { epochs++ }
            $2 == "Early" && $3 == "stopping" { stopped = 1 }
            $2 == "training" && $3 == "runtime:" { trained = 1 }
            $2 == "all" && $3 == "(micro" { evaluated = 1 }
            END {
                where = epochs ? "during epoch " (epochs + 1) : "before the end of the first epoch"
                if (stopped || trained) where = "after the training" (stopped ? ", stopped early at epoch " epochs : " of " epochs " epochs")
                if (evaluated) where = "after the evaluation"
                printf "%s: exit %s, %s (host %s, job %s)\n", name, status, where, host, job
            }' "$log"
        if [[ -s "$log.err" ]]; then
            grep -v '^[[:space:]]*$' "$log.err" | tail -6 | cut -c1-240 | sed 's/^/    /'
        else
            echo "    nothing on its standard error: see the end of $log"
        fi
    done < <(find "$run_dir/runs" -maxdepth 1 -name '*.log' | sort)
}

if [[ "$MODE" == report ]]; then
    RUN_DIR=${1:-}
    # without one, the latest comparison
    [[ -n "$RUN_DIR" ]] || RUN_DIR=$(find "$HOME/slurm_logs" -maxdepth 1 -name 'crf_loss_train_*' 2>/dev/null | sort | tail -1 || true)
    if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR/runs" ]]; then
        echo "No comparison in '${RUN_DIR}': give the LOG_DIR of a train run" >&2
        exit 2
    fi
    report "$RUN_DIR"
    exit 0
fi

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
    absolute_python
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

# What an array task of `train` runs, written to a file of its own: the settings as they are
# resolved here, then the two trainings of the task. With DRY_RUN it prints their commands.
write_task_script() {
    {
        echo '#!/bin/bash'
        echo "# Written by benchmark_crf_loss.sh train: task <index> of the comparison, its trainings one after the other."
        declare -p CHECKOUT LOG_DIR PYTHON_BIN ACTION RUN_TAG KEEP_MODELS MODELS ARCHITECTURES EMBEDDINGS SEEDS ARMS TAGGER_ARGS
        cat <<'TASK'
set -uo pipefail

index=${1:-${SLURM_ARRAY_TASK_ID:?give a task index}}
total=$((${#MODELS[@]} * ${#ARCHITECTURES[@]} * ${#EMBEDDINGS[@]} * ${#SEEDS[@]}))
if ((index < 0 || index >= total)); then
    echo "Invalid task index: $index (the comparison has $total tasks)" >&2
    exit 1
fi
# the models vary slowest, the seeds fastest
seed=${SEEDS[$((index % ${#SEEDS[@]}))]}
index=$((index / ${#SEEDS[@]}))
embedding=${EMBEDDINGS[$((index % ${#EMBEDDINGS[@]}))]}
index=$((index / ${#EMBEDDINGS[@]}))
architecture=${ARCHITECTURES[$((index % ${#ARCHITECTURES[@]}))]}
model=${MODELS[$((index / ${#ARCHITECTURES[@]}))]}
name=$model.$architecture.$embedding.seed$seed

# Every line with the time it was printed, which is how the report times the epochs.
stamp() {
    local line
    while IFS= read -r line || [[ -n "$line" ]]; do
        printf '%s %s\n' "$(date +%s.%N)" "$line"
    done
}

# where the training data and the embeddings are
cd "$CHECKOUT" || exit 1
failed=0
for arm in "${ARMS[@]}"; do
    code=$LOG_DIR/code/$arm
    log=$LOG_DIR/runs/$name.$arm.log
    # the model is saved under a name no other training has
    suffix=$RUN_TAG-$arm-${embedding//[^A-Za-z0-9._-]/-}-s$seed
    cmd=("$PYTHON_BIN" "$code/delft/applications/grobidTagger.py" "$model" "$ACTION"
         --architecture "$architecture" --seed "$seed" --suffix "$suffix")
    # no --embedding at all trains without pre-trained word embeddings
    [[ "$embedding" == none ]] || cmd+=(--embedding "$embedding")
    # the two largest models need more data loading workers, as in train_distributed_array.sh
    if [[ "$model" == header || "$model" == citation ]] && [[ " ${TAGGER_ARGS[*]} " != *" --num-workers "* ]]; then
        cmd+=(--num-workers 6)
    fi
    cmd+=("${TAGGER_ARGS[@]}")

    if [[ "${DRY_RUN:-false}" == true ]]; then
        printf 'PYTHONPATH=%q' "$code"
        printf ' %q' "${cmd[@]}"
        printf '\n'
        continue
    fi

    echo ">>> [$name] $arm: ${cmd[*]}"
    # the tagger is run by its path, so that the delft it imports is the one of PYTHONPATH,
    # not the one of the working directory; where.py says which one that is
    {
        echo "# run=$name"
        echo "# arm=$arm"
        echo "# host=${HOSTNAME:-$(hostname)}"
        echo "# job=${SLURM_JOB_ID:-}"
        echo "# command=${cmd[*]}"
        PYTHONPATH=$code "$PYTHON_BIN" "$LOG_DIR/where.py" 2>"$log.err" | grep -E '^(delft|torch|gpu|python)=' | sed 's/^/# /'
    } >"$log"
    if ! grep -qxF "# delft=$(realpath "$code")/delft" "$log"; then
        echo "# exit=wrong-delft" >>"$log"
        echo "$arm did not import the delft of $code: see $log and $log.err" >&2
        failed=1
        continue
    fi
    # the progress bars go to the .err file, where only the finished ones are kept
    PYTHONPATH=$code PYTHONUNBUFFERED=1 "${cmd[@]}" \
        2> >(tr '\r' '\n' | grep -v -E '(^|[^0-9])[0-9]{1,2}%\|' >>"$log.err") | stamp >>"$log"
    status=${PIPESTATUS[0]}
    echo "# exit=$status" >>"$log"
    if [[ "$status" != 0 ]]; then
        echo "$arm failed with status $status: see $log.err" >&2
        failed=1
    fi
    # The result is the log. Only the model of this training goes: the other tasks have the
    # same suffix for another model or architecture, and keep their best weights in theirs
    # while they train.
    [[ "$KEEP_MODELS" == true ]] || rm -rf "data/models/sequenceLabelling/grobid-$model-$architecture-$suffix"
done
exit "$failed"
TASK
    } >"$1"
}

if [[ "$MODE" == train ]]; then
    git -C "$CHECKOUT" rev-parse --git-dir >/dev/null 2>&1 || {
        echo "train needs a git checkout of delft, which $CHECKOUT is not: run it from one" >&2
        exit 2
    }
    BASE_REF=${BASE_REF:-origin/dev}
    LOSS_REF=${LOSS_REF:-origin/feature/faster-crf-loss}
    LOSS_FILE=delft/utilities/crf_pytorch.py
    REGISTRY=${REGISTRY:-$CHECKOUT/delft/resources-registry.json}
    FETCH=${FETCH:-true}
    ACTION=${ACTION:-train_eval}
    MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS:-4}
    KEEP_MODELS=${KEEP_MODELS:-false}
    RUN_TAG=crfloss$TIMESTAMP
    read -r -a MODELS <<<"${MODELS:-citation header affiliation-address reference-segmenter}"
    read -r -a EMBEDDINGS <<<"${EMBEDDINGS:-glove-840B none}"
    read -r -a ARCHITECTURES <<<"${ARCHITECTURES:-BidLSTM_CRF_FEATURES}"
    read -r -a SEEDS <<<"${SEEDS:-42}"
    read -r -a ARMS <<<"${ARMS:-baseline candidate}"
    for arm in "${ARMS[@]}"; do
        [[ "$arm" == baseline || "$arm" == candidate ]] || { echo "ARMS holds baseline and candidate, not '$arm'" >&2; exit 2; }
    done
    TAGGER_ARGS=("$@")
    TOTAL_TASKS=$((${#MODELS[@]} * ${#ARCHITECTURES[@]} * ${#EMBEDDINGS[@]} * ${#SEEDS[@]}))
    absolute_python

    [[ "$FETCH" != true ]] || step git -C "$CHECKOUT" fetch origin
    base_commit=$(git -C "$CHECKOUT" rev-parse --verify --quiet "$BASE_REF^{commit}") || {
        echo "No commit $BASE_REF in $CHECKOUT: set BASE_REF" >&2
        exit 2
    }
    loss_commit=$(git -C "$CHECKOUT" rev-parse --verify --quiet "$LOSS_REF^{commit}") || {
        echo "No commit $LOSS_REF in $CHECKOUT: set LOSS_REF" >&2
        exit 2
    }
    # The candidate is the baseline with one file of LOSS_REF, which is only the change of
    # LOSS_REF when the baseline still has that file as LOSS_REF found it.
    fork_commit=$(git -C "$CHECKOUT" merge-base "$base_commit" "$loss_commit")
    if ! git -C "$CHECKOUT" diff --quiet "$fork_commit" "$base_commit" -- "$LOSS_FILE"; then
        echo "$BASE_REF changed $LOSS_FILE since $LOSS_REF left it: bring $LOSS_REF up to date with it first" >&2
        exit 2
    fi
    if git -C "$CHECKOUT" diff --quiet "$base_commit" "$loss_commit" -- "$LOSS_FILE"; then
        echo "$LOSS_REF has the $LOSS_FILE of $BASE_REF: there is nothing to compare" >&2
        exit 2
    fi

    SUBMIT=(sbatch --parsable "${COMMON_OPTS[@]}" -p "$PARTITIONS"
            --job-name=delft_crf_train
            --array="0-$((TOTAL_TASKS - 1))%$MAX_PARALLEL_JOBS"
            --output="$LOG_DIR/slurm_%a.log"
            --error="$LOG_DIR/slurm_%a.log"
            --wrap="$(print_command bash "$LOG_DIR/task.sh")")
    describe() {
        echo "baseline:  delft of $BASE_REF ($(git -C "$CHECKOUT" log -1 --format='%h %s' "$base_commit"))"
        echo "candidate: the same, with $LOSS_FILE of $LOSS_REF ($(git -C "$CHECKOUT" log -1 --format='%h %s' "$loss_commit"))"
        echo "registry:  $([[ -f "$REGISTRY" ]] && echo "$REGISTRY" || echo "the one of $BASE_REF")"
        echo "models: ${MODELS[*]}; architectures: ${ARCHITECTURES[*]}; embeddings: ${EMBEDDINGS[*]}; seeds: ${SEEDS[*]}"
        echo "$TOTAL_TASKS tasks, each running: ${ARMS[*]}; action $ACTION${TAGGER_ARGS[*]:+; options: ${TAGGER_ARGS[*]}}"
    }

    if [[ "$DRY_RUN" == true ]]; then
        describe
        task_script=$(mktemp)
        write_task_script "$task_script"
        for ((i = 0; i < TOTAL_TASKS; i++)); do
            DRY_RUN=true bash "$task_script" "$i" | sed "s/^/$(printf '%3d' "$i")  /"
        done
        rm -f "$task_script"
        print_command "${SUBMIT[@]}"
        exit 0
    fi

    if [[ -e "$LOG_DIR/code" ]]; then
        echo "$LOG_DIR holds a comparison already: set another LOG_DIR" >&2
        exit 2
    fi
    mkdir -p "$LOG_DIR/runs"
    for arm in baseline candidate; do
        mkdir -p "$LOG_DIR/code/$arm"
        git -C "$CHECKOUT" archive "$base_commit" delft | tar -x -C "$LOG_DIR/code/$arm"
        # the paths of the embeddings on this machine
        [[ ! -f "$REGISTRY" ]] || cp "$REGISTRY" "$LOG_DIR/code/$arm/delft/resources-registry.json"
    done
    git -C "$CHECKOUT" show "$loss_commit:$LOSS_FILE" >"$LOG_DIR/code/candidate/$LOSS_FILE"
    cat >"$LOG_DIR/where.py" <<'WHERE'
# The delft a training imports, and what it runs on.
import os
import platform

import torch

import delft

print("delft=" + os.path.realpath(os.path.dirname(delft.__file__)))
print("torch=" + torch.__version__)
print("gpu=" + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"))
print("python=" + platform.python_version())
WHERE
    describe | tee "$LOG_DIR/run.txt"
    write_task_script "$LOG_DIR/task.sh"

    job_id=$("${SUBMIT[@]}" | cut -d';' -f1)
    echo "Submitted $job_id: $TOTAL_TASKS tasks, $MAX_PARALLEL_JOBS at once"
    echo "Logs:    $LOG_DIR"
    echo "Monitor: squeue -j $job_id"
    echo "Report:  $SELF report $LOG_DIR"
    exit 0
fi

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
