#!/bin/bash
# Retrain the DeLFT models of GROBID on a SLURM cluster, as one job array: one array task per
# model and architecture, each with its own batch size and maximum sequence length.
#
# Usage:
#   grobid_retraining.sh                 submit the array (DRY_RUN=true to print it instead)
#   grobid_retraining.sh __task ...      what sbatch runs in every array task
#
# Every task trains its model from scratch (iteration 0) on the latest training file of
# data/sequenceLabelling/grobid/<model>/, then INCREMENTAL_ITERATIONS more times with
# --incremental, every one starting from the model the previous one saved. The model is saved
# under OUTPUT_DIR as <model>-<architecture>, the name of its directory in grobid-home/models,
# and every iteration is also copied to OUTPUT_DIR/iter<N>/<model>-<architecture>. With
# TEST_DIR, every iteration is evaluated on TEST_DIR/<model>.test.
#
# Examples:
#   DRY_RUN=true scripts/grobid_retraining.sh
#   rsync -av traces4:/data/workspace/lfoppiano/data/models/delft-training-common-name-2026-11-21/*.test \
#       /netscratch/lfoppiano/delft/data/grobid-test-261121/
#   TEST_DIR=/netscratch/lfoppiano/delft/data/grobid-test-261121 scripts/grobid_retraining.sh
#   TASKS="header citation:BidLSTM_CRF_FEATURES" scripts/grobid_retraining.sh
#   ARRAY_SPEC=2,3 scripts/grobid_retraining.sh        # re-run two tasks of the array
# and once done, for GROBID:
#   cp -r $OUTPUT_DIR/header-BidLSTM_ChainCRF_FEATURES grobid/grobid-home/models/
#
# Environment overrides:
#   TASKS                   space-separated subset of the tasks, as model or model:architecture
#                           (default: all of them, see TASK_TABLE)
#   EMBEDDING               word embeddings (default: glove-840B)
#   OUTPUT_DIR              where the models are saved, relative to CONTAINER_WORKDIR unless
#                           absolute (default: data/models/grobid-retrain-<YYYYMMDD>)
#   INCREMENTAL_ITERATIONS  incremental trainings after the first one (default: 2)
#   TEST_DIR                directory of the <model>.test files; unset, nothing is evaluated
#   WANDB                   when "true", log the trainings to Weights and Biases
#   EXTRA_ARGS              further options of grobidTagger for every training, as one string
#   MAX_PARALLEL_JOBS       maximum number of array tasks running at once (default: 10)
#   ARRAY_SPEC              SLURM array index spec (default: every task, "0-<N-1>")
#   DRY_RUN                 when "true", print the commands of every task instead of submitting
#   LOG_DIR                 log directory (default: ~/slurm_logs/grobid_retraining_<timestamp>)
#   PYTHON_BIN              interpreter, relative to CONTAINER_WORKDIR (default: .venv/bin/python)
#
# Cluster settings, overridable the same way:
#   CONTAINER_IMAGE    enroot image; empty to run without the --container-* options
#   CONTAINER_WORKDIR  working directory of the tasks (default: the checkout of this script)
#   CONTAINER_MOUNTS   host paths mounted into the container
#   PARTITIONS         comma-separated partitions
#   CPUS_PER_TASK      CPU cores per task (default: 6)
#   MEMORY             host memory per task (default: 100G)
#   TIME_LIMIT         wall-clock limit per task, for all its iterations (default: 3-00:00)
#   SBATCH_EXTRA       any further sbatch options, as one string

set -euo pipefail

# model, architecture, batch size, maximum sequence length
TASK_TABLE=(
    "header               BidLSTM_ChainCRF_FEATURES  3   3000"
    "header               BidLSTM_CRF_FEATURES       3   3000"
    "reference-segmenter  BidLSTM_ChainCRF_FEATURES  3   3500"
    "reference-segmenter  BidLSTM_CRF_FEATURES       3   3500"
    "citation             BidLSTM_CRF_FEATURES       30  3000"
    "citation             BidLSTM_ChainCRF_FEATURES  30  3000"
    "affiliation-address  BidLSTM_CRF_FEATURES       20  600"
    "name-header          BidLSTM_CRF_FEATURES       20  3000"
    "name-citation        BidLSTM_CRF_FEATURES       20  3000"
    "patent-citation      BidLSTM_CRF_FEATURES       40  1000"
)
# the largest models get more data loading workers
MANY_WORKERS_MODELS=" header citation "

CONTAINER_IMAGE=${CONTAINER_IMAGE-/netscratch/lfoppiano/enroot/tensorflow-2.17.2-gpu-delft-updated.sqsh}
CONTAINER_WORKDIR=${CONTAINER_WORKDIR:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-"/netscratch:/netscratch,$HOME:$HOME"}
PARTITIONS=${PARTITIONS:-V100-32GB,RTX3090,RTXA6000}
CPUS_PER_TASK=${CPUS_PER_TASK:-6}
MEMORY=${MEMORY:-100G}
TIME_LIMIT=${TIME_LIMIT:-3-00:00}
SBATCH_EXTRA=${SBATCH_EXTRA:-}

MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS:-10}
DRY_RUN=${DRY_RUN:-false}
PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
TASKS=${TASKS:-}
EMBEDDING=${EMBEDDING:-glove-840B}
OUTPUT_DIR=${OUTPUT_DIR:-data/models/grobid-retrain-$(date +%Y%m%d)}
INCREMENTAL_ITERATIONS=${INCREMENTAL_ITERATIONS:-2}
TEST_DIR=${TEST_DIR:-}
WANDB=${WANDB:-false}
EXTRA_ARGS=${EXTRA_ARGS:-}

# An array task is started with `__task` and the settings the submitter resolved, as
# arguments: what a task runs does not depend on the environment reaching it.
TASK_MODE=false
if [[ "${1:-}" == __task ]]; then
    TASK_MODE=true
    shift
    while (($# > 0)); do
        case "$1" in
            --tasks) TASKS=$2 ;;
            --embedding) EMBEDDING=$2 ;;
            --output-dir) OUTPUT_DIR=$2 ;;
            --iterations) INCREMENTAL_ITERATIONS=$2 ;;
            --test-dir) TEST_DIR=$2 ;;
            --wandb) WANDB=$2 ;;
            --extra-args) EXTRA_ARGS=$2 ;;
            --python-bin) PYTHON_BIN=$2 ;;
            *) echo "Unknown task setting: $1" >&2; exit 2 ;;
        esac
        shift 2
    done
elif (($# > 0)); then
    sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 2
fi

if ! [[ "$INCREMENTAL_ITERATIONS" =~ ^[0-9]+$ ]]; then
    echo "INCREMENTAL_ITERATIONS must be a number, got '$INCREMENTAL_ITERATIONS'" >&2
    exit 2
fi

# The rows of TASK_TABLE selected by TASKS, in the order of the table.
ROWS=()
if [[ -z "$TASKS" ]]; then
    ROWS=("${TASK_TABLE[@]}")
else
    read -r -a selectors <<<"$TASKS"
    for selector in "${selectors[@]}"; do
        matched=false
        for row in "${TASK_TABLE[@]}"; do
            read -r model arch _ <<<"$row"
            [[ "$selector" == "$model" || "$selector" == "$model:$arch" ]] && matched=true
        done
        [[ "$matched" == true ]] || { echo "No task matches '$selector' (model or model:architecture)" >&2; exit 2; }
    done
    for row in "${TASK_TABLE[@]}"; do
        read -r model arch _ <<<"$row"
        for selector in "${selectors[@]}"; do
            if [[ "$selector" == "$model" || "$selector" == "$model:$arch" ]]; then
                ROWS+=("$row")
                break
            fi
        done
    done
fi
TOTAL_TASKS=${#ROWS[@]}

read -r -a EXTRA_ARGS_LIST <<<"$EXTRA_ARGS"

# The training command of iteration `iter` of a row, in the global CMD array.
build_train_command() {
    local model=$1 arch=$2 batch=$3 length=$4 iter=$5
    CMD=("$PYTHON_BIN" -m delft.applications.grobidTagger "$model" train
         --architecture "$arch" --embedding "$EMBEDDING" --output "$OUTPUT_DIR"
         --batch-size "$batch" --max-sequence-length "$length")
    [[ "$MANY_WORKERS_MODELS" != *" $model "* ]] || CMD+=(--num-workers 6)
    [[ "$WANDB" != true ]] || CMD+=(--wandb)
    ((iter == 0)) || CMD+=(--incremental)
    CMD+=("${EXTRA_ARGS_LIST[@]}")
}

# Evaluate the model NAME saved in MODELS_DIR on TEST_FILE. The eval action of grobidTagger
# loads its models from data/models/sequenceLabelling only.
EVAL_SCRIPT='
import sys
from delft.sequenceLabelling import Sequence
from delft.sequenceLabelling.reader import load_data_and_labels_crf_file

models_dir, name, test_file = sys.argv[1:4]
x_all, y_all, f_all = load_data_and_labels_crf_file(test_file)
print(len(x_all), "evaluation sequences")
model = Sequence(name)
model.load(dir_path=models_dir)
model.eval(x_all, y_all, features=f_all)
'

# Run every iteration of task `index`.
run_task() {
    local index=$1 model arch batch length name iter status=0
    read -r model arch batch length <<<"${ROWS[$index]}"
    name="$model-$arch"
    for ((iter = 0; iter <= INCREMENTAL_ITERATIONS; iter++)); do
        build_train_command "$model" "$arch" "$batch" "$length" "$iter"
        echo ">>> [$name iter $iter/$INCREMENTAL_ITERATIONS] ${CMD[*]}"
        "${CMD[@]}" || { status=$?; echo ">>> [$name iter $iter] training failed ($status)" >&2; return "$status"; }

        echo ">>> [$name iter $iter] copy to $OUTPUT_DIR/iter$iter/$name"
        rm -rf "${OUTPUT_DIR:?}/iter$iter/$name"
        mkdir -p "$OUTPUT_DIR/iter$iter"
        cp -r "$OUTPUT_DIR/$name" "$OUTPUT_DIR/iter$iter/"

        if [[ -n "$TEST_DIR" ]]; then
            echo ">>> [$name iter $iter] evaluation on $TEST_DIR/$model.test"
            "$PYTHON_BIN" -c "$EVAL_SCRIPT" "$OUTPUT_DIR/iter$iter" "$name" "$TEST_DIR/$model.test" \
                || { status=$?; echo ">>> [$name iter $iter] evaluation failed ($status)" >&2; }
        fi
    done
    return "$status"
}

# Running as an array task: run our row.
if [[ "$TASK_MODE" == true ]]; then
    index=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}
    if ((index < 0 || index >= TOTAL_TASKS)); then
        echo "Invalid array task index: $index ($TOTAL_TASKS tasks)" >&2
        exit 1
    fi
    echo ">>> [task $index/$((TOTAL_TASKS - 1))] ${ROWS[$index]}"
    run_task "$index"
    exit $?
fi

# Submitting: resolve the paths the tasks use, from where they run.
[[ "$OUTPUT_DIR" == /* ]] || OUTPUT_DIR="$CONTAINER_WORKDIR/$OUTPUT_DIR"
[[ -z "$TEST_DIR" || "$TEST_DIR" == /* ]] || TEST_DIR="$(pwd)/$TEST_DIR"

# The training file each model will be trained on: the latest <model>-YYMMDD.train[.gz].
training_file() {
    local dir="$CONTAINER_WORKDIR/data/sequenceLabelling/grobid/$1"
    find "$dir" -maxdepth 1 -regextype posix-extended -regex ".*/$1-[0-9]{6}\.train(\.gz)?" 2>/dev/null \
        | sed -E 's/\.gz$/ gz/' | sort -k1,1 -k2,2r | tail -n 1 | sed 's/ gz$/.gz/'
}

missing=false
for row in "${ROWS[@]}"; do
    read -r model _ <<<"$row"
    if [[ -z "$(training_file "$model")" ]]; then
        echo "No training file for $model under $CONTAINER_WORKDIR/data/sequenceLabelling/grobid/$model" >&2
        missing=true
    fi
    if [[ -n "$TEST_DIR" && ! -f "$TEST_DIR/$model.test" ]]; then
        echo "No evaluation file $TEST_DIR/$model.test" >&2
        missing=true
    fi
done
[[ "$missing" == false ]] || exit 1

ARRAY_SPEC=${ARRAY_SPEC:-"0-$((TOTAL_TASKS - 1))"}

if [[ "$DRY_RUN" == true ]]; then
    echo "$TOTAL_TASKS tasks (array spec: $ARRAY_SPEC), $((INCREMENTAL_ITERATIONS + 1)) trainings each"
    echo "Models saved under $OUTPUT_DIR (and iter<N>/ for every iteration)"
    for ((i = 0; i < TOTAL_TASKS; i++)); do
        read -r model arch batch length <<<"${ROWS[$i]}"
        printf '%3d  %s %s, training file %s\n' "$i" "$model" "$arch" "$(training_file "$model")"
        for ((iter = 0; iter <= INCREMENTAL_ITERATIONS; iter++)); do
            build_train_command "$model" "$arch" "$batch" "$length" "$iter"
            printf '       iter %d: %s\n' "$iter" "${CMD[*]}"
        done
        [[ -z "$TEST_DIR" ]] || printf '       evaluated on %s after every iteration\n' "$TEST_DIR/$model.test"
    done
    exit 0
fi

LOG_DIR=${LOG_DIR:-"${HOME}/slurm_logs/grobid_retraining_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "$LOG_DIR"

SBATCH_OPTS=()
if [[ -n "$CONTAINER_IMAGE" ]]; then
    SBATCH_OPTS+=(--container-image="$CONTAINER_IMAGE"
                  --container-workdir="$CONTAINER_WORKDIR"
                  --container-mounts="$CONTAINER_MOUNTS")
else
    SBATCH_OPTS+=(--chdir="$CONTAINER_WORKDIR")
fi
# shellcheck disable=SC2206  # SBATCH_EXTRA is a string of options to split on spaces
SBATCH_OPTS+=(--export=ALL
              --cpus-per-task="$CPUS_PER_TASK"
              --mem="$MEMORY"
              -p "$PARTITIONS"
              --gpus=1
              --nodes=1
              --time="$TIME_LIMIT"
              --job-name=delft_grobid_retraining
              --array="${ARRAY_SPEC}%${MAX_PARALLEL_JOBS}"
              --output="$LOG_DIR/%A_%a.log"
              --error="$LOG_DIR/%A_%a.log"
              $SBATCH_EXTRA)

# the settings of the tasks, resolved here once: a task reads nothing from the environment
TASK_ARGS=(--tasks "$TASKS" --embedding "$EMBEDDING" --output-dir "$OUTPUT_DIR"
           --iterations "$INCREMENTAL_ITERATIONS" --test-dir "$TEST_DIR" --wandb "$WANDB"
           --extra-args "$EXTRA_ARGS" --python-bin "$PYTHON_BIN")

job_id=$(sbatch --parsable "${SBATCH_OPTS[@]}" "$(readlink -f "$0")" __task "${TASK_ARGS[@]}" | cut -d';' -f1)

echo "Submitted array $job_id: tasks $ARRAY_SPEC of $TOTAL_TASKS (max $MAX_PARALLEL_JOBS concurrent)"
echo "Models:  $OUTPUT_DIR"
echo "Logs:    $LOG_DIR"
echo "Monitor: squeue -j $job_id"
