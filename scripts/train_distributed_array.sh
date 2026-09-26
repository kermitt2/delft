#!/bin/bash
# Submit DeLFT trainings to a SLURM cluster as one throttled job array: one array task per
# experiment, at most MAX_PARALLEL_JOBS running at once.
#
# Usage:
#   train_distributed_array.sh {train|train-eval|bert|bert-eval|license} [EMBEDDING]
#   train_distributed_array.sh sweep MODEL [--flag value[,value...] ...]
#   train_distributed_array.sh __task ...   (what sbatch runs in every array task)
#
# The standard profiles run a matrix of GROBID models by architectures (or transformers)
# with grobidTagger; `license` runs the license classifier. EMBEDDING defaults to glove-840B;
# `none` trains without word embeddings (character features only). `sweep` runs one model over the
# cartesian product of every flag given several comma-separated values, the other flags
# being passed as they are; each task gets a --suffix built from its swept values so that
# the saved models do not overwrite each other. A boolean flag of the tagger (--wandb,
# --incremental, --multi-gpu) is given alone, without a value.
#
# EMBEDDING can also name contextual embeddings from a frozen transformer (e.g. scibert-contextual,
# see doc/embeddings.md). Their vectors are computed once per corpus and cached: tasks running at
# the same time on the same corpus would each compute them. To avoid that, submit the first
# architecture of every model first, then the whole matrix once these tasks are done, e.g.
#   ARRAY_SPEC=0-43:4 train_distributed_array.sh train scibert-contextual
#
# EMBEDDING can also name contextual embeddings from a frozen transformer (e.g. scibert-contextual,
# see doc/embeddings.md). Their vectors are computed once per corpus and cached: tasks running at
# the same time on the same corpus would each compute them. To avoid that, submit the first
# architecture of every model first, then the whole matrix once these tasks are done, e.g.
#   ARRAY_SPEC=0-43:4 train_distributed_array.sh train scibert-contextual
#
# Environment overrides:
#   MODELS             space-separated subset of the models of a standard profile
#   ARCHITECTURES      space-separated architectures (train, train-eval, license)
#   TRANSFORMERS       space-separated transformers (bert, bert-eval)
#   INCREMENTAL        when "true", add --incremental to every task of a standard profile
#   SUFFIX             appended to the model name of every task, after the one the profile
#                      or the sweep builds (train and train-eval have none by default)
#   ACTION             sweep only: the tagger action (default: train_eval)
#   WANDB              sweep only: when "false", do not add --wandb (default: true)
#   MAX_PARALLEL_JOBS  maximum number of array tasks running at once (default: 4)
#   ARRAY_SPEC         SLURM array index spec, e.g. "3,7,12-15" to re-run a few failed
#                      tasks (default: the whole matrix, "0-<N-1>")
#   DRY_RUN            when "true", print every task command instead of submitting
#   LOG_DIR            log directory (default: ~/slurm_logs/<profile>_array_<timestamp>)
#   PYTHON_BIN         interpreter to use (default: .venv/bin/python)
#
# Cluster settings, overridable the same way:
#   CONTAINER_IMAGE    enroot image; empty to run without the --container-* options
#   CONTAINER_WORKDIR  working directory of the tasks (default: the checkout of this script)
#   CONTAINER_MOUNTS   host paths mounted into the container
#   PARTITIONS         comma-separated partitions
#   CPUS_PER_TASK      CPU cores per task, for the data loading workers (default: 6)
#   MEMORY             host memory per task (default: 100G)
#   TIME_LIMIT         wall-clock limit per task (default: 1-00:00)
#   SBATCH_EXTRA       any further sbatch options, as one string

set -euo pipefail

CONTAINER_IMAGE=${CONTAINER_IMAGE-/netscratch/lfoppiano/enroot/delft-pytorch.sqsh}
# the tasks run in the checkout this script is part of, unless told otherwise
CONTAINER_WORKDIR=${CONTAINER_WORKDIR:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-"/netscratch:/netscratch,$HOME:$HOME"}
PARTITIONS=${PARTITIONS:-RTX3090,RTXA6000,RTXB6000,L40S}
CPUS_PER_TASK=${CPUS_PER_TASK:-6}
MEMORY=${MEMORY:-100G}
TIME_LIMIT=${TIME_LIMIT:-1-00:00}
SBATCH_EXTRA=${SBATCH_EXTRA:-}

MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS:-4}
DRY_RUN=${DRY_RUN:-false}
PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
INCREMENTAL=${INCREMENTAL:-false}
SUFFIX=${SUFFIX:-}
ACTION=${ACTION:-train_eval}
WANDB=${WANDB:-true}

usage() {
    cat >&2 <<USAGE
Usage: $0 {train|train-eval|bert|bert-eval|license} [EMBEDDING]
       $0 sweep MODEL [--flag value[,value...] ...]
USAGE
    exit 2
}

# the lists given in the environment, read before the arrays of the same names are set
ENV_MODELS=${MODELS:-}
ENV_ARCHITECTURES=${ARCHITECTURES:-}
ENV_TRANSFORMERS=${TRANSFORMERS:-}

# An array task is started with `__task` and the settings the submitter resolved, as
# arguments: what a task runs does not depend on the environment reaching it.
TASK_MODE=false
TASK_ARGS=()
if [[ "${1:-}" == __task ]]; then
    TASK_MODE=true
    shift
    while (($# > 0)); do
        case "$1" in
            --profile) PROFILE=$2 ;;
            --embedding) EMBEDDING=$2 ;;
            --models) read -r -a MODELS <<<"$2" ;;
            --items) read -r -a ITEMS <<<"$2" ;;
            --model) MODEL=$2 ;;
            --action) ACTION=$2 ;;
            --wandb) WANDB=$2 ;;
            --suffix) SUFFIX=$2 ;;
            --incremental) INCREMENTAL=$2 ;;
            --python-bin) PYTHON_BIN=$2 ;;
            --) shift; TASK_ARGS=("$@"); break ;;
            *) echo "Unknown task setting: $1" >&2; exit 2 ;;
        esac
        shift 2
    done
    set -- "${TASK_ARGS[@]}"
else
    PROFILE=${1:-}
    [[ -n "$PROFILE" ]] || usage
    shift
fi

DEFAULT_ARCHITECTURES=(BidLSTM_CRF BidLSTM_CRF_FEATURES BidLSTM_ChainCRF BidLSTM_ChainCRF_FEATURES)
DEFAULT_TRANSFORMERS=(
    allenai/scibert_scivocab_cased
    allenai/scibert_scivocab_uncased
    answerdotai/ModernBERT-base
    microsoft/deberta-v3-base
    michiyasunaga/LinkBERT-base
)

# A list of a profile in the global LIST: the space-separated one given in the environment
# when there is one, else the default.
list_from_env() {
    local from_env=$1
    shift
    if [[ -n "$from_env" ]]; then
        read -r -a LIST <<<"$from_env"
    else
        LIST=("$@")
    fi
}

# A model name suffix (see delft/utilities/model_names.py) holds letters, digits, '.', '_'
# and '-', and starts with a letter or a digit.
sanitize_suffix() {
    local value=$1
    value=${value//[^A-Za-z0-9._-]/-}
    while [[ -n "$value" && ! "$value" =~ ^[A-Za-z0-9] ]]; do
        value=${value:1}
    done
    printf '%s' "$value"
}

# The --embedding option of a task, none for the embedding `none`.
embedding_option() {
    local embedding=$1
    EMBEDDING_OPTION=()
    [[ "$embedding" == none ]] || EMBEDDING_OPTION=(--embedding "$embedding")
}

# The suffix components of a task, joined with '-', $SUFFIX last.
join_suffix() {
    local joined="" part
    for part in "$@" "$SUFFIX"; do
        part=$(sanitize_suffix "$part")
        [[ -n "$part" ]] || continue
        joined+="${joined:+-}$part"
    done
    printf '%s' "$joined"
}

EMBEDDING=${EMBEDDING:-glove-840B}
MODEL=${MODEL:-}
SWEEP_FLAGS=()             # the flags of the sweep, in the order given
declare -A SWEEP_VALUES=() # flag -> its values, comma-separated ("" for a boolean flag)
SWEEP_DIMENSIONS=()        # the flags with several values

# The flags of a sweep, from the arguments after the model name.
parse_sweep_flags() {
    local flag value
    while (($# > 0)); do
        flag=$1
        [[ "$flag" == --* ]] || { echo "Expected a --flag, got '$flag'" >&2; usage; }
        shift
        value=""
        if (($# > 0)) && [[ "$1" != --* ]]; then
            value=$1
            shift
        fi
        [[ -z "${SWEEP_VALUES[$flag]+set}" ]] || { echo "Flag $flag given twice" >&2; exit 2; }
        SWEEP_FLAGS+=("$flag")
        SWEEP_VALUES[$flag]=$value
        [[ "$value" != *,* ]] || SWEEP_DIMENSIONS+=("$flag")
    done
}

if [[ "$TASK_MODE" == true ]]; then
    [[ "$PROFILE" != sweep ]] || parse_sweep_flags "$@"
else
case "$PROFILE" in
    train)
        MODELS=(affiliation-address citation date funding-acknowledgement header name-citation
                name-header reference-segmenter figure table segmentation)
        list_from_env "$ENV_ARCHITECTURES" "${DEFAULT_ARCHITECTURES[@]}"
        ;;
    train-eval)
        MODELS=(affiliation-address citation date funding-acknowledgement header name-citation
                name-header reference-segmenter figure table segmentation fulltext)
        list_from_env "$ENV_ARCHITECTURES" "${DEFAULT_ARCHITECTURES[@]}"
        ;;
    bert)
        MODELS=(affiliation-address citation date figure funding-acknowledgement header
                name-citation name-header reference-segmenter segmentation table)
        list_from_env "$ENV_TRANSFORMERS" "${DEFAULT_TRANSFORMERS[@]}"
        ;;
    bert-eval)
        MODELS=(affiliation-address citation date figure fulltext funding-acknowledgement header
                name-citation name-header quantities reference-segmenter segmentation table
                units values)
        list_from_env "$ENV_TRANSFORMERS" "${DEFAULT_TRANSFORMERS[@]}"
        ;;
    license)
        MODELS=(license)
        list_from_env "$ENV_ARCHITECTURES" gru
        ;;
    sweep)
        MODEL=${1:-}
        [[ -n "$MODEL" && "$MODEL" != --* ]] || usage
        shift
        parse_sweep_flags "$@"
        ;;
    *)
        usage
        ;;
esac
if [[ "$PROFILE" != sweep ]]; then
    [[ $# -le 1 ]] || usage
    EMBEDDING=${1:-$EMBEDDING}
    ITEMS=("${LIST[@]}")
    if [[ "$PROFILE" != license ]]; then
        list_from_env "$ENV_MODELS" "${MODELS[@]}"
        MODELS=("${LIST[@]}")
    fi
fi
fi

if [[ "$PROFILE" == sweep ]]; then
    TOTAL_TASKS=1
    for flag in "${SWEEP_DIMENSIONS[@]}"; do
        IFS=, read -r -a values <<<"${SWEEP_VALUES[$flag]}"
        TOTAL_TASKS=$((TOTAL_TASKS * ${#values[@]}))
    done
else
    TOTAL_TASKS=$((${#MODELS[@]} * ${#ITEMS[@]}))
fi

# Map an array index onto a training command, returned in the global CMD array.
build_task_command() {
    local index=$1

    if ((index < 0 || index >= TOTAL_TASKS)); then
        echo "Invalid array task index: $index (matrix has $TOTAL_TASKS tasks)" >&2
        exit 1
    fi

    if [[ "$PROFILE" == sweep ]]; then
        build_sweep_command "$index"
        return
    fi

    local model=${MODELS[$((index / ${#ITEMS[@]}))]}
    local item=${ITEMS[$((index % ${#ITEMS[@]}))]}
    local extra_args=()
    [[ "$INCREMENTAL" != true ]] || extra_args+=(--incremental)

    embedding_option "$EMBEDDING"
    if [[ "$PROFILE" == license ]]; then
        CMD=("$PYTHON_BIN" -m delft.applications.licenseClassifier train
             --architecture "$item" "${EMBEDDING_OPTION[@]}" "${extra_args[@]}")
        return
    fi

    local action=train
    if [[ "$PROFILE" == train-eval || "$PROFILE" == bert-eval ]]; then
        action=train_eval
        extra_args+=(--wandb)
    fi

    # The two largest models need more data loading workers; for plain training they also get
    # a longer sequence window.
    if [[ "$model" == header || "$model" == citation ]]; then
        extra_args+=(--num-workers 6)
        [[ "$PROFILE" != train ]] || extra_args+=(--max-sequence-length 3000)
    fi

    CMD=("$PYTHON_BIN" -m delft.applications.grobidTagger "$model" "$action")
    local suffix
    if [[ "$PROFILE" == bert || "$PROFILE" == bert-eval ]]; then
        # one model directory per transformer: grobid-header-BERT_CRF-ModernBERT-base
        suffix=$(join_suffix "${item##*/}")
        CMD+=(--architecture BERT_CRF --transformer "$item")
    else
        suffix=$(join_suffix)
        CMD+=(--architecture "$item" "${EMBEDDING_OPTION[@]}")
    fi
    [[ -z "$suffix" ]] || CMD+=(--suffix "$suffix")
    CMD+=("${extra_args[@]}")
}

# The command of task `index` of a sweep: the index is decomposed over the dimensions, the
# first one given varying slowest.
build_sweep_command() {
    local index=$1
    local -A chosen=()
    local flag values dimension value
    for ((dimension = ${#SWEEP_DIMENSIONS[@]} - 1; dimension >= 0; dimension--)); do
        flag=${SWEEP_DIMENSIONS[$dimension]}
        IFS=, read -r -a values <<<"${SWEEP_VALUES[$flag]}"
        chosen[$flag]=${values[$((index % ${#values[@]}))]}
        index=$((index / ${#values[@]}))
    done

    CMD=("$PYTHON_BIN" -m delft.applications.grobidTagger "$MODEL" "$ACTION")
    local suffix_parts=()
    for flag in "${SWEEP_FLAGS[@]}"; do
        if [[ -n "${chosen[$flag]+set}" ]]; then
            value=${chosen[$flag]}
            case "$flag" in
                --architecture) ;; # already in the model name
                --transformer) suffix_parts+=("${value##*/}") ;;
                --embedding) suffix_parts+=("$([[ "$value" == none ]] && echo no-embedding || echo "$value")") ;;
                *) suffix_parts+=("${flag//-/}$value") ;;
            esac
        else
            value=${SWEEP_VALUES[$flag]}
        fi
        # no --embedding at all trains without word embeddings
        [[ "$flag" == --embedding && "$value" == none ]] && continue
        CMD+=("$flag")
        [[ -z "$value" ]] || CMD+=("$value")
    done
    local suffix
    suffix=$(join_suffix "${suffix_parts[@]}")
    [[ -z "$suffix" ]] || CMD+=(--suffix "$suffix")
    if [[ "$WANDB" != false && -z "${SWEEP_VALUES[--wandb]+set}" ]]; then
        CMD+=(--wandb)
    fi
}

# Running as an array task: resolve our index and hand over to the training.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    build_task_command "$SLURM_ARRAY_TASK_ID"
    echo ">>> [task $SLURM_ARRAY_TASK_ID/$((TOTAL_TASKS - 1))] ${CMD[*]}"
    exec "${CMD[@]}"
fi

# Running on the login node: submit the array.
ARRAY_SPEC=${ARRAY_SPEC:-"0-$((TOTAL_TASKS - 1))"}
RUN_NAME="${PROFILE}${MODEL:+_$MODEL}"

if [[ "$DRY_RUN" == true ]]; then
    echo "Profile '$PROFILE'${MODEL:+ on $MODEL}: $TOTAL_TASKS tasks (array spec: $ARRAY_SPEC)"
    for ((i = 0; i < TOTAL_TASKS; i++)); do
        build_task_command "$i"
        printf '%3d  %s\n' "$i" "${CMD[*]}"
    done
    exit 0
fi

LOG_DIR=${LOG_DIR:-"${HOME}/slurm_logs/${RUN_NAME}_array_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "$LOG_DIR"

SBATCH_OPTS=()
if [[ -n "$CONTAINER_IMAGE" ]]; then
    SBATCH_OPTS+=(--container-image="$CONTAINER_IMAGE"
                  --container-workdir="$CONTAINER_WORKDIR"
                  --container-mounts="$CONTAINER_MOUNTS")
fi
# shellcheck disable=SC2206  # SBATCH_EXTRA is a string of options to split on spaces
SBATCH_OPTS+=(--export=ALL
              --cpus-per-task="$CPUS_PER_TASK"
              --mem="$MEMORY"
              -p "$PARTITIONS"
              --gpus=1
              --nodes=1
              --time="$TIME_LIMIT"
              --job-name="delft_$RUN_NAME"
              --array="${ARRAY_SPEC}%${MAX_PARALLEL_JOBS}"
              --output="$LOG_DIR/%A_%a.log"
              --error="$LOG_DIR/%A_%a.log"
              $SBATCH_EXTRA)

# the settings of the tasks, resolved here once: a task reads nothing from the environment
TASK_ARGS=(--profile "$PROFILE" --suffix "$SUFFIX" --python-bin "$PYTHON_BIN")
if [[ "$PROFILE" == sweep ]]; then
    TASK_ARGS+=(--model "$MODEL" --action "$ACTION" --wandb "$WANDB" --)
    for flag in "${SWEEP_FLAGS[@]}"; do
        TASK_ARGS+=("$flag")
        [[ -z "${SWEEP_VALUES[$flag]}" ]] || TASK_ARGS+=("${SWEEP_VALUES[$flag]}")
    done
else
    TASK_ARGS+=(--embedding "$EMBEDDING" --models "${MODELS[*]}" --items "${ITEMS[*]}" --incremental "$INCREMENTAL")
fi

job_id=$(sbatch --parsable "${SBATCH_OPTS[@]}" "$(readlink -f "$0")" __task "${TASK_ARGS[@]}" | cut -d';' -f1)

echo "Submitted array $job_id: tasks $ARRAY_SPEC of $TOTAL_TASKS (max $MAX_PARALLEL_JOBS concurrent)"
echo "Logs:    $LOG_DIR"
echo "Monitor: squeue -j $job_id"
