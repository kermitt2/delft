#!/bin/bash
# Submit a standard training matrix as a throttled SLURM job array.
#
# Usage: train_distributed_array.sh {train|train-eval|bert|bert-eval|header} [EMBEDDING]
#
# Environment overrides:
#   MAX_PARALLEL_JOBS  maximum number of array tasks running at once (default: 4)
#   ARRAY_SPEC         SLURM array index spec, e.g. "3,7,12-15" to re-run a few failed
#                      tasks (default: the whole matrix, "0-<N-1>")
#   DRY_RUN            when "true", print every task command instead of submitting
#   LOG_DIR            log directory (default: ~/slurm_logs/<profile>_array_<timestamp>)
#   PYTHON_BIN         interpreter to use (default: .venv/bin/python)
#   LEARNING_RATE      header profile only (default: 1e-3)

set -euo pipefail

PROFILE=${1:-}
EMBEDDING=${2:-glove-840B}
MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS:-4}
DRY_RUN=${DRY_RUN:-false}
PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
PYTHON_CMD=("$PYTHON_BIN" -m delft.applications.grobidTagger)

ARCHITECTURES=(
    BidLSTM_CRF
    BidLSTM_CRF_FEATURES
    BidLSTM_ChainCRF
    BidLSTM_ChainCRF_FEATURES
)
TRANSFORMERS=(
    allenai/scibert_scivocab_cased
    allenai/scibert_scivocab_uncased
    answerdotai/ModernBERT-base
    microsoft/deberta-v3-base
    michiyasunaga/LinkBERT-base
)

case "$PROFILE" in
    train)
        MODELS=(affiliation-address citation date funding-acknowledgement header name-citation
                name-header reference-segmenter figure table segmentation)
        ITEMS_PER_MODEL=${#ARCHITECTURES[@]}
        ;;
    train-eval)
        MODELS=(affiliation-address citation date funding-acknowledgement header name-citation
                name-header reference-segmenter figure table segmentation fulltext)
        ITEMS_PER_MODEL=${#ARCHITECTURES[@]}
        ;;
    bert)
        MODELS=(affiliation-address citation date figure funding-acknowledgement header
                name-citation name-header reference-segmenter segmentation table)
        ITEMS_PER_MODEL=${#TRANSFORMERS[@]}
        ;;
    bert-eval)
        MODELS=(affiliation-address citation date figure fulltext funding-acknowledgement header
                name-citation name-header quantities reference-segmenter segmentation table
                units values)
        ITEMS_PER_MODEL=${#TRANSFORMERS[@]}
        ;;
    header)
        # Hyperparameter sweep on the header model. "disabled" means --early-stop false, for
        # which the patience value is irrelevant (the tagger default of 5 is passed).
        ARCHITECTURES=(BidLSTM_CRF_FEATURES BidLSTM_ChainCRF_FEATURES)
        BATCH_SIZES=(4 8 16 32)
        MAX_EPOCHS=(50 100)
        PATIENCE_VALUES=(5 10 15 disabled)
        TOTAL_TASKS=$((${#ARCHITECTURES[@]} * ${#BATCH_SIZES[@]} * ${#MAX_EPOCHS[@]} * ${#PATIENCE_VALUES[@]}))
        ;;
    *)
        echo "Usage: $0 {train|train-eval|bert|bert-eval|header} [EMBEDDING]" >&2
        exit 2
        ;;
esac

if [[ "$PROFILE" != header ]]; then
    TOTAL_TASKS=$((${#MODELS[@]} * ITEMS_PER_MODEL))
fi

# Map an array index onto a training command, returned in the global CMD array.
build_task_command() {
    local index=$1

    if ((index < 0 || index >= TOTAL_TASKS)); then
        echo "Invalid array task index: $index (matrix has $TOTAL_TASKS tasks)" >&2
        exit 1
    fi

    CMD=("${PYTHON_CMD[@]}")

    if [[ "$PROFILE" == header ]]; then
        local patience=${PATIENCE_VALUES[$((index % ${#PATIENCE_VALUES[@]}))]}
        index=$((index / ${#PATIENCE_VALUES[@]}))
        local max_epoch=${MAX_EPOCHS[$((index % ${#MAX_EPOCHS[@]}))]}
        index=$((index / ${#MAX_EPOCHS[@]}))
        local batch_size=${BATCH_SIZES[$((index % ${#BATCH_SIZES[@]}))]}
        index=$((index / ${#BATCH_SIZES[@]}))
        local architecture=${ARCHITECTURES[$index]}

        local early_stop=true
        if [[ "$patience" == disabled ]]; then
            patience=5
            early_stop=false
        fi

        CMD+=(header train_eval
              --architecture "$architecture"
              --embedding "$EMBEDDING"
              --learning-rate "${LEARNING_RATE:-1e-3}"
              --batch-size "$batch_size"
              --max-epoch "$max_epoch"
              --patience "$patience"
              --early-stop "$early_stop"
              --wandb)
        return
    fi

    local model=${MODELS[$((index / ITEMS_PER_MODEL))]}
    local item_index=$((index % ITEMS_PER_MODEL))

    local action=train
    local extra_args=()
    if [[ "$PROFILE" == train-eval || "$PROFILE" == bert-eval ]]; then
        action=train_eval
        extra_args+=(--wandb)
    fi

    if [[ "$PROFILE" == bert || "$PROFILE" == bert-eval ]]; then
        CMD+=("$model" "$action"
              --architecture BERT_CRF
              --transformer "${TRANSFORMERS[$item_index]}"
              "${extra_args[@]}")
        return
    fi

    # The two largest models need more data loading workers; for plain training they also get a
    # longer sequence window.
    if [[ "$model" == header || "$model" == citation ]]; then
        extra_args+=(--num-workers 6)
        if [[ "$PROFILE" == train ]]; then
            extra_args+=(--max-sequence-length 3000)
        fi
    fi

    CMD+=("$model" "$action"
          --architecture "${ARCHITECTURES[$item_index]}"
          --embedding "$EMBEDDING"
          "${extra_args[@]}")
}

# Running as an array task: resolve our index and hand over to the tagger.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    build_task_command "$SLURM_ARRAY_TASK_ID"
    echo ">>> [task $SLURM_ARRAY_TASK_ID/$((TOTAL_TASKS - 1))] ${CMD[*]}"
    exec "${CMD[@]}"
fi

# Running on the login node: submit the array.
ARRAY_SPEC=${ARRAY_SPEC:-"0-$((TOTAL_TASKS - 1))"}

if [[ "$DRY_RUN" == true ]]; then
    echo "Profile '$PROFILE': $TOTAL_TASKS tasks (array spec: $ARRAY_SPEC)"
    for ((i = 0; i < TOTAL_TASKS; i++)); do
        build_task_command "$i"
        printf '%3d  %s\n' "$i" "${CMD[*]}"
    done
    exit 0
fi

TIME_LIMIT=3-00:00
if [[ "$PROFILE" == header ]]; then
    TIME_LIMIT=1-00:00
fi

LOG_DIR=${LOG_DIR:-"${HOME}/slurm_logs/${PROFILE}_array_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "$LOG_DIR"

job_id=$(sbatch \
    --container-mounts="/netscratch:/netscratch,$HOME:$HOME" \
    --container-workdir=/netscratch/lfoppiano/delft/delft-pytorch2 \
    --container-image=/netscratch/lfoppiano/enroot/delft-pytorch.sqsh \
    --export=ALL \
    --mem=100G \
    -p RTX3090,RTXA6000,RTXB6000,L40S \
    --gpus=1 \
    --nodes=1 \
    --time="$TIME_LIMIT" \
    --job-name="delft_${PROFILE}" \
    --array="${ARRAY_SPEC}%${MAX_PARALLEL_JOBS}" \
    --output="$LOG_DIR/%A_%a.log" \
    --error="$LOG_DIR/%A_%a.log" \
    "$(readlink -f "$0")" "$PROFILE" "$EMBEDDING" | grep -oP '\d+')

echo "Submitted array $job_id: tasks $ARRAY_SPEC of $TOTAL_TASKS (max $MAX_PARALLEL_JOBS concurrent)"
echo "Logs:    $LOG_DIR"
echo "Monitor: squeue -j $job_id"
