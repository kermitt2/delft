#!/bin/bash
# Distributed training script for all GROBID models with multiple architectures
# Uses sbatch for parallel job submission to fully exploit the cluster
#
# This script trains all available models with the following architectures:
# - BidLSTM_CRF
# - BidLSTM_CRF_FEATURES
# - BidLSTM_ChainCRF
# - BidLSTM_ChainCRF_FEATURES

set -e

# Parallelization settings
MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS:-4}
WAIT_INTERVAL=${WAIT_INTERVAL:-30}

# Common SLURM configuration
SBATCH_OPTS="--container-mounts=/netscratch:/netscratch,$HOME:$HOME \
--container-workdir=/netscratch/lfoppiano/delft/delft_tf2.17.1-updated \
--container-image=/netscratch/lfoppiano/enroot/tensorflow-2.17.2-gpu-delft-updated.sqsh \
--mem=100G \
-p V100-32GB,RTX3090,RTXA6000 \
--gpus=1 \
--nodes=1 \
--time=3-00:00"

PYTHON_CMD=".venv/bin/python -m delft.applications.grobidTagger"

# Architectures to train
ARCHITECTURES=(
    "BidLSTM_CRF"
    "BidLSTM_CRF_FEATURES"
    "BidLSTM_ChainCRF"
    "BidLSTM_ChainCRF_FEATURES"
)

# Models available in data/sequenceLabelling/grobid
MODELS=(
    "affiliation-address"
    "citation"
    "date"
    "figure"
    "funding-acknowledgement"
    "header"
    "name-citation"
    "name-header"
    "reference-segmenter"
    "table"
)

# Log directory for job outputs
LOG_DIR="${HOME}/slurm_logs/train_distributed_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Track submitted job IDs
declare -a JOB_IDS

# Function to count running jobs for this experiment set
count_running_jobs() {
    local count=0
    for job_id in "${JOB_IDS[@]}"; do
        if squeue -j "$job_id" &>/dev/null 2>&1; then
            state=$(squeue -j "$job_id" -h -o "%t" 2>/dev/null)
            if [[ "$state" == "R" || "$state" == "PD" ]]; then
                count=$((count + 1))
            fi
        fi
    done
    echo $count
}

# Function to wait until we have capacity for more jobs
wait_for_capacity() {
    while true; do
        running=$(count_running_jobs)
        if [[ $running -lt $MAX_PARALLEL_JOBS ]]; then
            break
        fi
        echo "Currently $running jobs running/pending (max: $MAX_PARALLEL_JOBS). Waiting..."
        sleep $WAIT_INTERVAL
    done
}

# Function to submit a training job
submit_job() {
    local model=$1
    local architecture=$2
    local experiment_id=$3

    local job_name="train_${model}_${architecture}"
    local log_file="${LOG_DIR}/${job_name}_%j.log"

    echo ">>> Submitting experiment $experiment_id: $job_name"

    if [[ "$model" == "header" ]]; then
        job_id=$(sbatch $SBATCH_OPTS \
            --job-name="$job_name" \
            --output="$log_file" \
            --error="$log_file" \
            --wrap="$PYTHON_CMD $model train --architecture $architecture --num-workers 6" 2>&1 | grep -oP '\d+')
    else
        job_id=$(sbatch $SBATCH_OPTS \
            --job-name="$job_name" \
            --output="$log_file" \
            --error="$log_file" \
            --wrap="$PYTHON_CMD $model train --architecture $architecture" 2>&1 | grep -oP '\d+')
    fi

    if [[ -n "$job_id" ]]; then
        JOB_IDS+=("$job_id")
        echo "    Submitted job ID: $job_id"
    else
        echo "    Warning: Failed to submit job for $job_name"
    fi
}

submit_job_incremental() {
    local model=$1
    local architecture=$2
    local experiment_id=$3

    local job_name="train_inc_${model}_${architecture}"
    local log_file="${LOG_DIR}/${job_name}_%j.log"

    echo ">>> Submitting incremental experiment $experiment_id: $job_name"

    if [[ "$model" == "header" ]] || [[ "$model" == "citation" ]]; then
        job_id=$(sbatch $SBATCH_OPTS \
            --job-name="$job_name" \
            --output="$log_file" \
            --error="$log_file" \
            --wrap="$PYTHON_CMD $model train --architecture $architecture --incremental --num-workers 6" 2>&1 | grep -oP '\d+')
    else
        job_id=$(sbatch $SBATCH_OPTS \
            --job-name="$job_name" \
            --output="$log_file" \
            --error="$log_file" \
            --wrap="$PYTHON_CMD $model train --architecture $architecture --incremental" 2>&1 | grep -oP '\d+')
    fi

    if [[ -n "$job_id" ]]; then
        JOB_IDS+=("$job_id")
        echo "    Submitted incremental job ID: $job_id"
    else
        echo "    Warning: Failed to submit incremental job for $job_name"
    fi
}

# Calculate total number of experiments
total_experiments=$((${#MODELS[@]} * ${#ARCHITECTURES[@]}))

# Main submission loop
echo "==========================================="
echo "Starting distributed training of all GROBID models"
echo "Total experiments: $total_experiments"
echo "Max parallel jobs: $MAX_PARALLEL_JOBS"
echo "Log directory: $LOG_DIR"
echo "==========================================="
echo ""

experiment_count=0

for model in "${MODELS[@]}"; do
    for arch in "${ARCHITECTURES[@]}"; do
        experiment_count=$((experiment_count + 1))
        wait_for_capacity
        submit_job "$model" "$arch" "$experiment_count"
    done
done

echo ""
echo "==========================================="
echo "All $total_experiments experiments submitted!"
echo "Job IDs: ${JOB_IDS[*]}"
echo "Monitor with: squeue -u \$USER"
echo "Logs in: $LOG_DIR"
echo "==========================================="

# Optional: Wait for all jobs to complete
if [[ "${WAIT_FOR_COMPLETION:-false}" == "true" ]]; then
    echo ""
    echo "Waiting for all jobs to complete..."
    for job_id in "${JOB_IDS[@]}"; do
        while squeue -j "$job_id" &>/dev/null 2>&1; do
            sleep $WAIT_INTERVAL
        done
    done
    echo "All jobs completed!"

    echo ""
    echo "==========================================="
    echo "Starting incremental training phase"
    echo "==========================================="
    echo ""

    experiment_count=0

    for model in "${MODELS[@]}"; do
        for arch in "${ARCHITECTURES[@]}"; do
            experiment_count=$((experiment_count + 1))
            wait_for_capacity
            submit_job_incremental "$model" "$arch" "$experiment_count"
        done
    done

    echo ""
    echo "==========================================="
    echo "All $total_experiments incremental experiments submitted!"
    echo "Job IDs: ${JOB_IDS[*]}"
    echo "Monitor with: squeue -u \$USER"
    echo "Logs in: $LOG_DIR"
    echo "==========================================="
fi