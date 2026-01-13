#!/bin/bash
# Training script for header model with parameter experimentation
# Generated on 2026-01-08
#
# This script trains the header model with different hyperparameter combinations
# using the following architectures:
# - BidLSTM_CRF_FEATURES
# - BidLSTM_ChainCRF_FEATURES
#
# Jobs are submitted in parallel using sbatch for faster execution.

set -e  # Exit on error

# Parallelization settings
MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS:-4}  # Maximum concurrent jobs (adjust based on cluster capacity)
WAIT_INTERVAL=${WAIT_INTERVAL:-30}          # Seconds to wait between checking job count

# Common SLURM configuration
SBATCH_OPTS="--container-mounts=/netscratch:/netscratch,$HOME:$HOME \
--container-workdir=/netscratch/lfoppiano/delft/delft-pytorch \
--container-image=/netscratch/lfoppiano/enroot/delft-pytorch.sqsh \
--mem=100G \
-p V100-32GB,RTX3090,RTXA6000 \
--gpus=1 \
--nodes=1 \
--time=3-00:00"

PYTHON_CMD=".venv/bin/python -m delft.applications.grobidTagger"

# Model to train
MODEL="header"

# Architectures to experiment with
ARCHITECTURES=(
    "BidLSTM_CRF_FEATURES"
    "BidLSTM_ChainCRF_FEATURES"
)

# Hyperparameter grid
LEARNING_RATE="1e-4"  # Fixed learning rate
BATCH_SIZES=(4 8 16 32)
MAX_EPOCHS=(50 100)
PATIENCE_VALUES=(5 10 15)
EARLY_STOP_VALUES=(true false)

# Log directory for job outputs
LOG_DIR="${HOME}/slurm_logs/header_experiments_$(date +%Y%m%d_%H%M%S)"
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
    local architecture=$1
    local batch_size=$2
    local max_epoch=$3
    local patience=$4
    local early_stop=$5
    local experiment_id=$6

    local job_name="header_${architecture}_bs${batch_size}_ep${max_epoch}_p${patience}_es${early_stop}"
    local log_file="${LOG_DIR}/${job_name}_%j.log"

    echo ">>> Submitting experiment $experiment_id: $job_name"

    job_id=$(sbatch $SBATCH_OPTS \
        --job-name="$job_name" \
        --output="$log_file" \
        --error="$log_file" \
        --wrap="$PYTHON_CMD $MODEL train_eval \
            --architecture $architecture \
            --learning-rate $LEARNING_RATE \
            --batch-size $batch_size \
            --max-epoch $max_epoch \
            --patience $patience \
            --early-stop $early_stop \
            --wandb" 2>&1 | grep -oP '\d+')

    if [[ -n "$job_id" ]]; then
        JOB_IDS+=("$job_id")
        echo "    Submitted job ID: $job_id"
    else
        echo "    Warning: Failed to submit job for $job_name"
    fi
}

# Calculate total number of experiments
total_experiments=$((${#ARCHITECTURES[@]} * ${#BATCH_SIZES[@]} * ${#MAX_EPOCHS[@]} * ${#PATIENCE_VALUES[@]} * ${#EARLY_STOP_VALUES[@]}))

# Main submission loop
echo "==========================================="
echo "Starting header model parameter experimentation"
echo "Total experiments: $total_experiments"
echo "Max parallel jobs: $MAX_PARALLEL_JOBS"
echo "Log directory: $LOG_DIR"
echo "==========================================="
echo ""

experiment_count=0

for arch in "${ARCHITECTURES[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
        for epochs in "${MAX_EPOCHS[@]}"; do
            for patience in "${PATIENCE_VALUES[@]}"; do
                for early_stop in "${EARLY_STOP_VALUES[@]}"; do
                    experiment_count=$((experiment_count + 1))
                    
                    # Wait if we've hit the max parallel jobs
                    wait_for_capacity
                    
                    submit_job "$arch" "$bs" "$epochs" "$patience" "$early_stop" "$experiment_count"
                done
            done
        done
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
fi
