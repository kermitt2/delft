  #!/bin/bash
# Distributed training script for the citation GROBID model with multiple architectures
# Uses sbatch for parallel job submission to fully exploit the cluster
#
# This script trains the citation model with the following architectures:
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

MODEL="citation"

# Log directory for job outputs
LOG_DIR="${HOME}/slurm_logs/train_citation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Track submitted job IDs
declare -a JOB_IDS

# Function to submit a training job
submit_job() {
    local architecture=$1
    local experiment_id=$2

    local job_name="train_${MODEL}_${architecture}"
    local log_file="${LOG_DIR}/${job_name}_%j.log"

    echo ">>> Submitting experiment $experiment_id: $job_name"

    job_id=$(sbatch $SBATCH_OPTS \
        --job-name="$job_name" \
        --output="$log_file" \
        --error="$log_file" \
        --wrap="$PYTHON_CMD $MODEL train --architecture $architecture --num-workers 6 --max-sequence-length 3000" 2>&1 | grep -oP '\d+')

    if [[ -n "$job_id" ]]; then
        JOB_IDS+=("$job_id")
        echo "    Submitted job ID: $job_id"
    else
        echo "    Warning: Failed to submit job for $job_name"
    fi
}

# Main submission loop
total_experiments=${#ARCHITECTURES[@]}

echo "==========================================="
echo "Starting distributed training of citation model"
echo "Total experiments: $total_experiments"
echo "Max parallel jobs: $MAX_PARALLEL_JOBS"
echo "Log directory: $LOG_DIR"
echo "==========================================="
echo ""

experiment_count=0

for arch in "${ARCHITECTURES[@]}"; do
    experiment_count=$((experiment_count + 1))
    submit_job "$arch" "$experiment_count"
done

echo ""
echo "==========================================="
echo "All $total_experiments experiments submitted!"
echo "Job IDs: ${JOB_IDS[*]}"
echo "Monitor with: squeue -u \$USER"
echo "Logs in: $LOG_DIR"
echo "==========================================="
