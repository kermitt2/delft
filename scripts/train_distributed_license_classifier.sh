#!/bin/bash
# Distributed training script for license classifier (copyright + license models)
# Uses GRU architecture
#
# Submits training as a SLURM job using sbatch --wrap pattern

set -e

# Common SLURM configuration
SBATCH_OPTS="--container-mounts=/netscratch:/netscratch,$HOME:$HOME \
--container-workdir=/netscratch/lfoppiano/delft/delft_tf2.17.1-updated \
--container-image=/netscratch/lfoppiano/enroot/tensorflow-2.17.2-gpu-delft-updated.sqsh \
--mem=100G \
-p V100-32GB,RTX3090,RTXA6000 \
--gpus=1 \
--nodes=1 \
--time=3-00:00"

PYTHON_CMD=".venv/bin/python -m delft.applications.licenseClassifier"

# Training parameters
ARCHITECTURE="gru"

# Log directory for job outputs
LOG_DIR="${HOME}/slurm_logs/train_license_classifier_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Track submitted job IDs
declare -a JOB_IDS

JOB_NAME="train_license_classifier_${ARCHITECTURE}"
LOG_FILE="${LOG_DIR}/${JOB_NAME}_%j.log"

echo "==========================================="
echo "Submitting license classifier training"
echo "Architecture: ${ARCHITECTURE}"
echo "Log directory: $LOG_DIR"
echo "==========================================="
echo ""

echo ">>> Submitting: $JOB_NAME"

job_id=$(sbatch $SBATCH_OPTS \
    --job-name="$JOB_NAME" \
    --output="$LOG_FILE" \
    --error="$LOG_FILE" \
    --wrap="$PYTHON_CMD train --architecture $ARCHITECTURE" 2>&1 | grep -oP '\d+')

if [[ -n "$job_id" ]]; then
    JOB_IDS+=("$job_id")
    echo "    Submitted job ID: $job_id"
else
    echo "    Warning: Failed to submit job for $JOB_NAME"
fi

echo ""
echo "==========================================="
echo "All experiments submitted!"
echo "Job IDs: ${JOB_IDS[*]}"
echo "Monitor with: squeue -u \$USER"
echo "Logs in: $LOG_DIR"
echo "==========================================="
