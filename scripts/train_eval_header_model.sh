#!/bin/bash
# Training script for header model with parameter experimentation
# Generated on 2026-01-08
#
# This script trains the header model with different hyperparameter combinations
# using the following architectures:
# - BidLSTM_CRF_FEATURES
# - BidLSTM_ChainCRF_FEATURES

set -e  # Exit on error

# Common SLURM configuration
SRUN_OPTS="--container-mounts=/netscratch:/netscratch,$HOME:$HOME \
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
LEARNING_RATES=(1e-4 5e-5 1e-5)
BATCH_SIZES=(4 8 16 32)
MAX_EPOCHS=(50 100)
PATIENCE_VALUES=(5 10 15)
EARLY_STOP_VALUES=(true false)

# Function to train a model with specific parameters
train_model() {
    local architecture=$1
    local learning_rate=$2
    local batch_size=$3
    local max_epoch=$4
    local patience=$5
    local early_stop=$6

    echo "=========================================="
    echo "Training model: $MODEL"
    echo "Architecture: $architecture"
    echo "Learning rate: $learning_rate"
    echo "Batch size: $batch_size"
    echo "Max epochs: $max_epoch"
    echo "Patience: $patience"
    echo "Early stop: $early_stop"
    echo "=========================================="

    srun $SRUN_OPTS $PYTHON_CMD $MODEL train_eval \
        --architecture $architecture \
        --learning-rate $learning_rate \
        --batch-size $batch_size \
        --max-epoch $max_epoch \
        --patience $patience \
        --early-stop $early_stop \
        --wandb

    echo "Completed: $MODEL with arch=$architecture lr=$learning_rate bs=$batch_size epochs=$max_epoch patience=$patience early_stop=$early_stop"
    echo ""
}

# Calculate total number of experiments
total_experiments=$((${#ARCHITECTURES[@]} * ${#LEARNING_RATES[@]} * ${#BATCH_SIZES[@]} * ${#MAX_EPOCHS[@]} * ${#PATIENCE_VALUES[@]} * ${#EARLY_STOP_VALUES[@]}))

# Main training loop
echo "Starting header model parameter experimentation..."
echo "Total experiments: $total_experiments"
echo ""

experiment_count=0

for arch in "${ARCHITECTURES[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        for bs in "${BATCH_SIZES[@]}"; do
            for epochs in "${MAX_EPOCHS[@]}"; do
                for patience in "${PATIENCE_VALUES[@]}"; do
                    for early_stop in "${EARLY_STOP_VALUES[@]}"; do
                        experiment_count=$((experiment_count + 1))
                        echo ">>> Experiment $experiment_count / $total_experiments"
                        train_model "$arch" "$lr" "$bs" "$epochs" "$patience" "$early_stop"
                    done
                done
            done
        done
    done
done

echo "=========================================="
echo "All $total_experiments experiments completed!"
echo "=========================================="

