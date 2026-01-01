#!/bin/bash
# Training script for all GROBID models with multiple architectures (training only)
# Generated on 2025-12-27
#
# This script trains all available models with the following architectures:
# - BidLSTM_CRF
# - BidLSTM_CRF_FEATURES
# - BidLSTM_ChainCRF
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
    "segmentation"
    "table"
)

# Function to train a model with a specific architecture
train_model() {
    local model=$1
    local architecture=$2
    
    echo "=========================================="
    echo "Training model: $model"
    echo "Architecture: $architecture"
    echo "=========================================="
    
    srun $SRUN_OPTS $PYTHON_CMD $model train \
        --architecture $architecture \
        --wandb
    
    echo "Completed: $model with $architecture"
    echo ""
}

# Main training loop
echo "Starting training of all GROBID models..."
echo "Total models: ${#MODELS[@]}"
echo "Architectures per model: ${#ARCHITECTURES[@]}"
echo "Total training runs: $((${#MODELS[@]} * ${#ARCHITECTURES[@]}))"
echo ""

for model in "${MODELS[@]}"; do
    for arch in "${ARCHITECTURES[@]}"; do
        train_model "$model" "$arch"
    done
done

echo "=========================================="
echo "All training runs completed!"
echo "=========================================="
