#!/bin/bash
# Distributed training script for license classifier (copyright + license models)
# Uses GRU architecture with multi-GPU support
#
# Generated for DeLFT

set -e  # Exit on error

# Common SLURM configuration for multi-GPU training
SRUN_OPTS="--container-mounts=/netscratch:/netscratch,$HOME:$HOME \
--container-workdir=/netscratch/lfoppiano/delft \
--container-image=/netscratch/lfoppiano/enroot/tensorflow-2.17.2-gpu-delft-updated.sqsh \
--mem=200G \
-p V100-32GB,RTX3090,RTXA6000 \
--gpus=4 \
--nodes=1 \
--time=3-00:00"

PYTHON_CMD=".venv/bin/python -m delft.applications.licenseClassifier"

# Training parameters
ARCHITECTURE="gru"
FOLD_COUNT=4
NUM_WORKERS=1

echo "=========================================="
echo "Distributed training - License classifier"
echo "Architecture: ${ARCHITECTURE}"
echo "Folds: ${FOLD_COUNT}"
echo "GPUs: 4"
echo "=========================================="

srun $SRUN_OPTS $PYTHON_CMD train \
    --architecture ${ARCHITECTURE} \
    --num-workers ${NUM_WORKERS}

echo "=========================================="
echo "Training completed"
echo "=========================================="
