#!/bin/bash
# Distributed training launcher for DeLFT
#
# Usage:
#   ./train_distributed.sh [NUM_GPUS] [TRAINING_COMMAND...]
#
# Examples:
#   # Auto-detect GPUs and train
#   ./train_distributed.sh python -m delft.applications.grobidTagger name-header train --multi-gpu
#
#   # Specify 2 GPUs
#   ./train_distributed.sh 2 python -m delft.applications.grobidTagger name-header train --multi-gpu
#
#   # Using module syntax
#   ./train_distributed.sh -m delft.applications.grobidTagger name-header train --multi-gpu

set -e

# Check if first argument is a number (GPU count)
if [[ "$1" =~ ^[0-9]+$ ]]; then
    NGPUS=$1
    shift
else
    # Auto-detect number of GPUs
    NGPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo "1")
fi

if [ "$NGPUS" -lt 1 ]; then
    echo "Error: No GPUs detected"
    exit 1
fi

echo "Launching distributed training with $NGPUS GPU(s)"
echo "Command: $@"
echo ""

# Launch with torchrun
# --standalone: Single-node training
# --nproc_per_node: Number of processes (GPUs) per node
exec torchrun \
    --standalone \
    --nproc_per_node="$NGPUS" \
    "$@"
