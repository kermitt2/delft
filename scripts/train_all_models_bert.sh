#!/bin/bash
# Training script for all GROBID models with BERT_CRF architecture (training only)
# Generated on 2025-12-27
#
# This script trains all available models with BERT_CRF using various transformers:
# - SciBERT (cased and uncased)
# - ModernBERT
# - DeBERTa
# - LinkBERT

set -e  # Exit on error

# Common SLURM configuration
SRUN_OPTS="--container-mounts=/netscratch:/netscratch,$HOME:$HOME \
--container-workdir=/netscratch/lfoppiano/delft/delft-pytorch \
--container-image=/netscratch/lfoppiano/enroot/delft-pytorch.sqsh \
--mem=100G \
-p V100-32GB,RTX3090,RTXA6000,L40S \
--gpus=1 \
--nodes=1 \
--time=3-00:00"

PYTHON_CMD=".venv/bin/python -m delft.applications.grobidTagger"

# Transformer models to use with BERT_CRF
TRANSFORMERS=(
    "allenai/scibert_scivocab_cased"
    "allenai/scibert_scivocab_uncased"
    "answerdotai/ModernBERT-base"
    "microsoft/deberta-v3-base"
    "michiyasunaga/LinkBERT-base"
)

# Models available in data/sequenceLabelling/grobid
MODELS=(
    "affiliation-address"
    "citation"
    "date"
    "figure"
    # "fulltext"
    "funding-acknowledgement"
    "header"
    "name-citation"
    "name-header"
    "reference-segmenter"
    "segmentation"
    "table"
)

# Function to train a model with a specific transformer
train_model() {
    local model=$1
    local transformer=$2
    
    echo "=========================================="
    echo "Training model: $model"
    echo "Architecture: BERT_CRF"
    echo "Transformer: $transformer"
    echo "=========================================="
    
    srun $SRUN_OPTS $PYTHON_CMD $model train \
        --architecture BERT_CRF \
        --transformer $transformer
    
    echo "Completed: $model with BERT_CRF ($transformer)"
    echo ""
}

# Main training loop
echo "Starting training of all GROBID models with BERT_CRF..."
echo "Total models: ${#MODELS[@]}"
echo "Transformers: ${#TRANSFORMERS[@]}"
echo "Total training runs: $((${#MODELS[@]} * ${#TRANSFORMERS[@]}))"
echo ""

for model in "${MODELS[@]}"; do
    for transformer in "${TRANSFORMERS[@]}"; do
        train_model "$model" "$transformer"
    done
done

echo "=========================================="
echo "All BERT_CRF training runs completed!"
echo "=========================================="
