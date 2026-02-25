#!/bin/bash
# Distributed training script for license classifier (copyright + license models)
# Uses GRU architecture
#
# Generated for DeLFT

set -e  # Exit on error

# Common SLURM configuration
SBATCH_OPTS="--container-mounts=/netscratch:/netscratch,$HOME:$HOME \
--container-workdir=/netscratch/lfoppiano/delft \
--container-image=/netscratch/lfoppiano/enroot/tensorflow-2.17.2-gpu-delft-updated.sqsh \
--mem=100G \
-p V100-32GB,RTX3090,RTXA6000 \
--gpus=1 \
--nodes=1 \
--time=3-00:00:00"

sbatch $SBATCH_OPTS --job-name=license_classifier <<'EOF'
#!/bin/bash
set -e  # Exit on error

PYTHON_CMD=".venv/bin/python -m delft.applications.licenseClassifier"

# Training parameters
ARCHITECTURE="gru"
NUM_WORKERS=1

echo "=========================================="
echo "Training license classifier"
echo "Architecture: ${ARCHITECTURE}"
echo "=========================================="

srun $PYTHON_CMD train \
    --architecture ${ARCHITECTURE} \
    --num-workers ${NUM_WORKERS}

echo "=========================================="
echo "Training completed"
echo "=========================================="
EOF
