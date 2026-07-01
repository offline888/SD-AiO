#!/bin/bash
# Stage 1 — 3D Classifier (haze / rain / noise) with Focal Loss
set -euo pipefail

source /base/mambaforge/etc/profile.d/conda.sh
conda activate allinone
cd /root/shared-nvme/SD-AiO

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="output/stage1_3d_${TIMESTAMP}"
mkdir -p "${OUT_DIR}"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== Stage 1: 3D Classifier (Focal Loss) ==="
echo "  Output: ${OUT_DIR}"

accelerate launch --num_processes=1 --mixed_precision=bf16 src/train_classifier.py \
    --task_config              configs/tasks_3d.yaml \
    --dino_type                /root/shared-nvme/model/dinov2 \
    --output_dir               "${OUT_DIR}" \
    --num_deg_types             3 \
    --image_size               256 \
    --train_batch_size         32 \
    --learning_rate_backbone    1e-5 \
    --learning_rate_head       1e-4 \
    --max_train_steps          20000 \
    --seed                     42 \
    --num_workers              8 \
    --focal_gamma              2.0 \
    --eval_freq                100 \
    --checkpointing_steps      500 \
    > "${OUT_DIR}/train.log" 2>&1

echo "=== Done (exit=$?) ==="
