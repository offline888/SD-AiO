#!/bin/bash
set -euo pipefail

source /base/mambaforge/etc/profile.d/conda.sh
conda activate allinone
cd /root/shared-nvme/SD-AiO

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="output/ablation/t150_${TIMESTAMP}"
mkdir -p "${OUT_DIR}/checkpoints" "${OUT_DIR}/eval"

CLASSIFIER="/root/shared-nvme/SD-AiO/output/stage1_3d_focal_20260620_172503/best_model.pth"

export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_TIMEOUT=3600

echo "=== t=150 + simple-conv + VAE LoRA + 2×4090 ==="
echo "  Output:     ${OUT_DIR}"

accelerate launch --num_processes=2 --mixed_precision=bf16 src/train.py \
    --data_config                 configs/tasks_3d.yaml \
    --output_dir                  "${OUT_DIR}" \
    --sd_path                     /root/shared-nvme/model/sd2-1 \
    --condition_type              deg-aware \
    --backbone_type               simple-conv \
    --timestep_value             150 \
    --dino_type                   /root/shared-nvme/model/dinov2 \
    --degradation_classifier_path  "${CLASSIFIER}" \
    --enable_lora \
    --lora_rank_vae               16 \
    --lambda_l2                   2.0 \
    --lambda_lpips                5.0 \
    --learning_rate               1e-4 \
    --lr_scheduler                cosine \
    --lr_warmup_steps            500 \
    --adam_weight_decay           0.0 \
    --max_grad_norm               1.0 \
    --gradient_accumulation_steps 16 \
    --max_train_steps             5000 \
    --mixed_precision             bf16 \
    --seed                        42 \
    --checkpointing_steps         2000 \
    --eval_freq                   500 \
    --num_images_save_eval        15 \
    > "${OUT_DIR}/train.log" 2>&1

echo "=== Done (exit=$?) ==="
