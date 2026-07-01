#!/bin/bash
set -euo pipefail

source /base/mambaforge/etc/profile.d/conda.sh
conda activate allinone
cd /root/shared-nvme/SD-AiO

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="output/ablation_roundrobin/gan_${TIMESTAMP}"
mkdir -p "${OUT_DIR}/checkpoints" "${OUT_DIR}/eval"

# ── Environment ──────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_TIMEOUT=3600
export PYTHONPATH="/root/.cache/torch/hub/facebookresearch_dino_main:${PYTHONPATH:-}"

CLASSIFIER="/root/shared-nvme/SD-AiO/output/stage1_3d_focal_20260620_172503/best_model.pth"
SD_PATH="/root/shared-nvme/model/sd2-1"
DINO_PATH="/root/shared-nvme/model/dinov2"

# ── Data ─────────────────────────────────────────────────
DATA_CONFIG="configs/tasks_3d_roundrobin.yaml"
TRAIN_BATCH_SIZE=2
TRAIN_IMAGE_SIZE=256
TEST_IMAGE_SIZE=256

# ── Model ────────────────────────────────────────────────
CONDITION_TYPE="deg-aware"
BACKBONE_TYPE="simple-conv"
TIMESTEP_VALUE=100
LORA_RANK_VAE=16

# ── Loss ─────────────────────────────────────────────────
LAMBDA_L2=2.0
LAMBDA_LPIPS=5.0
LAMBDA_GAN=2.0

# ── Generator optimizer ──────────────────────────────────
GEN_LR=1e-4
GEN_LR_SCHEDULER="cosine"
GEN_LR_WARMUP=500

# ── Discriminator (S3Diff-style) ─────────────────────────
DISC_LR=1e-4
DISC_LR_SCHEDULER="cosine"
DISC_LR_WARMUP=0

# ── Training loop ────────────────────────────────────────
GRAD_ACCUM=16
MAX_STEPS=5000
SEED=42
MIXED_PRECISION="bf16"

# ── Logging / eval ───────────────────────────────────────
CHECKPOINT_STEPS=2000
EVAL_FREQ=500
SAVE_EVAL_IMAGES=15

echo "=== GAN: t=${TIMESTEP_VALUE} + ${BACKBONE_TYPE} + VAE LoRA=${LORA_RANK_VAE} + DINO-Disc + DeepSpeed + 2x4090 ==="
echo "  Output: ${OUT_DIR}"

accelerate launch --num_processes=2 src/train.py \
    --data_config                 "${DATA_CONFIG}" \
    --output_dir                  "${OUT_DIR}" \
    --sd_path                     "${SD_PATH}" \
    --condition_type              "${CONDITION_TYPE}" \
    --backbone_type               "${BACKBONE_TYPE}" \
    --timestep_value             "${TIMESTEP_VALUE}" \
    --dino_type                   "${DINO_PATH}" \
    --degradation_classifier_path "${CLASSIFIER}" \
    --train_batch_size            "${TRAIN_BATCH_SIZE}" \
    --train_image_size            "${TRAIN_IMAGE_SIZE}" \
    --test_image_size             "${TEST_IMAGE_SIZE}" \
    --enable_lora \
    --lora_rank_vae               "${LORA_RANK_VAE}" \
    --round_robin \
    --lambda_l2                   "${LAMBDA_L2}" \
    --lambda_lpips                "${LAMBDA_LPIPS}" \
    --use_gan \
    --lambda_gan                  "${LAMBDA_GAN}" \
    --learning_rate               "${GEN_LR}" \
    --lr_scheduler                "${GEN_LR_SCHEDULER}" \
    --lr_warmup_steps             "${GEN_LR_WARMUP}" \
    --disc_learning_rate          "${DISC_LR}" \
    --disc_lr_scheduler           "${DISC_LR_SCHEDULER}" \
    --disc_lr_warmup_steps        "${DISC_LR_WARMUP}" \
    --adam_weight_decay           0.0 \
    --max_grad_norm               1.0 \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --mixed_precision             "${MIXED_PRECISION}" \
    --max_train_steps             "${MAX_STEPS}" \
    --seed                        "${SEED}" \
    --checkpointing_steps         "${CHECKPOINT_STEPS}" \
    --eval_freq                   "${EVAL_FREQ}" \
    --num_images_save_eval        "${SAVE_EVAL_IMAGES}" \
    > "${OUT_DIR}/train.log" 2>&1

echo "=== Done (exit=$?) ==="
