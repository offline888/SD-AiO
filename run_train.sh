#!/bin/bash

# ═══════════════════════════════════════════════════════
#  SD-AiO Training Launcher
# ═══════════════════════════════════════════════════════

set -e

export CUDA_VISIBLE_DEVICES=2,3
# ── GPU info ───────────────────────────────────────────
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

# ── Paths ──
ROOT_DIR="/root/shared-nvme/SD-AiO"
SD_PATH="/root/shared-nvme/model/sd2-1"
DINO_PATH="/root/shared-nvme/model/dinov2"
DEG_CLASSIFIER_PATH="/root/shared-nvme/model/best_model.pth"
TASK_CONFIG="${ROOT_DIR}/configs/tasks_subset.yaml"
OUTPUT_DIR="${ROOT_DIR}/output/cross"

# ── Experiment config ──
CONDITION_TYPE="deg_cross_attn"      # must match registry key
TIMESTEP_STRATEGY="fixed"             # fixed | random | list
TIMESTEP_VALUE=250                   # used when strategy=fixed
FREEZE_DECODER="--freeze_decoder"   # remove flag to fine-tune decoder
ENABLE_XFORMERS=""  # remove if xformers not installed

# ── LoRA fine-tuning (disabled by default) ──────────────
# ENABLE_LORA=""           # uncomment to enable LoRA
# LORA_RANK_UNET=32        # used when LoRA enabled
# LORA_RANK_VAE=16         # used when LoRA enabled

# ── Training hyperparameters ──
NUM_GPUS=2
TRAIN_BATCH_SIZE=2
GRAD_ACCUM=8
MAX_STEPS=50000
CHECKPOINT_STEPS=5000
EVAL_FREQ=500
LEARNING_RATE=5e-5
IMAGE_SIZE=512
NUM_DEG_TYPES=3
LOGGER="swanlab"                 

accelerate launch --mixed_precision=bf16 --num_processes=${NUM_GPUS} \
    ${ROOT_DIR}/src/train.py \
    --task_config ${TASK_CONFIG} \
    --output_dir ${OUTPUT_DIR} \
    --sd_path ${SD_PATH} \
    --condition_type ${CONDITION_TYPE} \
    --timestep_strategy ${TIMESTEP_STRATEGY} \
    --timestep_value ${TIMESTEP_VALUE} \
    --dino_type ${DINO_PATH} \
    --degradation_classifier_path ${DEG_CLASSIFIER_PATH} \
    --num_deg_types ${NUM_DEG_TYPES} \
    ${FREEZE_DECODER} \
    ${ENABLE_XFORMERS} \
    ${ENABLE_LORA:+--enable_lora} \
    ${LORA_RANK_UNET:+--lora_rank_unet ${LORA_RANK_UNET}} \
    ${LORA_RANK_VAE:+--lora_rank_vae ${LORA_RANK_VAE}} \
    --train_batch_size ${TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --max_train_steps ${MAX_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --checkpointing_steps ${CHECKPOINT_STEPS} \
    --eval_freq ${EVAL_FREQ} \
    --image_size ${IMAGE_SIZE} \
    --seed 42 \
    --log_with ${LOGGER} \
    --num_samples_eval 100 \
    --num_images_save_eval 20
