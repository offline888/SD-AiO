#!/bin/bash
# Stage 1 — Degradation Classifier Training Launcher
set -e

export CUDA_VISIBLE_DEVICES=2,3

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

ROOT_DIR="/root/shared-nvme/SD-AiO"
DINO_PATH="/root/shared-nvme/model/dinov2"
TASK_CONFIG="${ROOT_DIR}/configs/tasks.yaml"
OUTPUT_DIR="${ROOT_DIR}/output/classifier"

NUM_GPUS=2
NUM_DEG_TYPES=5
IMAGE_SIZE=256
BATCH_SIZE=32  # per GPU
MAX_STEPS=20000
CHECKPOINT_STEPS=2000
EVAL_FREQ=500
LR_BACKBONE=1e-5
LR_HEAD=1e-4

accelerate launch --mixed_precision=bf16 --num_processes=${NUM_GPUS} \
    ${ROOT_DIR}/src/train_classifier.py \
    --task_config ${TASK_CONFIG} \
    --dino_type ${DINO_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --num_deg_types ${NUM_DEG_TYPES} \
    --image_size ${IMAGE_SIZE} \
    --train_batch_size ${BATCH_SIZE} \
    --learning_rate_backbone ${LR_BACKBONE} \
    --learning_rate_head ${LR_HEAD} \
    --max_train_steps ${MAX_STEPS} \
    --checkpointing_steps ${CHECKPOINT_STEPS} \
    --eval_freq ${EVAL_FREQ} \
    --seed 42
