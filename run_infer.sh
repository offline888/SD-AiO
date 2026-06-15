#!/bin/bash
# SD-AiO Inference Launcher
# Usage: bash run_infer.sh

# ═══════════════════════════════════════════════════════
#  Paths
# ═══════════════════════════════════════════════════════
SD_PATH="/path/to/stable-diffusion-2-1-base"
DINO_PATH="/path/to/dinov2-vitl14"
DEG_CLASSIFIER_PATH="/path/to/deg_classifier.pth"

MODEL_PATH="./output/checkpoints/step_50000.pkl"
COND_MODULE_PATH="./output/checkpoints/cond_module_50000.pth"
TASK_CONFIG="./configs/tasks.yaml"

INPUT="/path/to/test/images/"
OUTPUT_DIR="./results"

# ═══════════════════════════════════════════════════════
#  Inference config (must match training)
# ═══════════════════════════════════════════════════════
TASK="derain"                       # derain | dehaze | lowlight
CONDITION_TYPE="deg_cross_attn"     # must match training config
TIMESTEP=999                        # must match training config
ALIGN_METHOD="wavelet"              # wavelet | adain | none

# ═══════════════════════════════════════════════════════

python src/infer.py \
    --model_path ${MODEL_PATH} \
    --cond_module_path ${COND_MODULE_PATH} \
    --task_config ${TASK_CONFIG} \
    --sd_path ${SD_PATH} \
    --dino_type ${DINO_PATH} \
    --degradation_classifier_path ${DEG_CLASSIFIER_PATH} \
    --num_deg_types 3 \
    --input ${INPUT} \
    --output_dir ${OUTPUT_DIR} \
    --task ${TASK} \
    --condition_type ${CONDITION_TYPE} \
    --timestep ${TIMESTEP} \
    --align_method ${ALIGN_METHOD}