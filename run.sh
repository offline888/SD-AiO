#!/bin/bash

PRETRAINED_MODEL_NAME_OR_PATH=/home/yhmi/data/model/flux.2-klein
DATASETS_CONFIG=/home/yhmi/All_in_one/options/train/data.yaml
NUM_GPUS=2
OUTPUT_DIR=/home/yhmi/data/output/flux2_lora
DEGRADATION_CLASSIFIER_PATH=/home/yhmi/data/model/best_model.pth
DINO_TYPE=/home/yhmi/data/model/dinov2-base
export ACCELERATE_DDP_FIND_UNUSED_PARAMETERS=true

accelerate launch \
    --num_processes=${NUM_GPUS} \
    /home/yhmi/All_in_one/diffusers_flux2.py\
    --pretrained_model_name_or_path ${PRETRAINED_MODEL_NAME_OR_PATH} \
    --datasets_config ${DATASETS_CONFIG} \
    --resolution 512 \
    --train_batch_size 2 \
    --gradient_accumulation_steps 8\
    --num_train_epochs 1 \
    --max_train_steps 50000 \
    --checkpointing_steps 2000 \
    --learning_rate 1e-4 \
    --guidance_scale 3.5 \
    --fixed_timestep 300 \
    --validation_epochs 1 \
    --lr_scheduler "cosine" \
    --output_dir ${OUTPUT_DIR} \
    --logging_dir "${OUTPUT_DIR}/logs" \
    --report_to "swanlab" \
    --mixed_precision "bf16" \
    --allow_tf32 \
    --seed 42 \
    --dataloader_num_workers 4 \
    --optimizer "AdamW" \
    --adam_weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --degradation_classifier_path ${DEGRADATION_CLASSIFIER_PATH} \
    --dino_type ${DINO_TYPE}

