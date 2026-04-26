#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
NUM_GPUS=1
PRETRAINED_MODEL_NAME_OR_PATH=/home/yhmi/data/model/flux.2-klein
DATASETS_CONFIG=/home/yhmi/All_in_one/options/train/data.yaml
OUTPUT_DIR=/home/yhmi/data/output/flux2_convnext_ft_3
DEGRADATION_CLASSIFIER_PATH=/home/yhmi/data/model/best_model.pth
DINO_TYPE=/home/yhmi/data/model/dinov2-base

#--num_processes=${NUM_GPUS} \
#--multi_gpu \
#--num_machines=1 \
#--machine_rank=0 \
#--main_training_function=main \

# Training
accelerate launch \
    /home/yhmi/All_in_one/train.py \
    --pretrained_model_name_or_path "${PRETRAINED_MODEL_NAME_OR_PATH}" \
    --datasets_config "${DATASETS_CONFIG}" \
    --resolution 512 \
    --seed 42 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 10 \
    --max_train_steps 10000 \
    --save_checkpointing_steps 2000 \
    --val_monitor_steps 10 \
    --num_val_samples_per_dataset 3 \
    --learning_rate 2e-4 \
    --optimizer AdamW \
    --lr_scheduler cosine \
    --lr_warmup_steps 2000 \
    --guidance_scale 3.5 \
    --fixed_timestep 900 \
    --num_inference_steps 1 \
    --degradation_classifier_path "${DEGRADATION_CLASSIFIER_PATH}" \
    --dino_type "${DINO_TYPE}" \
    --num_deg_types 4 \
    --mod_lq_type convnext \
    --dataloader_num_workers 8 \
    --output_dir "${OUTPUT_DIR}" \
    --logging_dir "${OUTPUT_DIR}/logs" \
    --report_to swanlab \
    --mixed_precision "bf16" \
    --allow_tf32

    # --lr_num_cycles 1 \
    # --lr_power 1.0 \
    # --adam_beta1 0.9 \
    # --adam_beta2 0.999 \
    # --adam_weight_decay 0.01 \
    # --adam_epsilon 1e-8 \
    # --max_grad_norm 1.0 \
    # --checkpoints_total_limit 10 \
    # --resume_from_checkpoint latest \
    # --gradient_checkpointing \