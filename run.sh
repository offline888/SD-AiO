#!/bin/bash
export ACCELERATE_DDP_FIND_UNUSED_PARAMETERS=true

NUM_GPUS=2
PRETRAINED_MODEL_NAME_OR_PATH=/home/yhmi/data/model/flux.2-klein
DATASETS_CONFIG=/home/yhmi/All_in_one/options/train/data.yaml
OUTPUT_DIR=/home/yhmi/data/output/flux2_lora
DEGRADATION_CLASSIFIER_PATH=/home/yhmi/data/model/best_model.pth
DINO_TYPE=/home/yhmi/data/model/dinov2-base

#  Inference
python /home/yhmi/All_in_one/test_flux2_ir.py \
    --pretrained_model_name_or_path "${PRETRAINED_MODEL_NAME_OR_PATH}" \
    --modulation_weights "${OUTPUT_DIR}/checkpoint-50000/modulation_weights.pt" \
    # --lq_image /path/to/lq.png \
    # --prompt "A photo of high quality" \
    # --output_dir "${OUTPUT_DIR}/inference_results" \
    # --device cuda \
    # --dtype float32 \
    # --guidance_scale 3.5 \
    # --num_inference_steps 1 \
    # --fixed_timestep 300 \
    # --seed 42 \
    # --disable_cpu_offload \
    # --resize_to 512 \
    # --degradation_classifier_path "${DEGRADATION_CLASSIFIER_PATH}" \
    # --dino_type "${DINO_TYPE}" \
    # --num_deg_types 4 \

# Training
accelerate launch \
    --num_processes=${NUM_GPUS} \
    --multi_gpu \
    --num_machines=1 \
    --machine_rank=0 \
    --main_training_function=main \
    /home/yhmi/All_in_one/train_flux2_ir.py \
    --pretrained_model_name_or_path "${PRETRAINED_MODEL_NAME_OR_PATH}" \
    --datasets_config "${DATASETS_CONFIG}" \
    # --resolution 512 \
    # --seed 42 \
    # --train_batch_size 2 \
    # --gradient_accumulation_steps 8 \
    # --num_train_epochs 1 \
    # --max_train_steps 50000 \
    # --checkpointing_steps 2000 \
    # --checkpoints_total_limit 10 \
    # --resume_from_checkpoint latest \
    # --gradient_checkpointing \
    # --learning_rate 1e-4 \
    # --optimizer AdamW \
    # --adam_beta1 0.9 \
    # --adam_beta2 0.999 \
    # --adam_weight_decay 0.01 \
    # --adam_epsilon 1e-8 \
    # --max_grad_norm 1.0 \
    # --lr_scheduler cosine \
    # --lr_warmup_steps 500 \
    # --lr_num_cycles 1 \
    # --lr_power 1.0 \
    # --guidance_scale 3.5 \
    # --fixed_timestep 300 \
    # --num_inference_steps 1 \
    # --degradation_classifier_path "${DEGRADATION_CLASSIFIER_PATH}" \
    # --dino_type "${DINO_TYPE}" \
    # --num_deg_types 4 \
    # --dataloader_num_workers 4 \
    --output_dir "${OUTPUT_DIR}" \
    --logging_dir "${OUTPUT_DIR}/logs" \
    # --report_to swanlab \
    --mixed_precision "bf16" \
    # --allow_tf32 \
    # --local_rank -1 \