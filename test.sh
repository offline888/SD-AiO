#!/bin/bash
export ACCELERATE_DDP_FIND_UNUSED_PARAMETERS=true

NUM_GPUS=4
PRETRAINED_MODEL_NAME_OR_PATH=/home/yhmi/data/model/flux.2-klein
DATASETS_CONFIG=/home/yhmi/All_in_one/options/train/data.yaml
OUTPUT_DIR=/home/yhmi/data/output/flux2_lora
DEGRADATION_CLASSIFIER_PATH=/home/yhmi/data/model/best_model.pth
DINO_TYPE=/home/yhmi/data/model/dinov2-base

#  Inference (YAML mode: batch-infer on all ValDataset images)
python /home/yhmi/All_in_one/test.py \
    --pretrained_model_name_or_path "${PRETRAINED_MODEL_NAME_OR_PATH}" \
    --modulation_weights "${OUTPUT_DIR}/checkpoint-2000/modulation_weights.pt" \
    --lq_image /home/yhmi/data/patches/08Haze/LQ_val/0347289_patch_02248_00200.jpg \
    --prompt "remove haze from this image" \
    --output_dir "${OUTPUT_DIR}/inference_results" \
    --device cuda \
    --dtype float32 \
    --guidance_scale 3.5 \
    --num_inference_steps 1 \
    --fixed_timestep 500 \
    --seed 42 \
    --resize_to 512 \
    --degradation_classifier_path "${DEGRADATION_CLASSIFIER_PATH}" \
    --dino_type "${DINO_TYPE}" \
    --num_deg_types 4 \
    --disable_cpu_offload \