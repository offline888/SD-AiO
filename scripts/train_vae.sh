#!/bin/bash
# Stage 0: Train PreRestoreEncoder — LQ→latent alignment with AdaIN + F_Deg
accelerate launch src/train_vae.py \
  --sd_path /root/shared-nvme/model/sd2-1 \
  --data_config configs/tasks_3d.yaml \
  --output_dir ./output/vae_pretrain \
  --degradation_classifier_path /root/shared-nvme/SD-AiO/checkpoints/classifier/best_model.pth \
  --dino_type /root/shared-nvme/model/dinov2 \
  --adaln_layers down2 down3 mid \
  --learning_rate 1e-4 \
  --max_train_steps 20000 \
  --train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --mixed_precision bf16 \
  --eval_freq 500 \
  --checkpointing_steps 2000
