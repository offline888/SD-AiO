#!/usr/bin/env python3
"""Precise debug: reproduce exactly what train.py counts."""
import sys
sys.path.insert(0, "src")

import torch
import torch.nn as nn
from accelerate import Accelerator

# Simulate train.py's exact argument parsing (default = no LoRA)
class Args:
    task_config = "./configs/tasks.yaml"
    output_dir = "./output_debug"
    sd_path = "/root/shared-nvme/model/sd2-1"
    pretrained_path = None
    enable_lora = False
    lora_rank_unet = 0
    lora_rank_vae = 0
    num_inference_steps = 1
    condition_type = "deg_cross_attn"
    condition_embed_dim = 256
    timestep_strategy = "fixed"
    timestep_value = 100
    timestep_list = None
    timestep_range = [0, 999]
    lambda_l1 = 2.0
    lambda_lpips = 5.0
    use_gan = False
    image_size = 512
    train_batch_size = 1
    gradient_accumulation_steps = 4
    learning_rate = 2e-5
    max_train_steps = 50000
    num_training_epochs = 1000
    mixed_precision = "bf16"
    enable_xformers = False
    num_workers = 0
    seed = 42
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_weight_decay = 0.01
    adam_epsilon = 1e-8
    max_grad_norm = 1.0
    lr_scheduler = "constant"
    lr_warmup_steps = 500
    set_grads_to_none = False
    checkpointing_steps = 5000
    eval_freq = 100
    num_samples_eval = 0
    save_val = True
    num_deg_types = 5
    dino_type = "/root/shared-nvme/model/dinov2"
    degradation_classifier_path = None
    freeze_decoder = True
    log_with = "swanlab"  # what run_train.sh uses

args = Args()

print("=" * 70)
print(f"Args: enable_lora={args.enable_lora}, lora_rank_unet={args.lora_rank_unet}, lora_rank_vae={args.lora_rank_vae}")
print("=" * 70)

# ─── Step 1: Create model ─────────────────────────────────────────────────────
from model import SDSingleStepRestoration
model = SDSingleStepRestoration(
    sd_path=args.sd_path,
    lora_rank_unet=args.lora_rank_unet if args.enable_lora else 0,
    lora_rank_vae=args.lora_rank_vae if args.enable_lora else 0,
    num_inference_steps=args.num_inference_steps,
    enable_xformers=args.enable_xformers,
)
print(f"Model created. Device: {next(model.parameters()).device}")

# ─── Step 2: set_train() ──────────────────────────────────────────────────────
model.set_train()

# ─── Step 3: trainable_parameters() ──────────────────────────────────────────
trainable_params = list(model.trainable_parameters())
trainable_count = sum(p.numel() for p in trainable_params)
print(f"\n--- model.trainable_parameters() ---")
print(f"  Count: {trainable_count:,} ({trainable_count/1e6:.3f}M)")
print(f"  Devices: {set(p.device.type for p in trainable_params)}")

# Show breakdown
for n, p in model.named_parameters():
    if "lora" in n or "conv_in" in n:
        print(f"    {n:60s} {str(list(p.shape)):20s} numel={p.numel():>10,} dev={p.device}")

# ─── Step 4: cond_module ──────────────────────────────────────────────────────
from cond_module import build_condition_module
cond_module = build_condition_module(
    args.condition_type, args.condition_embed_dim,
    torch.device("cpu"), model.unet, training=True, args=args
)
cond_params = list(cond_module.parameters())
cond_count = sum(p.numel() for p in cond_params)
print(f"\n--- cond_module.parameters() ---")
print(f"  Count: {cond_count:,} ({cond_count/1e6:.3f}M)")
for n, p in cond_module.named_parameters():
    print(f"    {n:60s} {str(list(p.shape)):20s} numel={p.numel():>10,} req_grad={p.requires_grad}")

# ─── Step 5: Combined ─────────────────────────────────────────────────────────
combined = trainable_params + cond_params
combined_count = sum(p.numel() for p in combined)
print(f"\n--- COMBINED trainable_params ---")
print(f"  Count: {combined_count:,} ({combined_count/1e6:.3f}M)")

# ─── Step 6: Device analysis ──────────────────────────────────────────────────
on_gpu = sum(p.numel() for p in combined if p.is_cuda)
on_cpu = sum(p.numel() for p in combined if not p.is_cuda)
print(f"\n  On GPU: {on_gpu:,} ({on_gpu/1e6:.3f}M)")
print(f"  On CPU: {on_cpu:,} ({on_cpu/1e6:.3f}M)")

# ─── Step 7: Check DegFeatExtractor ──────────────────────────────────────────
if hasattr(cond_module, '_deg_extractor') and cond_module._deg_extractor is not None:
    deg = cond_module._deg_extractor
    deg_total = sum(p.numel() for p in deg.parameters())
    deg_trainable = sum(p.numel() for p in deg.parameters() if p.requires_grad)
    print(f"\n--- DegFeatExtractor (from cond_module._deg_extractor) ---")
    print(f"  Total: {deg_total:,}, Trainable: {deg_trainable:,}")
else:
    print(f"\n--- DegFeatExtractor: NOT BUILT yet (built lazily on first forward) ---")

# ─── Step 8: What does model.parameters() count? ───────────────────────────────
total_model_params = sum(p.numel() for p in model.parameters())
print(f"\n--- All model.parameters() ---")
print(f"  Total: {total_model_params:,} ({total_model_params/1e9:.2f}B)")

# ─── Step 9: Are there any unexpected trainable params? ───────────────────────
all_params = {n: p for n, p in model.named_parameters()}
trainable_nonasmodel = [n for n, p in model.named_parameters()
                        if p.requires_grad and "lora" not in n and "conv_in" not in n]
print(f"\n--- Trainable params NOT in ['lora', 'conv_in'] filter ---")
if trainable_nonasmodel:
    for n in trainable_nonasmodel:
        print(f"  {n}: {all_params[n].numel():,}")
else:
    print("  (none)")

print("\n" + "=" * 70)
print("SUMMARY:")
print(f"  model.trainable_parameters()  = {trainable_count:,} ({trainable_count/1e6:.3f}M)")
print(f"  cond_module.parameters()     = {cond_count:,} ({cond_count/1e6:.3f}M)")
print(f"  COMBINED                    = {combined_count:,} ({combined_count/1e6:.3f}M)")
print("=" * 70)
