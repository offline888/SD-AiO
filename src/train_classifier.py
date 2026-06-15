#!/usr/bin/env python3
"""Stage 1 — Train multi-degradation classifier (DINOv2 + Classifier head)."""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from degnet import DegNet_DINO
from utils.cls_dataset import ClassificationDataset


def main():
    parser = argparse.ArgumentParser(description="Train Degradation Classifier")
    parser.add_argument("--task_config", default="./configs/tasks.yaml")
    parser.add_argument("--dino_type", required=True, help="Path to DINOv2 model")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_deg_types", type=int, default=3)
    parser.add_argument("--freeze_encoder", action="store_true", default=False,
                        help="Freeze DINOv2 backbone (default: trainable)")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Classifier trains on smaller crops for speed")
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate_backbone", type=float, default=1e-5)
    parser.add_argument("--learning_rate_head", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=20000)
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpointing_steps", type=int, default=2000)
    parser.add_argument("--eval_freq", type=int, default=500)
    parser.add_argument("--log_with", type=str, default=None)
    args = parser.parse_args()

    task_cfg = OmegaConf.load(args.task_config)
    train_tasks = OmegaConf.to_container(task_cfg.train, resolve=True)
    val_tasks = OmegaConf.to_container(task_cfg.test, resolve=True)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.log_with,
    )
    set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    print(f"Building DegNet_DINO with {args.num_deg_types} degradation types")
    print(f"  DINO path: {args.dino_type}")
    print(f"  Freeze encoder: {args.freeze_encoder}")

    model = DegNet_DINO(
        dino_type=args.dino_type,
        num_types=args.num_deg_types,
        freeze_encoder=args.freeze_encoder,
    )

    train_dataset = ClassificationDataset(
        train_tasks, num_deg_types=args.num_deg_types,
        image_size=args.image_size, training=True,
    )
    val_dataset = ClassificationDataset(
        val_tasks, num_deg_types=args.num_deg_types,
        image_size=args.image_size, training=False,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.train_batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    backbone_params = list(model.encoder.parameters()) if not args.freeze_encoder else []
    head_params = list(model.decoder.parameters())
    backbone_params_to_optimize = [p for p in backbone_params if p.requires_grad]
    head_params_to_optimize = [p for p in head_params if p.requires_grad]

    optimizer = torch.optim.AdamW([
        {'params': head_params_to_optimize, 'lr': args.learning_rate_head},
        {'params': backbone_params_to_optimize, 'lr': args.learning_rate_backbone},
    ])

    def count(m): return sum(p.numel() for p in m if p.requires_grad)
    print(f"  Trainable: backbone={count(backbone_params_to_optimize):,}  head={count(head_params_to_optimize):,}")

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader,
    )

    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(
        args.mixed_precision, torch.float32)
    model.to(accelerator.device, dtype=weight_dtype)

    progress_bar = tqdm(range(args.max_train_steps), desc="Steps",
                        disable=not accelerator.is_local_main_process)
    global_step = 0
    best_val_acc = 0.0

    for epoch in range(1000):
        for batch in train_loader:
            model.train()
            lq = batch['lq'].to(accelerator.device, dtype=weight_dtype)
            labels = batch['label'].to(accelerator.device, dtype=torch.long)

            logits = model(lq)  # [B, C, 2]
            B, C, _ = logits.shape
            # Reshape to [B*C, 2] for per-degradation 2-way classification
            loss = F.cross_entropy(logits.view(B * C, 2), labels.view(B * C))

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1

            if accelerator.is_main_process:
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)  # [B, C]
                    acc = (preds == labels).float().mean().item()
                progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.3f}")

                if global_step % args.checkpointing_steps == 0:
                    unwrapped = accelerator.unwrap_model(model)
                    ckpt_path = os.path.join(args.output_dir, f"classifier_step_{global_step}.pth")
                    torch.save(unwrapped.state_dict(), ckpt_path)

                if global_step % args.eval_freq == 0 and len(val_loader) > 0:
                    val_accs = []
                    model.eval()
                    with torch.no_grad():
                        for val_batch in val_loader:
                            val_lq = val_batch['lq'].to(accelerator.device, dtype=weight_dtype)
                            val_labels = val_batch['label'].to(accelerator.device, dtype=torch.long)
                            val_logits = model(val_lq)
                            val_preds = val_logits.argmax(dim=-1)
                            val_accs.append((val_preds == val_labels).float().mean().item())
                    val_acc = np.mean(val_accs)
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        unwrapped = accelerator.unwrap_model(model)
                        torch.save(unwrapped.state_dict(),
                                   os.path.join(args.output_dir, "best_model.pth"))
                        print(f"\n  [Step {global_step}] New best val_acc={val_acc:.4f} → saved best_model.pth")

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.end_training()
    print(f"Training complete. Best val acc: {best_val_acc:.4f}")
    print(f"Model saved to {args.output_dir}/best_model.pth")


if __name__ == "__main__":
    main()
