#!/usr/bin/env python3
"""Stage 1 — Train multi-degradation classifier (DINOv2 + Classifier head)."""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from degnet import DegNet_DINO
from utils.dataset import ClassificationDataset


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification: FL = -(1-p_t)^γ * log(p_t)"""
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            at = self.alpha[targets]
            focal = at * focal
        return focal.mean()


def compute_binary_metrics(preds, labels):
    preds = preds.long()
    labels = labels.long()

    per_class = []
    for c in range(labels.shape[1]):
        pred_c = preds[:, c]
        label_c = labels[:, c]
        tp = ((pred_c == 1) & (label_c == 1)).sum().item()
        fp = ((pred_c == 1) & (label_c == 0)).sum().item()
        fn = ((pred_c == 0) & (label_c == 1)).sum().item()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class.append({
            'accuracy': (pred_c == label_c).float().mean().item(),
            'precision': precision,
            'recall': recall,
            'f1': f1,
        })

    precision_vals = [m['precision'] for m in per_class]
    recall_vals = [m['recall'] for m in per_class]
    f1_vals = [m['f1'] for m in per_class]

    return {
        'accuracy': (preds == labels).float().mean().item(),
        'exact_match': (preds == labels).all(dim=1).float().mean().item(),
        'precision': sum(precision_vals) / max(len(precision_vals), 1),
        'recall': sum(recall_vals) / max(len(recall_vals), 1),
        'f1': sum(f1_vals) / max(len(f1_vals), 1),
        'per_class': per_class,
    }


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
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal Loss gamma; 0 = plain CrossEntropy")
    parser.add_argument("--round_robin", action="store_true",
                        help="Use repeat_ratio from task config to balance classes")
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
            loss = FocalLoss(gamma=args.focal_gamma)(
                logits.view(B * C, 2), labels.view(B * C))

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
                model.eval()
                gathered_preds = []
                gathered_labels = []
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_lq = val_batch['lq'].to(accelerator.device, dtype=weight_dtype)
                        val_labels = val_batch['label'].to(accelerator.device, dtype=torch.long)
                        val_logits = model(val_lq)
                        val_preds = val_logits.argmax(dim=-1)
                        gathered_preds.append(accelerator.gather(val_preds))
                        gathered_labels.append(accelerator.gather(val_labels))

                if accelerator.is_main_process and gathered_preds:
                    all_preds = torch.cat(gathered_preds, dim=0)
                    all_labels = torch.cat(gathered_labels, dim=0)
                    metrics = compute_binary_metrics(all_preds, all_labels)
                    per_class = [
                        f"c{c}=acc:{m['accuracy']:.3f}/p:{m['precision']:.3f}/r:{m['recall']:.3f}/f1:{m['f1']:.3f}"
                        for c, m in enumerate(metrics['per_class'])
                    ]
                    print(
                        f"\n  [Eval step {global_step}] acc={metrics['accuracy']:.4f}"
                        f" exact_match={metrics['exact_match']:.4f}"
                        f" precision={metrics['precision']:.4f}"
                        f" recall={metrics['recall']:.4f}"
                        f" f1={metrics['f1']:.4f}  {' '.join(per_class)}",
                        flush=True,
                    )
                    if metrics['f1'] > best_val_acc:
                        best_val_acc = metrics['f1']
                        unwrapped = accelerator.unwrap_model(model)
                        torch.save(unwrapped.state_dict(),
                                   os.path.join(args.output_dir, "best_model.pth"))
                        print(f"  → New best → saved best_model.pth")
                model.train()

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.end_training()
    print(f"Training complete. Best val acc: {best_val_acc:.4f}")
    print(f"Model saved to {args.output_dir}/best_model.pth")


if __name__ == "__main__":
    main()
