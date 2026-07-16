#!/usr/bin/env python3
from json import encoder
import argparse, gc, os, warnings
warnings.filterwarnings("ignore")

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm.auto import tqdm

from degnet import DegFeatExtractor
from vae import PreRestoreEncoder
from utils.dataset import build_dataloaders

@torch.no_grad()
def evaluate(
    unwarp_encoder, 
    frozen_vae, 
    deg_extractor, 
    valid_loaders, 
    net_lpips,
    weight_dtype, 
    args, 
    global_step, 
    val_task_names, 
    device):

    unwarp_encoder.eval()
    task_id = {n: i for i, n in enumerate(sorted(val_task_names))}
    rows, vis_buf = [], {}
    scale = frozen_vae.config.scaling_factor
    vae_dtype = next(frozen_vae.parameters()).dtype

    for dataloader in valid_loaders.values():
        for batch in dataloader:
            tn = batch['task_name'][0]
            lq = batch['conditioning_pixel_values'].to(device, dtype=weight_dtype)
            gt = batch['output_pixel_values'].to(device, dtype=weight_dtype)

            f_deg = deg_extractor(lq)
            z_raw = unwarp_encoder(lq, f_deg)
            z_mean = frozen_vae.quant_conv(z_raw.to(dtype=vae_dtype))[:, :4]
            pred = frozen_vae.decode(z_mean / scale).sample.clamp(-1, 1)

            pf, gf = pred.float(), gt.float()
            lp = net_lpips(pf, gf).mean().item()
            pred_np = ((pf[0].permute(1, 2, 0).cpu().numpy() + 1) / 2).clip(0, 1).astype(np.float32)
            gt_np = ((gf[0].permute(1, 2, 0).cpu().numpy() + 1) / 2).clip(0, 1).astype(np.float32)
            lq_np = ((batch['conditioning_pixel_values'][0].permute(1, 2, 0).cpu().numpy() + 1) / 2).clip(0, 1).astype(np.float32)
            rows.append([task_id[tn], psnr(gt_np, pred_np, data_range=1),
                         ssim(gt_np, pred_np, data_range=1, channel_axis=-1), lp])
            vis_buf.setdefault(tn, []).append((pred_np, gt_np, lq_np))

    if not rows:
        return
    id2n = {i: n for n, i in task_id.items()}
    pt = {}
    for r in rows:
        pt.setdefault(id2n[int(r[0])], [[], [], []]); pt[id2n[int(r[0])]][0].append(r[1]); pt[id2n[int(r[0])]][1].append(r[2]); pt[id2n[int(r[0])]][2].append(r[3])
    lines = [f"{tn}: PSNR={np.mean(m[0]):.2f} SSIM={np.mean(m[1]):.4f} LPIPS={np.mean(m[2]):.4f}" for tn, m in sorted(pt.items())]
    line = f"[Eval {global_step}] n={sum(len(m[0]) for m in pt.values())} | {' | '.join(lines)}"
    print(line, flush=True)
    with open(os.path.join(args.output_dir, "eval_results.txt"), "a") as ef:
        ef.write(line + "\n\n")

    if vis_buf and args.num_images_save_eval > 0:
        num_per_task = max(1, args.num_images_save_eval // max(1, len(vis_buf)))
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)
        for task_name, items in sorted(vis_buf.items()):
            for idx, (pred_np, gt_np, lq_np) in enumerate(items[:num_per_task]):
                strip = (np.concatenate([lq_np, pred_np, gt_np], axis=1).clip(0, 1) * 255).astype(np.uint8)
                Image.fromarray(strip).save(
                    os.path.join(args.output_dir, "eval", f"step_{global_step}_{task_name}_{idx:02d}.png"))
    unwarp_encoder.train()
    gc.collect(); torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sd_path", required=True)
    p.add_argument("--data_config", default="./configs/tasks_3d.yaml")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--adaln_layers", type=str, nargs="+", default=["down2", "down3", "mid"])
    p.add_argument("--cond_dim", type=int, default=768,
                   help="F_Deg dimension (768 for DINOv2-base)")
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--lambda_l1", type=float, default=1.0)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--max_train_steps", type=int, default=50000)
    p.add_argument("--mixed_precision", type=str, default="bf16")
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--adam_weight_decay", type=float, default=0.01)
    p.add_argument("--adam_epsilon", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--lr_scheduler", type=str, default="cosine")
    p.add_argument("--lr_warmup_steps", type=int, default=500)
    p.add_argument("--checkpointing_steps", type=int, default=5000)
    p.add_argument("--eval_freq", type=int, default=500)
    p.add_argument("--num_images_save_eval", type=int, default=10)
    p.add_argument("--log_with", type=str, default=None)
    p.add_argument("--degradation_classifier_path", type=str, required=True)
    p.add_argument("--num_deg_types", type=int, default=3)
    p.add_argument("--dino_type", type=str, required=True)
    p.add_argument("--train_batch_size", type=int, default=None)
    p.add_argument("--train_image_size", type=int, default=None)
    p.add_argument("--test_image_size", type=int, default=None)
    args = p.parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.log_with,
    )
    is_main = accelerator.is_local_main_process
    
    if args.seed is not None:
        set_seed(args.seed)
    
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        import json
        with open(os.path.join(args.output_dir, "config.json"), "w") as cf:
            json.dump(vars(args), cf, indent=2, default=str)

    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.mixed_precision, torch.float32)

    frozen_vae = AutoencoderKL.from_pretrained(args.sd_path, subfolder="vae")
    frozen_vae.requires_grad_(False).eval()
    frozen_vae.to(accelerator.device)
    vae_dtype = next(frozen_vae.parameters()).dtype  # float32

    deg_extractor = DegFeatExtractor(
        inner_dim=args.cond_dim, num_deg_types=args.num_deg_types,
        weight_dtype=weight_dtype, args=args, device=accelerator.device,
    ).eval().requires_grad_(False)

    encoder = PreRestoreEncoder(
        encoder=frozen_vae.encoder,
        block_out_channels=frozen_vae.config.block_out_channels,
        cond_dim=args.cond_dim,
        adaln_layers=args.adaln_layers,
    )

    train_loader, valid_loaders = build_dataloaders(
        args.data_config, full_image_eval=False,
        train_batch_size=args.train_batch_size,
        train_image_size=args.train_image_size,
        test_image_size=args.test_image_size,
    )
    val_task_names = list(valid_loaders.keys())

    # ── Optimizer ──
    trainable_params = list(encoder.parameters())
    if is_main:
        total = sum(p.numel() for p in trainable_params)
        print(f"Trainable: {total/1e6:.1f}M  adaln_layers={args.adaln_layers}  lr={args.learning_rate}  grad_accum={args.gradient_accumulation_steps}", flush=True)
    
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), 
        weight_decay=args.adam_weight_decay, 
        eps=args.adam_epsilon)

    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
                                  num_warmup_steps=args.lr_warmup_steps,
                                  num_training_steps=args.max_train_steps)

    net_lpips = lpips.LPIPS(net="vgg").to(accelerator.device).requires_grad_(False)

    encoder, optimizer, lr_scheduler, train_loader = accelerator.prepare(
        encoder, optimizer, lr_scheduler, train_loader)
    # train mode: prepared model，save/eval mode: unwrapped model
    unwarp_encoder = accelerator.unwrap_model(encoder)

    global_step = 0
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location="cpu")
        unwarp_encoder.load_state_dict(ckpt["encoder"])
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        global_step = ckpt["global_step"]
        if is_main:
            print(f"Resumed from step {global_step}", flush=True)

    pbar = tqdm(range(args.max_train_steps), initial=global_step, desc="Steps", disable=not is_main)

    for _ in range(999999):
        for _ , batch in enumerate(train_loader):
            lq = batch['conditioning_pixel_values'].to(accelerator.device, dtype=weight_dtype)
            hq = batch['output_pixel_values'].to(accelerator.device, dtype=weight_dtype)

            with accelerator.accumulate(encoder):
                f_deg = deg_extractor(lq)

                # after quant_conv,[:4] -> mean , [4:] -> log(sigma^2)
                # Align latent mean!
                z_lq_raw = encoder(lq, f_deg)
                z_lq = frozen_vae.quant_conv(z_lq_raw.to(dtype=vae_dtype))[:, :4]

                with torch.no_grad():
                    z_hq_raw = frozen_vae.encoder(hq.to(dtype=vae_dtype))
                    z_hq = frozen_vae.quant_conv(z_hq_raw)[:, :4]

                loss = F.l1_loss(z_lq.float(), z_hq.float()) * args.lambda_l1
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                pbar.update(1)
                pbar.set_postfix_str(f"loss={loss.detach().item():.4f}")
                if is_main:
                    if global_step % args.eval_freq == 0 and valid_loaders:
                        evaluate(unwarp_encoder, 
                                frozen_vae, deg_extractor, valid_loaders,
                                 net_lpips, weight_dtype, args, global_step, val_task_names, accelerator.device)
                    if global_step % args.checkpointing_steps == 0:
                        ckpt_path = os.path.join(args.output_dir, f"checkpoint_{global_step}.pt")
                        torch.save({
                            "encoder": unwarp_encoder.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "global_step": global_step,
                        }, ckpt_path)
                        if is_main:
                            print(f"[Checkpoint] {ckpt_path}")
                    if global_step % 25 == 0:
                        gc.collect(); torch.cuda.empty_cache()
                if global_step >= args.max_train_steps:
                    break
        if global_step >= args.max_train_steps:
            break

    if is_main and valid_loaders:
        evaluate(unwarp_encoder, frozen_vae, deg_extractor, valid_loaders,
                 net_lpips, weight_dtype, args, global_step, val_task_names, accelerator.device)
    accelerator.wait_for_everyone()
    if is_main:
        print(f"Training complete. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
