import argparse
import gc
import os

import diffusers
import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from torchvision import transforms
from tqdm.auto import tqdm

from utils.image_utils import rgb2ycbcr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

try:
    import swanlab
except ImportError:
    swanlab = None

from cond_module import build_condition_module
from model import SDSingleStepRestoration
from utils.dataset import PairedRestorationDataset
from utils.text_cache import TextEmbeddingCache


class TrainBundle(nn.Module):
    def __init__(self, model, cond_module):
        super().__init__()
        self.model = model
        self.cond_module = cond_module


def sample_timestep(cfg):
    strategy = cfg.get("strategy", None)
    if strategy is None:
        return None
    if strategy == "fixed":
        return cfg.get("value", 999)
    if strategy == "random":
        lo, hi = cfg.get("range", [0, 999])
        return torch.randint(lo, hi + 1, (1,)).item()
    if strategy == "list":
        choices = cfg.get("list", [999])
        return int(np.random.choice(choices))
    raise ValueError(f"Unknown timestep strategy: {strategy}")


def _optimize_step(accelerator, loss, params, optimizer, lr_scheduler, max_grad_norm, set_grads_to_none):
    accelerator.backward(loss)
    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(params, max_grad_norm)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad(set_to_none=set_grads_to_none)


def main():
    parser = argparse.ArgumentParser(description="SD-AiO Training")
    parser.add_argument("--task_config", default="./configs/tasks.yaml")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sd_path", required=True)
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="Path to model checkpoint (.pkl) for weight init or inference")
    parser.add_argument("--resume_cond_module_path", type=str, default=None,
                        help="Path to cond_module checkpoint (.pth) to resume training")
    parser.add_argument("--enable_lora", action="store_true", default=False,
                        help="Enable LoRA fine-tuning (requires --lora_rank_unet / --lora_rank_vae to also be set)")
    parser.add_argument("--lora_rank_unet", type=int, default=0)
    parser.add_argument("--lora_rank_vae", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=1)
    parser.add_argument("--condition_type", type=str, default="deg_cross_attn")
    parser.add_argument("--condition_embed_dim", type=int, default=256)
    parser.add_argument("--timestep_strategy", type=str, default="fixed")
    parser.add_argument("--timestep_value", type=int, default=150)
    parser.add_argument("--timestep_list", type=int, nargs="*", default=None)
    parser.add_argument("--timestep_range", type=int, nargs=2, default=[0, 999])
    parser.add_argument("--lambda_l1", type=float, default=2.0)
    parser.add_argument("--lambda_lpips", type=float, default=5.0)
    parser.add_argument("--use_gan", action="store_true")
    parser.add_argument("--lambda_gan", type=float, default=0.5)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_train_steps", type=int, default=50000)
    parser.add_argument("--num_training_epochs", type=int, default=1000)
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--enable_xformers", action="store_true")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--set_grads_to_none", action="store_true")
    parser.add_argument("--checkpointing_steps", type=int, default=5000)
    parser.add_argument("--eval_freq", type=int, default=500)
    parser.add_argument("--num_samples_eval", type=int, default=100)
    parser.add_argument("--no_save_val", action="store_false", dest="save_val")
    parser.add_argument("--num_images_save_eval", type=int, default=20)
    parser.add_argument("--num_deg_types", type=int, default=3)
    parser.add_argument("--dino_type", type=str, default=None)
    parser.add_argument("--degradation_classifier_path", type=str, default=None)
    parser.add_argument("--freeze_decoder", action="store_true", default=True)
    parser.add_argument("--log_with", type=str, default=None)

    args = parser.parse_args()

    task_cfg = OmegaConf.load(args.task_config)
    train_tasks = OmegaConf.to_container(task_cfg.train, resolve=True)
    val_tasks = OmegaConf.to_container(task_cfg.test, resolve=True)

    timestep_cfg = {
        "strategy": args.timestep_strategy,
        "value": args.timestep_value,
        "list": args.timestep_list,
        "range": args.timestep_range,
    }

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.log_with,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    if args.log_with == "swanlab" and swanlab is not None and accelerator.is_main_process:
        swanlab.init(project="sd-aio", config=vars(args))

    model = SDSingleStepRestoration(
        sd_path=args.sd_path,
        lora_rank_unet=args.lora_rank_unet if args.enable_lora else 0,
        lora_rank_vae=args.lora_rank_vae if args.enable_lora else 0,
        num_inference_steps=args.num_inference_steps,
        enable_xformers=args.enable_xformers,
    )
    if args.pretrained_path:
        model.load_checkpoint(args.pretrained_path)
    model.set_train()

    if args.enable_xformers:
        if is_xformers_available():
            model.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers not available. Install with: pip install xformers")

    text_cache = TextEmbeddingCache(model.text_encoder, model.tokenizer, torch.device("cpu"))
    for task in train_tasks:
        text_cache.add_task(task['name'], task['prompt'])
    model.free_text_encoder()
    torch.cuda.empty_cache()

    train_dataset = PairedRestorationDataset(train_tasks, image_size=args.image_size, training=True)
    val_dataset = PairedRestorationDataset(val_tasks, image_size=args.image_size, training=False)
    train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.train_batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True)  
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0)

    cond_module = build_condition_module(
        args.condition_type, args.condition_embed_dim,
        accelerator.device, model.unet, training=True,
        args=args)
    if args.resume_cond_module_path:
        print(f"[resume] Loading cond_module from {args.resume_cond_module_path}")
        cond_module.load_state_dict(
            torch.load(args.resume_cond_module_path, map_location="cpu"), strict=False)

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    trainable_params = list(model.trainable_parameters())
    if cond_module is not None:
        trainable_params += [p for p in cond_module.parameters() if p.requires_grad]

    # ── Parameter statistics ──────────────────────────────────────────────────
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_count = sum(p.numel() for p in trainable_params)
    total_on_gpu = sum(p.numel() for p in model.parameters() if p.is_cuda)
    trainable_on_gpu = sum(p.numel() for p in trainable_params if p.is_cuda)

    def fmt(n):
        if n >= 1e9: return f"{n/1e9:.2f}B"
        if n >= 1e6: return f"{n/1e6:.2f}M"
        if n >= 1e3: return f"{n/1e3:.2f}K"
        return str(n)

    print(f"\n{'='*60}")
    print(f"  Total parameters      : {fmt(total_params)} ({total_params:,})")
    print(f"  Trainable parameters  : {fmt(trainable_params_count)} ({trainable_params_count:,})")
    print(f"  Total on GPU          : {fmt(total_on_gpu)} ({total_on_gpu:,})")
    print(f"  Trainable on GPU      : {fmt(trainable_on_gpu)} ({trainable_on_gpu:,})")
    print(f"{'='*60}\n")

    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes)

    train_bundle = TrainBundle(model, cond_module)
    train_bundle, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        train_bundle, optimizer, train_loader, lr_scheduler)
    model = train_bundle.model
    cond_module = train_bundle.cond_module

    net_disc, optimizer_disc, lr_scheduler_disc = None, None, None
    if args.use_gan:
        import vision_aided_loss
        net_disc = vision_aided_loss.Discriminator(
            cv_type='dino', output_type='conv_multi_level',
            loss_type="multilevel_sigmoid_s", device="cuda").cuda()
        net_disc.cv_ensemble.requires_grad_(False)
        net_disc.train()
        optimizer_disc = torch.optim.AdamW(
            net_disc.parameters(), lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
        lr_scheduler_disc = get_scheduler(
            args.lr_scheduler, optimizer=optimizer_disc,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes)
        net_disc, optimizer_disc, lr_scheduler_disc = accelerator.prepare(
            net_disc, optimizer_disc, lr_scheduler_disc)

    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(
        args.mixed_precision, torch.float32)
    model.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)
    if net_disc is not None:
        net_disc.to(accelerator.device, dtype=weight_dtype)

    if args.use_gan:
        for name, module in net_disc.named_modules():
            if "attn" in name:
                module.fused_attn = False

    # Unwrap before validation loop
    raw_model = accelerator.unwrap_model(model)
    raw_cond_module = accelerator.unwrap_model(cond_module)

    progress_bar = tqdm(
        range(args.max_train_steps), initial=0, desc="Steps",
        disable=not accelerator.is_local_main_process)
    global_step = 0

    for epoch in range(args.num_training_epochs):
        for step, batch in enumerate(train_loader):
            lq = batch['lq'].to(accelerator.device, dtype=weight_dtype)
            gt = batch['gt'].to(accelerator.device, dtype=weight_dtype)
            task_names = batch['task']

            accumulate_models = [model]
            if net_disc is not None:
                accumulate_models.append(net_disc)

            with accelerator.accumulate(*accumulate_models):
                t = sample_timestep(timestep_cfg)
                text_embed = text_cache.get_batch(task_names, accelerator.device)

                pred = model(lq, text_embed, timestep=t, cond_module=raw_cond_module)

                pred_f = pred.float()
                gt_f = gt.float()
                loss_l1 = F.l1_loss(pred_f, gt_f, reduction="mean") * args.lambda_l1
                loss_lpips = net_lpips(pred_f, gt_f).mean() * args.lambda_lpips
                loss = loss_l1 + loss_lpips

                _optimize_step(accelerator, loss, trainable_params, optimizer,
                               lr_scheduler, args.max_grad_norm, args.set_grads_to_none)

                if args.use_gan and net_disc is not None:
                    lossG = net_disc(pred, for_G=True).mean() * args.lambda_gan
                    _optimize_step(accelerator, lossG, trainable_params, optimizer,
                                   lr_scheduler, args.max_grad_norm, args.set_grads_to_none)

                    # Single combined discriminator backward
                    lossD_real = net_disc(gt.detach(), for_real=True).mean()
                    lossD_fake = net_disc(pred.detach(), for_real=False).mean()
                    lossD = (lossD_real + lossD_fake) * args.lambda_gan
                    _optimize_step(accelerator, lossD, net_disc.parameters(),
                                   optimizer_disc, lr_scheduler_disc,
                                   args.max_grad_norm, args.set_grads_to_none)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {
                        "loss_l1": loss_l1.detach().item(),
                        "loss_lpips": loss_lpips.detach().item(),
                        "timestep": t if t is not None else -1,
                    }
                    if args.use_gan and net_disc is not None:
                        logs["lossG"] = lossG.detach().item()
                        logs["lossD"] = lossD.detach().item()
                    progress_bar.set_postfix(**logs)

                    if global_step % args.checkpointing_steps == 0:
                        ckpt_path = os.path.join(args.output_dir, "checkpoints",
                                                 f"step_{global_step}.pkl")
                        raw_model.save_checkpoint(ckpt_path)
                        torch.save(raw_cond_module.state_dict(),
                                   os.path.join(args.output_dir, "checkpoints",
                                                f"cond_module_{global_step}.pth"))

                    if global_step % args.eval_freq == 0 and len(val_loader) > 0:
                        l_l1, l_lpips_vals = [], []
                        l_psnr, l_ssim_vals = [], []
                        val_count = 0
                        for val_batch in val_loader:
                            if val_count >= args.num_samples_eval:
                                break
                            val_lq = val_batch['lq'].to(accelerator.device, dtype=weight_dtype)
                            val_gt = val_batch['gt'].to(accelerator.device, dtype=weight_dtype)
                            val_task = val_batch['task']

                            with torch.no_grad():
                                val_text_embed = text_cache.get_batch(val_task, accelerator.device)
                                val_pred = raw_model(
                                    val_lq, val_text_embed, timestep=args.timestep_value,
                                    cond_module=raw_cond_module)
                                val_f = val_pred.float()
                                val_gt_f = val_gt.float()
                                l_l1.append(F.l1_loss(val_f, val_gt_f, reduction="mean").item())
                                l_lpips_vals.append(net_lpips(val_f, val_gt_f).mean().item())

                                val_np = (val_f[0].permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5
                                val_np = val_np.clip(0, 255).astype(np.uint8)
                                gt_np  = (val_gt_f[0].permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5
                                gt_np = gt_np.clip(0, 255).astype(np.uint8)
                                val_y = rgb2ycbcr(val_np, only_y=True)
                                gt_y  = rgb2ycbcr(gt_np, only_y=True)
                                l_psnr.append(psnr(gt_y, val_y, data_range=255))
                                l_ssim_vals.append(ssim(gt_y, val_y, data_range=255))

                            if args.save_val and val_count < args.num_images_save_eval:
                                combined = torch.cat([
                                    val_lq.cpu().detach().mul_(0.5).add_(0.5),
                                    val_pred.cpu().detach().mul_(0.5).add_(0.5),
                                    val_gt.cpu().detach().mul_(0.5).add_(0.5),
                                ], dim=3)
                                transforms.ToPILImage()(combined[0].clamp(0, 1)).save(
                                    os.path.join(args.output_dir, "eval",
                                                 f"step_{global_step}_{val_count}.png"))
                            val_count += 1

                        logs["val/l1"] = np.mean(l_l1) if l_l1 else 0
                        logs["val/lpips"] = np.mean(l_lpips_vals) if l_lpips_vals else 0
                        logs["val/psnr"] = np.mean(l_psnr) if l_psnr else 0
                        logs["val/ssim"] = np.mean(l_ssim_vals) if l_ssim_vals else 0
                        gc.collect()

                    if args.log_with == "swanlab" and swanlab is not None:
                        swanlab.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.end_training()
    print(f"Training complete. Checkpoints saved to {args.output_dir}/checkpoints/")


if __name__ == "__main__":
    main()
