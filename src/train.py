import argparse, gc, os, warnings
import lpips, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
warnings.filterwarnings("ignore")
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm.auto import tqdm

from cond_module import build_condition_module
from model import SDSingleStepRestoration
from utils.dataset import build_dataloaders
from utils.text_cache import TextEmbeddingContainer


class TrainBundle(nn.Module):
    def __init__(self, model, cond_module):
        super().__init__()
        self.model = model
        self.cond_module = cond_module
    def forward(self, lq_image, text_embed, timestep=None, cond_module=None):
        cm = cond_module if cond_module is not None else self.cond_module
        return self.model(lq_image, text_embed, timestep=timestep, cond_module=cm)


def sample_timestep(cfg):
    if cfg.get("strategy", "fixed") == "fixed":
        return cfg.get("value", 150)
    low, high = cfg.get("range", [50, 150])
    return torch.randint(low, high + 1, (1,)).item()


@torch.no_grad()
def evaluate(raw_model, raw_cond_module, valid_loaders, net_lpips,
             text_cache, weight_dtype, args, global_step, val_task_names, device):
    raw_model.eval()
    if raw_cond_module is not None:
        raw_cond_module.eval()
    task_id = {n: i for i, n in enumerate(sorted(val_task_names))}
    rows, vis_buf = [], {}
    for dl in valid_loaders.values():
        for batch in dl:
            tn = batch['task_name'][0]
            lq = batch['conditioning_pixel_values'].to(device, dtype=weight_dtype)
            gt = batch['output_pixel_values'].to(device, dtype=weight_dtype)
            pred = raw_model(lq, text_cache.get_batch(batch['task_name'], device),
                             timestep=args.timestep_value, cond_module=raw_cond_module)
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
    # Save visualization images
    if vis_buf and args.num_images_save_eval > 0:
        num_per_task = max(1, args.num_images_save_eval // max(1, len(vis_buf)))
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)
        for task_name, items in sorted(vis_buf.items()):
            for idx, (pred_np, gt_np, lq_np) in enumerate(items[:num_per_task]):
                strip = (np.concatenate([lq_np, pred_np, gt_np], axis=1).clip(0, 1) * 255).astype(np.uint8)
                Image.fromarray(strip).save(
                    os.path.join(args.output_dir, "eval", f"step_{global_step}_{task_name}_{idx:02d}.png"))
    raw_model.train()
    if raw_cond_module is not None:
        raw_cond_module.train()
    gc.collect(); torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sd_path", required=True)
    p.add_argument("--data_config", default="./configs/tasks.yaml")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--enable_lora", action="store_true")
    p.add_argument("--lora_rank_unet", type=int, default=0)
    p.add_argument("--lora_rank_vae", type=int, default=0)
    p.add_argument("--condition_type", type=str, default="deg-aware")
    p.add_argument("--condition_embed_dim", type=int, default=256)
    p.add_argument("--backbone_type", type=str, default="resnet18", choices=["simple-conv", "resnet18", "convnext_tiny"])
    p.add_argument("--num_inference_steps", type=int, default=1)
    p.add_argument("--timestep_strategy", type=str, default="fixed")
    p.add_argument("--timestep_value", type=int, default=150)
    p.add_argument("--timestep_range", type=int, nargs=2, default=[50, 150])
    p.add_argument("--lambda_l2", type=float, default=1.0)
    p.add_argument("--lambda_lpips", type=float, default=0.5)
    p.add_argument("--lpips_model", type=str, default="vgg", choices=["vgg", "dino"])
    p.add_argument("--lpips_layers", type=str, default="all",
                   help="DINO LPIPS layers, '1-4' for shallow, 'all' for default")
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--max_train_steps", type=int, default=50000)
    p.add_argument("--mixed_precision", type=str, default="bf16")
    p.add_argument("--enable_xformers", action="store_true")
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
    p.add_argument("--degradation_classifier_path", type=str, default=None)
    p.add_argument("--num_deg_types", type=int, default=3)
    p.add_argument("--dino_type", type=str, default=None)
    p.add_argument("--dino_lpips_ckpt", type=str, default=None,
                   help="Path to classifier checkpoint for fine-tuned DINO LPIPS backbone")
    p.add_argument("--round_robin", action="store_true")
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
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)
        # Save full config for reproducibility
        import json
        with open(os.path.join(args.output_dir, "config.json"), "w") as cf:
            json.dump(vars(args), cf, indent=2, default=str)

    # Model first (text_encoder is freed after caching)
    model = SDSingleStepRestoration(
        sd_path=args.sd_path,
        lora_rank_unet=args.lora_rank_unet if args.enable_lora else 0,
        lora_rank_vae=args.lora_rank_vae if args.enable_lora else 0,
        num_inference_steps=args.num_inference_steps,
        enable_xformers=args.enable_xformers,
    )
    model.set_train()

    # Text embeddings cache, then free encoder
    data_cfg = OmegaConf.load(args.data_config)
    text_cache = TextEmbeddingContainer(model.text_encoder, model.tokenizer, torch.device("cpu"))
    for task in OmegaConf.to_container(data_cfg.train, resolve=True) + OmegaConf.to_container(data_cfg.test, resolve=True):
        text_cache.add_embedding(task['name'], task['prompt'])
    model.free_text_encoder()
    torch.cuda.empty_cache()

    # Data
    train_loader, valid_loaders = build_dataloaders(args.data_config, full_image_eval=False,
                                                    round_robin=args.round_robin,
                                                    train_batch_size=args.train_batch_size,
                                                    train_image_size=args.train_image_size,
                                                    test_image_size=args.test_image_size)
    val_task_names = list(valid_loaders.keys())

    # Condition module (separate from model, injected into UNet)
    cond_module = build_condition_module(args.condition_type, args.condition_embed_dim,
        accelerator.device, model.unet, training=True,
        backbone_type=args.backbone_type, args=args)
    if hasattr(cond_module, 'build_deg_extractor'):
        cond_module.build_deg_extractor(accelerator.device)

    # Optimizer (both model + cond_module trainable params)
    trainable_params = list(model.trainable_parameters())
    if cond_module is not None:
        trainable_params += [p for p in cond_module.parameters() if p.requires_grad]
    if is_main:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in trainable_params)
        print(f"Total: {total/1e6:.1f}M  Trainable: {trainable/1e6:.1f}M  ({trainable/total*100:.1f}%)", flush=True)
        print(f"  backbone={args.backbone_type}  cond={args.condition_type}  t={args.timestep_value}  grad_accum={args.gradient_accumulation_steps}", flush=True)
        _lpips_info = f"lpips={args.lpips_model}"
        if args.lpips_model == "dino":
            _lpips_info += f" layers={args.lpips_layers}"
            if args.dino_lpips_ckpt:
                _lpips_info += f" ckpt={args.dino_lpips_ckpt}"
        print(f"  {_lpips_info}  λ_l2={args.lambda_l2}  λ_lpips={args.lambda_lpips}  bs={args.train_batch_size or 'yaml'}  lr={args.learning_rate}", flush=True)

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps, num_training_steps=args.max_train_steps)

    if args.lpips_model == "dino":
        from dino_perceptual import DINOPerceptual
        from degnet import export_dino_backbone
        _dino_name = args.dino_lpips_ckpt or "/root/shared-nvme/model/dinov2"
        if args.dino_lpips_ckpt:
            _dino_name = export_dino_backbone(args.dino_lpips_ckpt,
                os.path.join(args.output_dir, "ft_dinov2"), dino_type=args.dino_type)
        _layers = [int(x) for x in args.lpips_layers.split(",")] if args.lpips_layers != "all" else "all"
        net_lpips = DINOPerceptual(
            model_name=_dino_name, version="v2", target_size=512, layers=_layers
        ).to(accelerator.device).bfloat16().eval()
    else:
        net_lpips = lpips.LPIPS(net="vgg").to(accelerator.device)
        net_lpips.requires_grad_(False)

    # Prepare (S3Diff-style: all training in one prepare, eval loaders stay raw)
    train_bundle = TrainBundle(model, cond_module)
    train_bundle, optimizer, lr_scheduler, train_loader = accelerator.prepare(
        train_bundle, optimizer, lr_scheduler, train_loader)

    raw_bundle = accelerator.unwrap_model(train_bundle)
    raw_model = raw_bundle.model
    raw_cond_module = raw_bundle.cond_module

    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.mixed_precision, torch.float32)

    # Resume
    global_step = 0
    if args.resume_from:
        global_step = raw_model.load_training_state(args.resume_from, raw_cond_module, optimizer, lr_scheduler)
        if is_main and global_step > 0:
            print(f"Resumed from step {global_step}", flush=True)

    pbar = tqdm(range(args.max_train_steps), initial=global_step, desc="Steps", disable=not is_main)

    for epoch in range(999999):
        for step, batch in enumerate(train_loader):
            lq = batch['conditioning_pixel_values'].to(accelerator.device, dtype=weight_dtype)
            gt = batch['output_pixel_values'].to(accelerator.device, dtype=weight_dtype)
            timestep = sample_timestep({"strategy": args.timestep_strategy, "value": args.timestep_value, "range": args.timestep_range})
            text_embed = text_cache.get_batch(batch['task_name'], accelerator.device)

            with accelerator.accumulate(train_bundle):
                predicted = train_bundle(lq, text_embed, timestep=timestep)
                loss_l2 = F.mse_loss(predicted.float(), gt.float()) * args.lambda_l2
                loss_lpips = net_lpips(predicted.float(), gt.float()).mean() * args.lambda_lpips
                loss = loss_l2 + loss_lpips
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                pbar.update(1)
                pbar.set_postfix_str(f"l2={loss_l2.detach().item():.4f} lpips={loss_lpips.detach().item():.4f}")
                if is_main:
                    if global_step % args.eval_freq == 0 and valid_loaders:
                        evaluate(raw_model, raw_cond_module, valid_loaders,
                                 net_lpips, text_cache, weight_dtype, args, global_step, val_task_names, accelerator.device)
                    if global_step % args.checkpointing_steps == 0:
                        raw_model.save_training_state(args.output_dir, global_step, raw_cond_module, optimizer, lr_scheduler)
                    if global_step % 25 == 0:
                        gc.collect(); torch.cuda.empty_cache()
                if global_step >= args.max_train_steps:
                    break
        if global_step >= args.max_train_steps:
            break

    if is_main and valid_loaders:
        evaluate(raw_model, raw_cond_module, valid_loaders,
                 net_lpips, text_cache, weight_dtype, args, global_step, val_task_names, accelerator.device)
    accelerator.wait_for_everyone()
    if is_main:
        print(f"Training complete. Checkpoints: {args.output_dir}/checkpoints/")


if __name__ == "__main__":
    main()
