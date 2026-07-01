import os
import math
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torchvision import transforms

from model import SDSingleStepRestoration
from utils.text_cache import TextEmbeddingContainer
from utils.image_utils import imread, img2tensor
from cond_module import build_condition_module

def main():
    parser = argparse.ArgumentParser(description="SD-AiO Inference")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--cond_module_path", type=str, default=None,
                        help="Path to cond_module checkpoint")
    parser.add_argument("--task_config", default="./configs/tasks.yaml")
    parser.add_argument("--sd_path", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--lora_rank_unet", type=int, default=32)
    parser.add_argument("--lora_rank_vae", type=int, default=16)
    parser.add_argument("--num_inference_steps", type=int, default=1)
    parser.add_argument("--condition_type", type=str, default="deg-aware")
    parser.add_argument("--condition_embed_dim", type=int, default=256)
    parser.add_argument("--timestep", type=int, default=999)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--enable_vae_tile", action="store_true")
    parser.add_argument("--vae_tile_size", type=int, default=512)
    parser.add_argument("--merge_lora", action="store_true")
    parser.add_argument("--dino_type", type=str, default=None)
    parser.add_argument("--degradation_classifier_path", type=str, default=None)
    parser.add_argument("--num_deg_types", type=int, default=3)
    args = parser.parse_args()
        
    task_cfg = OmegaConf.load(args.task_config)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = SDSingleStepRestoration(
        sd_path=args.sd_path,
        lora_rank_unet=args.lora_rank_unet,
        lora_rank_vae=args.lora_rank_vae,
        num_inference_steps=args.num_inference_steps,
        enable_vae_tile=args.enable_vae_tile,
        vae_tile_size=args.vae_tile_size,
        merge_lora=args.merge_lora,
    ).to(device)
    model.load_checkpoint(args.model_path)
    model.unet.eval()
    model.vae.eval()

    text_cache = TextEmbeddingContainer(model.text_encoder, model.tokenizer, device)
    all_tasks = OmegaConf.to_container(task_cfg.train, resolve=True) + \
                OmegaConf.to_container(task_cfg.test, resolve=True)
    for task in all_tasks:
        text_cache.add_embedding(task['name'], task['prompt'])
    if args.task not in text_cache:
        raise ValueError(f"Task '{args.task}' not in config. Available: {text_cache.task_names}")

    cond_module = build_condition_module(
        args.condition_type, args.condition_embed_dim,
        device, model.unet, training=False,
        args=args)
    if args.cond_module_path is not None:
        cond_module.load_state_dict(torch.load(args.cond_module_path, map_location=device), strict=False)

    input_path = Path(args.input)
    lr_paths = [input_path] if input_path.is_file() else sorted(
        p for p in input_path.glob("*")
        if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp'))

    for lr_path in lr_paths:
        print(f"Processing: {lr_path.name}")
        im_lq = imread(str(lr_path), chn='rgb', dtype='float32')
        im_tensor = img2tensor(im_lq).to(device)

        im_input = im_tensor * 2.0 - 1.0

        _, _, h, w = im_input.shape
        pad_h = (math.ceil(h / 64) * 64 - h)
        pad_w = (math.ceil(w / 64) * 64 - w)
        im_input = F.pad(im_input, (0, pad_w, 0, pad_h), mode='reflect')

        text_embed = text_cache[args.task].unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(im_input, text_embed, timestep=args.timestep,
                         cond_module=cond_module)
            pred = pred[:, :, :h, :w]

        pred = (pred * 0.5 + 0.5).clamp(0, 1)
        pred_pil = transforms.ToPILImage()(pred[0].cpu())

        out_path = os.path.join(args.output_dir, f"{lr_path.stem}_restored.png")
        pred_pil.save(out_path)
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
