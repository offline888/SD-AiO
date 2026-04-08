#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import torch
from PIL import Image
from diffusers import Flux2KleinPipeline
from torchvision import transforms
from PIL import ImageDraw

# ============ 配置 ============
MODEL_PATH = "/home/yhmi/data/model/flux.2-klein"
OUTPUT_DIR = "/home/yhmi/All_in_one/data/timestep_select/Haze"
IMAGE_PATH = "/home/yhmi/data/patches/08Haze/LQ_train/0347402_patch_01468_00420.jpg"


def tensor_to_pil(tensor):
    img = tensor.float().clamp(0, 1)
    return transforms.ToPILImage()(img.cpu())  

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pipeline = Flux2KleinPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    
    vae = pipeline.vae
    scheduler = pipeline.scheduler
    
    print(f"Scheduler timesteps shape: {scheduler.timesteps.shape}")
    print(f"Total timesteps: {len(scheduler.timesteps)}")
    
    print("Loading image...")
    image = Image.open(IMAGE_PATH).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # 归一化到[-1, 1]
    ])
    pixel_values = transform(image).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
    
    with torch.no_grad():
        latents = vae.encode(pixel_values).latent_dist.sample()
    
    print(f"Latent shape: {latents.shape}")  
    
    original_img = (pixel_values / 2 + 0.5).clamp(0, 1)
    tensor_to_pil(original_img[0]).save(
        os.path.join(OUTPUT_DIR, "00_original.png")
    )
    
    print("\n=== Scheduler Info ===")
    print(f"Timesteps range: [{scheduler.timesteps[0]}, {scheduler.timesteps[-1]}]")
    
    num_timesteps = len(scheduler.timesteps)
    
    timesteps_to_visualize = list(range(0, num_timesteps, num_timesteps // 10))
    
    print(f"\nVisualizing timesteps: {timesteps_to_visualize}")
    
    torch.manual_seed(42)
    fixed_noise = torch.randn_like(latents)
    
    for idx in timesteps_to_visualize:
        t = scheduler.timesteps[idx].item()
        
        
        if hasattr(scheduler, 'sigmas'):
            sigma = scheduler.sigmas[idx].item()
        else:
            sigma = t / 1000.0
        
        print(f"Timestep index {idx}: t={t}, sigma={sigma:.4f}")
        
        noisy_latents = (1 - sigma) * latents + sigma * fixed_noise
        
        with torch.no_grad():
            decoded = vae.decode(noisy_latents).sample.float()
            decoded = (decoded / 2 + 0.5).clamp(0, 1)

        prefix = f"step_{idx:04d}_t{int(t):04d}_s{sigma:.3f}"

        tensor_to_pil(decoded[0]).save(os.path.join(OUTPUT_DIR, f"{prefix}.png"))
    
    create_comparison_grid(OUTPUT_DIR, timesteps_to_visualize, scheduler)
    
    print(f"\nDone! Images saved to: {OUTPUT_DIR}")


def create_comparison_grid(output_dir, timestep_indices, scheduler):
   
    images = []
    for idx in timestep_indices:
        t = scheduler.timesteps[idx].item()
        prefix = f"step_{idx:04d}_t{int(t):04d}"
        img_path = os.path.join(output_dir, f"{prefix}.png")
        if os.path.exists(img_path):
            images.append((t, Image.open(img_path)))
    
    if not images:
        return
    
    # 创建grid (一行多个)
    img_size = images[0][1].size
    num_images = len(images)
    
    # 横向排列
    grid_width = num_images * img_size[0]
    grid_height = img_size[1]
    grid = Image.new('RGB', (grid_width, grid_height))
    
    for i, (t, img) in enumerate(images):
        grid.paste(img, (i * img_size[0], 0))
        
        # 添加文字标注
        draw = ImageDraw.Draw(grid)
        draw.text((i * img_size[0] + 5, 5), f"t={t}", fill='white')
    
    grid.save(os.path.join(output_dir, "comparison_grid.png"))


if __name__ == "__main__":
    main()