#!/usr/bin/env python3
"""
调试 BN 对称性测试
"""
import torch
from diffusers.models.autoencoders import AutoencoderKLFlux2

device = 'cuda:0'

print("=== 加载 VAE ===")
vae = AutoencoderKLFlux2.from_pretrained(
    '/home/yhmi/data/model/flux.2-klein',
    subfolder='vae'
)
vae.to(device)
vae.eval()

# BN stats
bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device)
bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(device)
print(f"BN mean range: [{bn_mean.min().item():.4f}, {bn_mean.max().item():.4f}]")
print(f"BN std range: [{bn_std.min().item():.4f}, {bn_std.max().item():.4f}]")

# Patchify 函数
def patchify_latents(latents):
    batch_size, num_channels, height, width = latents.shape
    latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(batch_size, num_channels * 4, height // 2, width // 2)
    return latents

def unpatchify_latents(latents):
    batch_size, num_channels, height, width = latents.shape
    channels = num_channels // 4
    latents = latents.view(batch_size, channels, 4, height, width)
    latents = latents.permute(0, 1, 3, 2, 4)
    latents = latents.reshape(batch_size, channels, height * 2, width * 2)
    return latents

print("\n=== 使用固定种子 ===")
torch.manual_seed(123)

# LQ 和 HQ latent (C=32)
lq = torch.randn(1, 32, 128, 128, device=device)
hq = torch.randn(1, 32, 128, 128, device=device)

print(f"原始 LQ: mean={lq.mean().item():.4f}, std={lq.std().item():.4f}")
print(f"原始 HQ: mean={hq.mean().item():.4f}, std={hq.std().item():.4f}")

# Step 1: Patchify
lq_patch = patchify_latents(lq)
hq_patch = patchify_latents(hq)
print(f"\nAfter patchify:")
print(f"  LQ: mean={lq_patch.mean().item():.4f}, std={lq_patch.std().item():.4f}, shape={lq_patch.shape}")
print(f"  HQ: mean={hq_patch.mean().item():.4f}, std={hq_patch.std().item():.4f}, shape={hq_patch.shape}")

# Step 2: BN normalize
lq_norm = (lq_patch - bn_mean) / bn_std
hq_norm = (hq_patch - bn_mean) / bn_std
print(f"\nAfter BN normalize:")
print(f"  LQ: mean={lq_norm.mean().item():.4f}, std={lq_norm.std().item():.4f}")
print(f"  HQ: mean={hq_norm.mean().item():.4f}, std={hq_norm.std().item():.4f}")

# Step 3: 加噪
sigma = 0.1
noise = torch.randn_like(lq_norm)
noisy = (1 - sigma) * lq_norm + sigma * noise
print(f"\n加噪后: mean={noisy.mean().item():.4f}, std={noisy.std().item():.4f}")

# Step 4: Target
target = noise + ((1 - sigma) * lq_norm - hq_norm) / sigma
print(f"Target: mean={target.mean().item():.4f}, std={target.std().item():.4f}")

# Step 5: 完美模型
model_pred = target.clone()

# Step 6: Euler 更新
denoised_norm = noisy - sigma * model_pred
print(f"\nEuler 去噪后 (normalized): mean={denoised_norm.mean().item():.4f}, std={denoised_norm.std().item():.4f}")
print(f"HQ normalized: mean={hq_norm.mean().item():.4f}, std={hq_norm.std().item():.4f}")

# 检查是否等于 hq_norm
diff_norm = (denoised_norm - hq_norm).abs().mean().item()
print(f"与 HQ_norm 的差异: {diff_norm:.8f}")

# Step 7: BN denormalize
denoised_patch = denoised_norm * bn_std + bn_mean

# Step 8: Unpatchify
denoised = unpatchify_latents(denoised_patch)

print(f"\n最终结果:")
print(f"  去噪后: mean={denoised.mean().item():.4f}, std={denoised.std().item():.4f}")
print(f"  原始 HQ: mean={hq.mean().item():.4f}, std={hq.std().item():.4f}")
diff = (denoised - hq).abs().mean().item()
print(f"  与原始 HQ 的差异: {diff:.8f}")
print(f"  完美对称: {torch.allclose(denoised, hq, atol=1e-4)}")

print("\n\n=== 数学推导验证 ===")
print("noisy = (1-σ)*LQ_norm + σ*noise")
print("target = noise + ((1-σ)*LQ_norm - HQ_norm)/σ")
print("denoised_norm = noisy - σ*target")
print("            = (1-σ)*LQ_norm + σ*noise - σ*noise - ((1-σ)*LQ_norm - HQ_norm)")
print("            = (1-σ)*LQ_norm - (1-σ)*LQ_norm + HQ_norm")
print("            = HQ_norm ✓")

# 手动计算验证
manual_denoised_norm = (1-sigma)*lq_norm + sigma*noise - sigma*(noise + ((1-sigma)*lq_norm - hq_norm)/sigma)
print(f"\n手动计算 denoised_norm: mean={manual_denoised_norm.mean().item():.8f}")
print(f"直接计算 denoised_norm: mean={denoised_norm.mean().item():.8f}")
print(f"差异: {(manual_denoised_norm - denoised_norm).abs().max().item():.8f}")
