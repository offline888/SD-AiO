#!/usr/bin/env python3
"""
完整的端到端对称性验证
使用与 pipeline 完全相同的实现
"""
import torch
from diffusers.models.autoencoders import AutoencoderKLFlux2

device = 'cuda:0'

print("=== 1. 加载 VAE ===")
vae = AutoencoderKLFlux2.from_pretrained(
    '/home/yhmi/data/model/flux.2-klein',
    subfolder='vae'
)
vae.to(device)
vae.eval()

# BN stats
bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device)
bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(device)
print(f"BN mean shape: {bn_mean.shape}")
print(f"BN std shape: {bn_std.shape}")
print(f"BN mean range: [{bn_mean.min().item():.4f}, {bn_mean.max().item():.4f}]")
print(f"BN std range: [{bn_std.min().item():.4f}, {bn_std.max().item():.4f}]")

# Pipeline 中的 patchify/unpatchify 实现
def _patchify_latents(latents):
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
    return latents

def _unpatchify_latents(latents):
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), height * 2, width * 2)
    return latents

print("\n=== 2. 验证 patchify/unpatchify 循环一致性 ===")
torch.manual_seed(42)
orig = torch.randn(1, 32, 128, 128, device=device)
patched = _patchify_latents(orig)
unpatched = _unpatchify_latents(patched)
print(f"原始: {orig.shape}")
print(f"Patchified: {patched.shape}")
print(f"Unpatchified: {unpatched.shape}")
print(f"循环误差: {(orig - unpatched).abs().mean().item():.10f}")
print(f"完美恢复: {torch.allclose(orig, unpatched, atol=1e-6)}")

print("\n=== 3. 训练公式对称性验证 ===")
torch.manual_seed(123)

# 模拟 LQ 和 HQ latent (C=32)
lq_latent = torch.randn(1, 32, 128, 128, device=device)
hq_latent = torch.randn(1, 32, 128, 128, device=device)

print(f"LQ latent: mean={lq_latent.mean().item():.4f}, std={lq_latent.std().item():.4f}")
print(f"HQ latent: mean={hq_latent.mean().item():.4f}, std={hq_latent.std().item():.4f}")

# 训练流程 (与 trainer.py 一致):
# 1. Patchify
lq_patch = _patchify_latents(lq_latent)  # C=32 -> C=128
hq_patch = _patchify_latents(hq_latent)

# 2. BN normalize
lq_norm = (lq_patch - bn_mean) / bn_std
hq_norm = (hq_patch - bn_mean) / bn_std

print(f"\nAfter BN normalize:")
print(f"  LQ norm: mean={lq_norm.mean().item():.4f}, std={lq_norm.std().item():.4f}")
print(f"  HQ norm: mean={hq_norm.mean().item():.4f}, std={hq_norm.std().item():.4f}")

# 3. 加噪
sigma = 0.1
noise = torch.randn_like(lq_norm)
noisy = (1 - sigma) * lq_norm + sigma * noise

# 4. Target (与 trainer.py line 638 一致)
target = noise + ((1 - sigma) * lq_norm - hq_norm) / sigma

print(f"\n加噪后: mean={noisy.mean().item():.4f}, std={noisy.std().item():.4f}")
print(f"Target: mean={target.mean().item():.4f}, std={target.std().item():.4f}")

print("\n=== 4. 推理公式对称性验证 ===")
# 假设完美模型: model_pred = target
model_pred = target.clone()

# Euler 更新: latents = latents - sigma * model_pred
denoised_norm = noisy - sigma * model_pred

print(f"Euler 去噪后 (normalized):")
print(f"  mean={denoised_norm.mean().item():.4f}, std={denoised_norm.std().item():.4f}")
print(f"  与 HQ_norm 的差异: {(denoised_norm - hq_norm).abs().mean().item():.10f}")

# BN denormalize
denoised_patch = denoised_norm * bn_std + bn_mean

# Unpatchify
denoised_latent = _unpatchify_latents(denoised_patch)

print(f"\n最终去噪 latent:")
print(f"  mean={denoised_latent.mean().item():.4f}, std={denoised_latent.std().item():.4f}")
print(f"  与原始 HQ latent 的差异: {(denoised_latent - hq_latent).abs().mean().item():.10f}")

print(f"\n=== 5. 结论 ===")
if torch.allclose(denoised_latent, hq_latent, atol=1e-4):
    print("✅ 公式完全对称！Euler 一步即可完美恢复 HQ latent")
else:
    print("❌ 公式不对称！存在误差")
    print(f"最大误差: {(denoised_latent - hq_latent).abs().max().item():.8f}")
    print(f"平均误差: {(denoised_latent - hq_latent).abs().mean().item():.8f}")

print("\n=== 6. 数学推导验证 ===")
print("""
训练时的公式:
  noisy = (1-σ)*LQ_norm + σ*noise
  target = noise + ((1-σ)*LQ_norm - HQ_norm)/σ

推理时的 Euler 更新:
  denoised_norm = noisy - σ*model_pred

当 model_pred = target (完美模型):
  denoised_norm = (1-σ)*LQ_norm + σ*noise - σ*(noise + ((1-σ)*LQ_norm - HQ_norm)/σ)
                = (1-σ)*LQ_norm + σ*noise - σ*noise - ((1-σ)*LQ_norm - HQ_norm)
                = (1-σ)*LQ_norm - (1-σ)*LQ_norm + HQ_norm
                = HQ_norm ✓

然后:
  denoised_patch = HQ_norm * σ_std + σ_mean = HQ_patch
  denoised_latent = _unpatchify_latents(HQ_patch) = HQ ✓
""")
