#!/usr/bin/env python3
"""
验证 BN normalize/denormalize 的对称性
"""
import torch
from diffusers.models.autoencoders import AutoencoderKLFlux2

device = 'cuda:0'

print("=== 1. 加载 VAE 并检查 BN 统计量 ===")
vae = AutoencoderKLFlux2.from_pretrained(
    '/home/yhmi/data/model/flux.2-klein',
    subfolder='vae'
)
vae.to(device)
vae.eval()

# BN stats 的实际 shape
bn_mean_raw = vae.bn.running_mean.view(-1)
bn_var_raw = vae.bn.running_var.view(-1)
print(f"BN running_mean shape: {bn_mean_raw.shape}  (C=128)")
print(f"BN running_var shape: {bn_var_raw.shape}")
print(f"BN mean range: [{bn_mean_raw.min().item():.4f}, {bn_mean_raw.max().item():.4f}]")
print(f"BN std range: [{torch.sqrt(bn_var_raw + vae.config.batch_norm_eps).min().item():.4f}, {torch.sqrt(bn_var_raw + vae.config.batch_norm_eps).max().item():.4f}]")

# 这表明 VAE 的 latent space 是 C=128！
# 所以 patchify 后变成 C=512
print(f"\n>>> 关键发现: BN 在 C=128 维度，VAE latent 空间是 128 通道！")

print("\n=== 2. patchify 函数验证 ===")
# patchify: (B, C, H, W) -> (B, C*4, H//2, W//2)
# 所以如果 C=128, patchify 后变成 C=512

# 模拟测试
test_latent = torch.randn(1, 128, 32, 32, device=device, dtype=torch.bfloat16)
print(f"VAE latent shape: {test_latent.shape}")

# 手动 patchify
def patchify_latents(latents):
    batch_size, num_channels, height, width = latents.shape
    latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(batch_size, num_channels * 4, height // 2, width // 2)
    return latents

patchified = patchify_latents(test_latent)
print(f"Patchified shape: {patchified.shape}")

print("\n=== 3. BN normalize/denormalize 对称性测试 ===")
torch.manual_seed(42)

# BN stats reshape: (128,) -> (1, 128, 1, 1)
bn_mean = bn_mean_raw.view(1, -1, 1, 1)
bn_std = torch.sqrt(bn_var_raw + vae.config.batch_norm_eps).view(1, -1, 1, 1)

# 模拟 LQ 和 HQ (在 VAE latent space, C=128)
lq_latent = torch.randn(1, 128, 64, 64, device=device, dtype=torch.bfloat16)
hq_latent = torch.randn(1, 128, 64, 64, device=device, dtype=torch.bfloat16)

print(f"LQ latent shape: {lq_latent.shape}")

# 训练流程:
# 1. BN normalize
lq_norm = (lq_latent - bn_mean.to(lq_latent.dtype)) / bn_std.to(lq_latent.dtype)
hq_norm = (hq_latent - bn_mean.to(hq_latent.dtype)) / bn_std.to(hq_latent.dtype)

# 2. Patchify
lq_patch = patchify_latents(lq_norm)
hq_patch = patchify_latents(hq_norm)

print(f"After patchify: LQ={lq_patch.shape}, HQ={hq_patch.shape}")

# 3. 加噪
sigma = 0.1
noise = torch.randn_like(lq_patch)
noisy = (1 - sigma) * lq_patch + sigma * noise

# 4. Target (在 patchified space)
target = noise + ((1 - sigma) * lq_patch - hq_patch) / sigma

# 5. 假设完美模型: model_pred = target

# 6. Euler 更新
denoised_patch = noisy - sigma * target

# 7. Unpatchify
def unpatchify_latents(latents):
    batch_size, num_channels, height, width = latents.shape
    channels = num_channels // 4
    latents = latents.view(batch_size, channels, 4, height, width)
    latents = latents.permute(0, 1, 3, 2, 4)
    latents = latents.reshape(batch_size, channels, height * 2, width * 2)
    return latents

denoised_unpatch = unpatchify_latents(denoised_patch)

# 8. BN denormalize (推理时需要！)
denoised = denoised_unpatch * bn_std.to(denoised_unpatch.dtype) + bn_mean.to(denoised_unpatch.dtype)

print(f"\n结果对比:")
print(f"  去噪后 latent mean: {denoised.mean().item():.6f}")
print(f"  原始 HQ mean:       {hq_latent.mean().item():.6f}")
print(f"  误差: {torch.abs(denoised - hq_latent).mean().item():.8f}")
print(f"  完美对称: {torch.allclose(denoised, hq_latent, atol=1e-4)}")

print("\n=== 4. 检查训练代码中 BN 的使用 ===")
print("在 trainer.py 中:")
print("  1. lq_latent = vae.encode(lq_pixel).mode()")
print("  2. model_input = patchify_latents(lq_latent)")
print("  3. model_input = (model_input - bn_mean) / bn_std  <-- BN 在 patchify 之后应用!")
print("  4. noisy = (1-sigma)*model_input + sigma*noise")
print("  5. target = noise + ((1-sigma)*lq_input - hq_target)/sigma")
print("\n问题是: BN stats shape (1, 128, 1, 1) vs patchified shape (1, 512, H, W)")
print("广播后: 只有 C=128 被归一化，不对称！")

# 验证
print(f"\n>>> 验证 BN stats 与 patchified 的 broadcast:")
print(f"BN mean shape: {bn_mean.shape}")
print(f"Patchified shape: {lq_patch.shape}")
print(f"两者能否广播: 可以，但只在 C 维度取前 128 通道！")

# 实际上 Python 会广播 C 维度:
# (1, 128, H, W) broadcast to (1, 512, H, W)
# 结果是前 128 通道用 BN stats，后 384 通道用相同的最后 BN 值
test_norm = (lq_patch - bn_mean.to(lq_patch.dtype))
print(f"\nNormalized 后唯一值数量 (每个位置): {test_norm[0, :, 0, 0].unique().shape}")

print("\n=== 5. 数学对称性验证 (简化版) ===")
# 简化：假设 VAE latent C=16, patchify 后 C=64
# 但 BN stats C=128，这不对应任何 VAE latent 空间
# 需要重新检查...

# 实际上从 bn_mean_raw.shape = (128,) 来看
# VAE 的 latent space 应该是 C=16, 但 BN 在更深的层
# 或者 C=128 是整个 VAE encoder 的输出

# 检查 vae.config
print(f"\nVAE latent channels: {vae.config.latent_channels}")
print(f"这说明 VAE 输出 latent 是 {vae.config.latent_channels} 通道")
print(f"但 BN running_mean 有 {bn_mean_raw.shape[0]} 通道")
print(f">>> BN 在 VAE encoder 的中间层，不在 latent 输出层！")

# 让我检查 VAE 的结构
print("\n=== 6. 检查 VAE 结构中 BN 的位置 ===")
# bn 是 vae.bn，这是 VAEEncoder 或类似组件中的一个 BatchNorm 层
# 它的作用是什么？

# 从 AutoencoderKLFlux2 的结构来看:
# encoder 输出可能是 C=128，然后有个 BN 层
# 但最终的 latent_dist 可能是不同的通道数

# 让我打印 VAE 结构
print("\nVAE bn layer type:", type(vae.bn))
print("VAE bn layer:", vae.bn)
