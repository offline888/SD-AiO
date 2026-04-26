#!/usr/bin/env python3
import torch

device = 'cuda:0'

# 模拟一个简单的测试
torch.manual_seed(42)

# 假设 batch_size=1, latent shape (1, 16, 32, 32)
bsz = 1
C, H, W = 16, 32, 32
lq = torch.randn(bsz, C, H, W, device=device, dtype=torch.bfloat16)
hq = torch.randn(bsz, C, H, W, device=device, dtype=torch.bfloat16)
noise = torch.randn_like(lq)

# 固定的 sigma (对应 fixed_idx=900)
sigma = 0.1  # 对应 90% LQ + 10% 噪声

print("=== Training 公式 ===")
print(f"sigma = {sigma}")
print(f"LQ mean: {lq.mean().item():.4f}, std: {lq.std().item():.4f}")
print(f"HQ mean: {hq.mean().item():.4f}, std: {hq.std().item():.4f}")
print(f"Noise mean: {noise.mean().item():.4f}, std: {noise.std().item():.4f}")

# 加噪公式
noisy_lq = (1.0 - sigma) * lq + sigma * noise
print(f"\n加噪: noisy = (1-sigma)*LQ + sigma*Noise")
print(f"noisy mean: {noisy_lq.mean().item():.4f}, std: {noisy_lq.std().item():.4f}")

# Target 公式 (来自 trainer.py line 638)
target = noise + ((1 - sigma) * lq - hq) / sigma
print(f"\nTarget: noise + ((1-sigma)*LQ - HQ) / sigma")
print(f"target mean: {target.mean().item():.4f}, std: {target.std().item():.4f}")

# 验证：model_pred 应该是 target 的某种近似
# 假设模型是"完美"的，model_pred = target
model_pred = target.clone()
print(f"\n假设 model_pred = target (完美模型)")

# 反推去噪
# noisy - sigma * model_pred = ?
denoised = noisy_lq - sigma * model_pred
print(f"去噪: noisy - sigma * model_pred = LQ - ((1-sigma)*LQ - HQ) - noise")
print(f"     = LQ - (1-sigma)*LQ + HQ - noise")
print(f"     = sigma * LQ + HQ - noise")
print(f"去噪后 mean: {denoised.mean().item():.4f}, std: {denoised.std().item():.4f}")

# 期望: 去噪后应该接近 HQ
print(f"\n对比:")
print(f"  去噪后 mean={denoised.mean().item():.4f}, std={denoised.std().item():.4f}")
print(f"  原始 HQ mean={hq.mean().item():.4f}, std={hq.std().item():.4f}")

# 误差
diff = (denoised - hq).abs().mean().item()
print(f"  与HQ的绝对误差: {diff:.4f}")

# =============================================
# 关键验证: 从推理角度反向推导
# =============================================
print("\n\n=== 推理公式 (Pipeline) ===")

# Pipeline 中：
# 1. 加噪: current = (1-sigma)*LQ + sigma*Noise  (与训练一致)
# 2. 模型输入: current_latents
# 3. 模型输出: model_pred (期望是 noise + ((1-sigma)*LQ - HQ)/sigma)
# 4. Euler更新: latents -= sigma * model_pred

# 验证 Euler 更新是否正确
print(f"Euler更新: latents_new = latents - sigma * model_pred")
latents_new = noisy_lq - sigma * model_pred
print(f"  latents_new = {(1-sigma)*lq + sigma*noise - sigma*(noise + ((1-sigma)*lq - hq)/sigma)}")
print(f"  latents_new = {(1-sigma)*lq + sigma*noise - sigma*noise - ((1-sigma)*lq - hq)}")
print(f"  latents_new = {(1-sigma)*lq - (1-sigma)*lq + hq}")
print(f"  latents_new = hq (完美重建!)")
print(f"\n  实际计算: mean={latents_new.mean().item():.4f}, std={latents_new.std().item():.4f}")
print(f"  期望 HQ:   mean={hq.mean().item():.4f}, std={hq.std().item():.4f}")
print(f"  误差: {torch.abs(latents_new - hq).mean().item():.8f}")

# =============================================
# 检查 normalize/denormalize 是否在 pipeline 中正确处理
# =============================================
print("\n\n=== 检查 BN normalize/denormalize 对称性 ===")

# 在训练中：
# 1. lq_latent = vae.encode(lq_pixel).mode()
# 2. lq_normalized = (lq_latent - bn_mean) / bn_std
# 3. noisy = (1-sigma)*lq_normalized + sigma*noise
# 4. target = noise + ((1-sigma)*lq_normalized - hq_normalized) / sigma

# 假设 bn_mean=0, bn_std=1 (理想情况)
bn_mean = torch.zeros_like(lq)
bn_std = torch.ones_like(lq)

lq_norm = (lq - bn_mean) / bn_std
hq_norm = (hq - bn_mean) / bn_std
noisy_norm = (1-sigma)*lq_norm + sigma*noise
target_norm = noise + ((1-sigma)*lq_norm - hq_norm) / sigma

# 推理中 (pipeline):
# 1. lq_latent = vae.encode(lq_pixel).mode()
# 2. lq_normalized = (lq_latent - bn_mean) / bn_std
# 3. noisy = (1-sigma)*lq_normalized + sigma*noise
# 4. model_pred = ...
# 5. latents_new = noisy - sigma * model_pred
# 6. decode(latents_new * bn_std + bn_mean)

latents_new_norm = noisy_norm - sigma * target_norm  # 完美模型
print(f"Euler去噪后 (normalized): mean={latents_new_norm.mean().item():.4f}")
print(f"HQ normalized: mean={hq_norm.mean().item():.4f}")

# 关键: 解码前需要反归一化吗？
# latents_new_norm = hq_norm (当模型完美时)
# 但 VAE decode 需要原始尺度的 latent
# decoded = vae.decode(latents_new_norm * bn_std + bn_mean)
latents_for_decode = latents_new_norm * bn_std + bn_mean
print(f"解码前 (denormalized): mean={latents_for_decode.mean().item():.4f}")
print(f"期望 HQ latent: mean={hq.mean().item():.4f}")

# =============================================
# 检查实际的 bn_mean/bn_std
# =============================================
print("\n\n=== 检查实际 VAE BN 统计量 ===")
from diffusers.models.autoencoders import AutoencoderKLFlux2
vae = AutoencoderKLFlux2.from_pretrained(
    '/home/yhmi/data/model/flux.2-klein',
    subfolder='vae'
)
vae.to(device)
vae.eval()

bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device)
bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(device)
print(f"bn_mean: min={bn_mean.min().item():.4f}, max={bn_mean.max().item():.4f}")
print(f"bn_std: min={bn_std.min().item():.4f}, max={bn_std.max().item():.4f}")
print(f"bn_mean 是否全零: {(bn_mean == 0).all().item()}")
print(f"bn_std 是否全一: {((bn_std - 1.0).abs() < 0.01).all().item()}")
