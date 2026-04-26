#!/usr/bin/env python3
"""
训练 vs 推理 数学对称性验证 - 纯数学推导，无需 GPU
"""
import numpy as np

print("=" * 70)
print("训练 vs 推理 数学对称性验证")
print("=" * 70)

# =============================================================
# 定义符号
# =============================================================
# lq_latent: LQ 图像的 VAE latent (patchified, C=64)
# hq_latent: HQ 图像的 VAE latent (patchified, C=64)
# lq_norm: BN 归一化后的 lq_latent = (lq_latent - mean) / std
# hq_norm: BN 归一化后的 hq_latent
# noise: 高斯噪声
# sigma: 噪声等级
# bn_mean, bn_std: BatchNorm 的 running mean/std

print("\n【符号定义】")
print("  lq_latent: LQ 图像的 VAE latent (patchified)")
print("  hq_latent: HQ 图像的 VAE latent (patchified)")
print("  lq_norm = (lq_latent - μ) / σ_bn")
print("  hq_norm = (hq_latent - μ) / σ_bn")
print("  noise: 高斯噪声")
print("  σ: 噪声等级 (如 0.1 = 10% 噪声)")

# =============================================================
# 关键参数
# =============================================================
sigma = 0.1  # 固定噪声等级
bn_mean = 0.5  # BN 均值
bn_std = 1.5  # BN 标准差

print(f"\n【参数】")
print(f"  σ (sigma) = {sigma}")
print(f"  BN mean = {bn_mean}")
print(f"  BN std = {bn_std}")

# =============================================================
# 训练路径推导
# =============================================================
print("\n" + "=" * 70)
print("【训练路径】")
print("=" * 70)

print("\n1. 编码 + BN 归一化:")
print(f"   lq_norm = (lq_latent - {bn_mean}) / {bn_std}")
print(f"   hq_norm = (hq_latent - {bn_mean}) / {bn_std}")

print("\n2. 加噪 (公式与推理相同):")
print(f"   noisy = (1-σ) * lq_norm + σ * noise")
print(f"   noisy = (1-{sigma}) * lq_norm + {sigma} * noise")
print(f"   noisy = {1-sigma} * lq_norm + {sigma} * noise")

print("\n3. Target 定义 (训练代码 line 638):")
print(f"   target = noise + ((1-σ) * lq_norm - hq_norm) / σ")
print(f"          = noise + ({1-sigma} * lq_norm - hq_norm) / {sigma}")

print("\n4. 反向去噪推导 (假设 model_pred = target):")
print("   denoised = noisy - σ * target")
print("            = noisy - σ * [noise + ((1-σ)*lq_norm - hq_norm)/σ]")
print("            = noisy - σ*noise - ((1-σ)*lq_norm - hq_norm)")
print("            = [(1-σ)*lq_norm + σ*noise] - σ*noise - (1-σ)*lq_norm + hq_norm")
print("            = hq_norm")

print("\n   **结论: 如果 model_pred = target，反向去噪得到 hq_norm ✓**")

print("\n5. BN 反归一化 (如果需要解码):")
print("   denoised_latent = denoised * σ_bn + μ")
print("                    = hq_norm * σ_bn + μ")
print("                    = [(hq_latent - μ) / σ_bn] * σ_bn + μ")
print("                    = hq_latent - μ + μ")
print("                    = hq_latent")

print("\n   **结论: 反归一化后得到原始的 hq_latent，可以解码为 HQ 图像 ✓**")

# =============================================================
# 推理路径推导
# =============================================================
print("\n" + "=" * 70)
print("【推理路径 - Pipeline 流程】")
print("=" * 70)

print("\n1. LQ 图像编码:")
print("   lq_latent = vae.encode(lq_pixel).mode()")
print("   lq_latent_patched = patchify(lq_latent)")

print("\n2. BN 归一化 (line 476-478):")
print(f"   lq_norm = (lq_latent_patched - {bn_mean}) / {bn_std}")

print("\n3. Pack + 加噪:")
print(f"   lq_norm_packed = pack(lq_norm)")
print(f"   noisy = (1-σ) * lq_norm_packed + σ * noise")
print(f"         = {1-sigma} * lq_norm_packed + {sigma} * noise")

print("\n4. 模型推理:")
print("   model_pred = transformer(noisy, timestep, ...)")
print("")
print("   **期望: model_pred = noise + ((1-σ)*lq_norm - hq_norm) / σ**")
print("          (与训练时的 target 相同)")

print("\n5. Euler 更新:")
print("   latents_new = noisy - σ * model_pred")

print("\n   情况分析:")
print("   ─────────────────────────────────────────────────────────")
print("   情况 A: model_pred = 0 (未训练好的模型)")
print("   ─────────────────────────────────────────────────────────")
print("   latents_new = noisy - σ * 0 = noisy")
print("               = (1-σ) * lq_norm_packed + σ * noise")
print("")
print("   BN 反归一化后:")
print("   denoised = latents_new * σ_bn + μ")
print("            = [(1-σ)*lq_norm + σ*noise] * σ_bn + μ")
print("            = (1-σ)*(lq_latent-μ) + σ*noise*σ_bn + μ")
print("            = (1-σ)*lq_latent - (1-σ)*μ + σ*noise*σ_bn + μ")
print("            = (1-σ)*lq_latent + σ*noise*σ_bn + σ*μ")  # 因为 - (1-σ)*μ + μ = σ*μ
print("")
print("   **问题: 这不是原始的 lq_latent！**")
print("   **它是 90% lq_latent + 10% 缩放后的噪声**")

print("\n   ─────────────────────────────────────────────────────────")
print("   情况 B: model_pred = target (完美训练的模型)")
print("   ─────────────────────────────────────────────────────────")
print("   latents_new = noisy - σ * target")
print("               = noisy - σ * [noise + ((1-σ)*lq_norm - hq_norm)/σ]")
print("               = hq_norm")
print("")
print("   BN 反归一化后:")
print("   denoised = hq_norm * σ_bn + μ = hq_latent")
print("")
print("   **正确: 解码后得到 HQ 图像 ✓**")

print("\n   ─────────────────────────────────────────────────────────")
print("   情况 C: model_pred 恰好抵消了 σ*noise 部分")
print("   ─────────────────────────────────────────────────────────")
print("   如果 model_pred ≈ noise (忽略 ((1-σ)*lq_norm - hq_norm)/σ 项)")
print("   latents_new = (1-σ)*lq_norm_packed + σ*noise - σ*noise = (1-σ)*lq_norm")
print("               = 0.9 * lq_norm")
print("")
print("   BN 反归一化后:")
print("   denoised ≈ 0.9 * lq_latent")
print("")
print("   **问题: 恢复到 90% 的 LQ，不是完全等于 LQ！**")

# =============================================================
# 关键问题分析
# =============================================================
print("\n" + "=" * 70)
print("【关键问题分析】")
print("=" * 70)

print("""
如果 model_pred = 0:
  latents_new = noisy = 0.9 * lq_norm + 0.1 * noise
  BN 反归一化: 0.9 * (lq_latent-μ) + 0.1*noise*σ_bn + μ
              = 0.9 * lq_latent - 0.9*μ + 0.1*noise*σ_bn + μ
              = 0.9 * lq_latent + 0.1*noise*σ_bn + 0.1*μ

如果 noise 是标准高斯分布，均值为 0，方差为 1：
  E[denoised] = 0.9 * lq_latent + 0.1*μ

这不等于 lq_latent！
除非 noise*σ_bn 的均值恰好等于 0.1*(lq_latent - μ)
但这是不可能的，因为 noise 是随机的。

结论: 如果 model_pred = 0，推理结果应该是略微模糊的 0.9*LQ + 0.1*noise，
      而不是完全等于 LQ。
""")

print("\n" + "=" * 70)
print("【实际 Pipeline 代码检查】")
print("=" * 70)

print("""
Pipeline 关键代码 (flux2_klein.py):

1. 编码 LQ 图像 (line 1121-1123):
   lq_latents = self._encode_vae_image(img, generator)
   # 返回: patchify(latent) / BN normalize

2. 加噪 (line 1152):
   current_latents = (1.0 - sigma_start) * lq_latents_packed + sigma_start * noise
   # 注意: lq_latents_packed 已经是 BN 归一化后的

3. Euler 更新 (line 1221):
   current_latents = current_latents - sigma_t * model_pred

4. BN 反归一化 (line 1249-1253):
   pred_latents = pred_latents * latents_bn_std + latents_bn_mean

5. Unpatchify + Decode (line 1254-1259):
   pred_latents = self._unpatchify_latents(pred_latents)
   image = self.vae.decode(pred_latents.to(dtype=self.vae.dtype), ...)

**检查: BN 反归一化前后的 latent 形状是否匹配？**

在 line 1245-1247:
   pred_latents = self._unpack_latents_with_ids(current_latents, img_ids, ...)

img_ids 来自 line 1128:
   img_ids = self._prepare_latent_ids(lq_latents).to(device)

**关键问题**: lq_latents 是 patchify 后的 (B, 64, H/2, W/2)，
而 _unpack_latents_with_ids 需要正确的 latent_ids 来 unpack。

如果 model_pred = 0:
  current_latents = noisy = 0.9 * lq_norm_packed + 0.1 * noise_packed
  经过 _unpack_latents_with_ids -> _unpatchify -> BN denorm -> decode

  结果应该不等于原始 lq_latent 解码后的图像！

除非有其他 bug...
""")

# =============================================================
# 结论
# =============================================================
print("\n" + "=" * 70)
print("【最终结论】")
print("=" * 70)

print("""
数学推导表明:
1. 如果 model_pred = 0，推理结果 = 0.9*LQ + 0.1*noise (略微模糊)
2. Pipeline 有正确的 BN 反归一化步骤 (line 1249-1253)
3. 数学上应该不会完全等于 LQ

如果推理结果完全等于 LQ，可能的原因:
  A. model_pred 输出恰好抵消了 σ*noise，使 latents ≈ 0.9*lq_norm
     → 这仍然应该是略微模糊的
  B. Pipeline 中有 bug，绕过了加噪步骤
  C. VAE 解码的特性：0.9*LQ + 0.1*noise 解码后视觉上接近 LQ
  D. 模型实际上输出了接近 target 的值

**对称性分析**:
- 训练时: lq_norm --加噪--> noisy --Euler反推--> hq_norm ✓
- 推理时: lq_norm --加噪--> noisy --Euler更新--> ???

推理路径的 ???
  = noisy - σ * model_pred
  = hq_norm (如果 model_pred = target)
  = 0.9 * lq_norm + 0.1 * noise (如果 model_pred = 0)

数学上完全对称，但前提是 model_pred = target。
""")
