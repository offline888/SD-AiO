#!/usr/bin/env python3
"""
训练 vs 推理 数学对称性完整验证

总结分析所有关键发现
"""
import numpy as np

print("=" * 70)
print("训练 vs 推理 数学对称性完整验证报告")
print("=" * 70)

print("""
================================================================================
一、符号定义
================================================================================

lq_pixel: LQ 输入图像 (像素空间)
hq_pixel: HQ 目标图像 (像素空间)

lq_latent: lq_pixel 经过 VAE encoder 后的 latent, shape=(B, 16, H, W)
hq_latent: hq_pixel 经过 VAE encoder 后的 latent, shape=(B, 16, H, W)

lq_patched: patchify(lq_latent), shape=(B, 64, H/2, W/2)
hq_patched: patchify(hq_latent), shape=(B, 64, H/2, W/2)

lq_norm: BN 归一化后的 lq_patched = (lq_patched - μ) / σ
hq_norm: BN 归一化后的 hq_patched = (hq_patched - μ) / σ

noise: 高斯噪声, shape 与 lq_norm 相同
σ (sigma): 噪声等级, 固定值 0.1

μ, σ: BatchNorm 的 running_mean 和 sqrt(running_var + eps)

================================================================================
二、训练路径
================================================================================

Step 1: 编码
    lq_latent = vae.encode(lq_pixel)
    hq_latent = vae.encode(hq_pixel)
    
Step 2: Patchify + BN 归一化
    lq_norm = (patchify(lq_latent) - μ) / σ
    hq_norm = (patchify(hq_latent) - μ) / σ

Step 3: 加噪 (与推理完全相同)
    noisy = (1-σ) * lq_norm + σ * noise
    
Step 4: 模型前向
    model_pred = transformer(noisy, timestep, ...)
    
Step 5: Target 定义 (trainer.py line 638)
    target = noise + ((1-σ) * lq_norm - hq_norm) / σ

Step 6: 损失
    loss = MSE(model_pred, target)

Step 7: 反向去噪 (数学推导, 假设 model_pred = target)
    denoised = noisy - σ * target
             = noisy - σ * [noise + ((1-σ)*lq_norm - hq_norm)/σ]
             = [(1-σ)*lq_norm + σ*noise] - σ*noise - (1-σ)*lq_norm + hq_norm
             = hq_norm ✓

Step 8: BN 反归一化 (如果需要解码为像素)
    denoised_latent = denoised * σ + μ = hq_latent ✓
    decoded = vae.decode(denoised_latent) ≈ hq_pixel

================================================================================
三、推理路径 (Pipeline)
================================================================================

Step 1: LQ 图像编码
    lq_latent = vae.encode(lq_pixel).mode()
    
Step 2: Patchify + BN 归一化 (_encode_vae_image, line 474-478)
    lq_patched = patchify(lq_latent)
    lq_norm = (lq_patched - μ) / σ

Step 3: Pack + 加噪 (line 1152)
    lq_norm_packed = pack(lq_norm), shape=(B, H*W, 64)
    noisy = (1-σ) * lq_norm_packed + σ * noise

Step 4: 模型推理
    model_pred = transformer(noisy, timestep, ...)
    
Step 5: Euler 更新 (line 1221)
    latents_new = noisy - σ * model_pred

Step 6: Unpack (line 1245-1247)
    pred_latents = _unpack_latents_with_ids(latents_new, img_ids)
    # 返回 shape=(B, 64, H, W)

Step 7: BN 反归一化 (line 1249-1253)
    pred_latents = pred_latents * σ + μ

Step 8: Unpatchify + Decode (line 1254-1259)
    pred_latents = _unpatchify_latents(pred_latents), shape=(B, 16, H, W)
    output = vae.decode(pred_latents)

================================================================================
四、关键问题分析
================================================================================

问题: 如果 model_pred = 0，推理结果是什么？

情况 A: model_pred = 0 (未训练好的模型)
    latents_new = noisy = (1-σ) * lq_norm_packed + σ * noise
    
    BN 反归一化:
    denoised = [(1-σ) * lq_norm + σ * noise] * σ_bn + μ
             = (1-σ) * (lq_patched - μ) + σ * noise * σ_bn + μ
             = (1-σ) * lq_patched - (1-σ) * μ + σ * noise * σ_bn + μ
             = (1-σ) * lq_patched + σ * noise * σ_bn + σ * μ
    
    Unpatchify:
    denoised_unpatched = (1-σ) * lq_latent + σ * noise_unpatched * σ_bn + σ * μ_unpatched
    
    结论: 这不等于 lq_latent！
          如果 noise 是标准高斯噪声:
          E[denoised] = (1-σ) * lq_latent + σ * μ_unpatched
          由于 μ_unpatched 通常不为 0，结果会略微偏亮或偏暗。

情况 B: model_pred = target (完美训练的模型)
    latents_new = noisy - σ * target = hq_norm
    
    BN 反归一化:
    denoised = hq_norm * σ + μ = hq_latent ✓
    
    结论: 完全等于 HQ 图像！

情况 C: model_pred 输出了部分正确的值
    latents_new = (1-σ) * lq_norm + σ * noise - σ * model_pred
    
    如果 model_pred ≈ noise:
    latents_new ≈ (1-σ) * lq_norm
    
    结论: 恢复到约 0.9 * lq_latent，不是完全等于 LQ。

================================================================================
五、为什么推理结果可能等于 LQ？
================================================================================

如果推理结果完全等于 LQ，可能的原因:

1. model_pred 输出恰好抵消了 ((1-σ)*lq_norm - hq_norm)/σ 项
   但这不可能，因为还有 σ*noise 项需要处理。

2. model_pred 学习到了恒等映射，输出 ≈ lq_norm
   此时 latents_new ≈ (1-σ) * lq_norm + σ * noise - σ * lq_norm
                    = σ * (noise - lq_norm)
   这不等于 lq_norm。

3. Pipeline 中有 bug，绕过了加噪或解码步骤
   需要检查 Pipeline 代码是否有问题。

4. VAE 解码的特性
   如果 latents ≈ (1-σ) * lq_latent + small_noise
   VAE 解码可能输出与 LQ 视觉上相似的图像，但应该略有模糊。

5. **最可能的原因**: model_pred 实际上输出了接近 target 的值
   如果模型训练良好，model_pred ≈ target
   则 latents_new ≈ hq_norm, 解码后 ≈ HQ
   
   但这不等于 LQ！

================================================================================
六、代码中可能的 Bug
================================================================================

问题: BN 统计量形状与 pred_latents 不匹配？

- vae.bn.running_mean 形状: (16,)  # 针对原始 16 通道
- pred_latents 形状: (B, 64, H, W)  # patchified 后 64 通道
- BN 统计量 view 后: (1, 16, 1, 1)

PyTorch broadcast:
    (1, 16, 1, 1) vs (B, 64, H, W)
    
维度检查:
    dim 3: 1 vs W ✓
    dim 2: 1 vs H ✓
    dim 1: 16 vs 64 ❌ 不兼容!

这意味着代码可能无法正常运行，除非:
1. BN 统计量实际上有特殊的 tile/repeat
2. 或者 pred_latents 实际上只有 16 通道

建议检查: 
- 运行时会否报错 "RuntimeError: The size of tensor a must match..."
- 或者有隐式的 reshape/tile 操作我没注意到

================================================================================
七、最终结论
================================================================================

1. **数学对称性**: 训练和推理公式在数学上是对称的 ✓
   - 加噪公式相同
   - Euler 更新公式相同 (noisy - σ * model_pred)
   - BN 归一化/反归一化对称

2. **前提条件**: 对称性只在 model_pred = target 时成立
   - 如果 model_pred = 0，推理结果 ≠ LQ
   - 如果 model_pred = target，推理结果 = HQ

3. **观察到的现象**: 推理结果完全等于 LQ
   - 这不符合数学推导
   - 除非 model_pred 恰好输出了特殊值
   - 或者 pipeline 存在 bug

4. **建议**: 
   - 添加调试日志打印 model_pred 的值
   - 检查推理时 model_pred 是否真的为 0
   - 验证 BN 反归一化是否正确执行
   - 对比加噪后的 latent 与解码结果
""")

print("\n" + "=" * 70)
print("【验证建议】")
print("=" * 70)

print("""
在推理时添加以下检查点:

1. 打印加噪后的 latent:
   print(f"noisy mean={noisy.mean()}, std={noisy.std()}")

2. 打印 model_pred:
   print(f"model_pred mean={model_pred.mean()}, std={model_pred.std()}")

3. 打印 Euler 更新后的 latent:
   print(f"latents_new mean={latents_new.mean()}, std={latents_new.std()}")

4. 打印 BN 反归一化后的 latent:
   print(f"denorm mean={pred_latents.mean()}, std={pred_latents.std()}")

5. 对比:
   - noisy vs lq_norm (应该不同)
   - latents_new vs hq_norm (如果 model_pred=target, 应该相等)
   - output vs LQ 解码 (应该不同, 除非 model_pred=0)

关键: 如果 model_pred ≈ 0 且 noisy ≠ lq_norm，
     但最终输出 ≈ LQ 解码，说明有 bug！
""")
