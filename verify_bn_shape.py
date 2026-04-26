#!/usr/bin/env python3
"""
检查 BN mean/std 的形状与 pred_latents 的形状是否匹配
"""
import numpy as np

print("=" * 70)
print("BN 形状匹配分析")
print("=" * 70)

# 假设配置
original_channels = 16  # VAE latent 的原始通道数
patchify_factor = 4      # 2x2 patchify，所以是 4 倍
patchified_channels = original_channels * patchify_factor  # 64

print(f"\n【通道配置】")
print(f"  原始 VAE latent 通道数: {original_channels}")
print(f"  Patchify 后通道数: {patchified_channels} (= {original_channels} * 4)")

# BN mean/std 的形状
bn_mean_shape = (1, original_channels, 1, 1)  # view(1, -1, 1, 1) 后的形状
bn_std_shape = (1, original_channels, 1, 1)

print(f"\n【BN 统计量形状】")
print(f"  running_mean view(1, -1, 1, 1): {bn_mean_shape}")
print(f"  running_var view(1, -1, 1, 1): {bn_std_shape}")

# pred_latents 的形状
B, H, W = 1, 32, 32
pred_latents_shape = (B, patchified_channels, H, W)

print(f"\n【pred_latents 形状】")
print(f"  形状: {pred_latents_shape}")

print("\n" + "=" * 70)
print("【Broadcast 分析】")
print("=" * 70)

print(f"\nbn_mean/std: {bn_mean_shape}")
print(f"pred_latents: {pred_latents_shape}")

print("""
当执行: pred_latents * latents_bn_std + latents_bn_mean

Broadcast 规则:
  bn_mean: (1, 16, 1, 1)
  pred:    (B, 64, H, W)
  
  广播过程:
    (1, 16, 1, 1) -> (1, 16, 1, 1)
    (B, 64, H, W) -> (B, 64, H, W)
    
  维度对齐 (从右到左):
    dim 3: 1 vs W  -> W
    dim 2: 1 vs H  -> H
    dim 1: 16 vs 64 -> ??? 不兼容！16 != 64
    dim 0: 1 vs B   -> B

问题: 16 != 64，broadcast 会失败！

除非...BN 统计量的形状实际上不同？
""")

print("\n" + "=" * 70)
print("【可能的解决方案】")
print("=" * 70)

print("""
情况 1: BN 是针对 16 通道的，代码有 bug
  → 需要在 view 之前正确 reshape 或重复
  → 或者代码应该 view(1, -1, 1, 1) 但实际需要 (1, 16, 1, 1) -> (1, 64, 1, 1)

情况 2: BN 是针对 64 通道的
  → vae.bn 的权重/偏置形状本身就是 (64,)
  → view(1, -1, 1, 1) = (1, 64, 1, 1)，可以正确 broadcast

情况 3: 代码使用了错误的 BN 统计量
  → 可能应该使用不同的 mean/std
""")

print("\n" + "=" * 70)
print("【检查实际 BN 形状】")
print("=" * 70)

print("""
flux2_klein.py line 1249-1252:
    latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1)
    latents_bn_std = torch.sqrt(
        self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
    )
    
需要检查 vae.bn.running_mean 的实际形状是多少！

可能的情况:
  1. 如果 vae.bn 是标准的 BatchNorm1d，running_mean 形状是 (C,)
     - 如果 C = 64，则 view(1, -1, 1, 1) = (1, 64, 1, 1)，可以正确 broadcast
  2. 如果 C = 16，则 view(1, -1, 1, 1) = (1, 16, 1, 1)，broadcast 会失败

让我检查 AutoencoderKLFlux2 的配置...
""")

print("\n" + "=" * 70)
print("【深入分析 Patchify 与 BN 的关系】")
print("=" * 70)

print("""
Flux 的 patchify 过程:
  原始 latent: (B, 16, H, W)
       ↓ patchify
  patched latent: (B, 64, H/2, W/2)
  
  原来每个 2x2 patch 的 4 个像素被展平成通道维度
  
问题: BN 统计量是针对哪个空间的？

选项 A: BN 针对原始 16 通道 latent
  - running_mean 形状 (16,)
  - 对 patched latent (64通道) 应用需要重复 4 次
  - 代码没有重复，直接 broadcast 会失败

选项 B: BN 针对 patched 64 通道 latent
  - running_mean 形状 (64,) 或 (16,) 但有特殊处理
  - 代码直接应用 view(1, -1, 1, 1) 可能正确

Flux 论文中提到 BN 是用来归一化 latent 空间的。
需要查看 AutoencoderKLFlux2 的具体实现...
""")
