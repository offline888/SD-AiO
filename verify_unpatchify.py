#!/usr/bin/env python3
"""
比较两种 unpatchify 实现
"""
import torch

device = 'cuda:0'

def unpatchify_latents_v1(latents):
    """我的版本"""
    batch_size, num_channels, height, width = latents.shape
    channels = num_channels // 4
    latents = latents.view(batch_size, channels, 4, height, width)
    latents = latents.permute(0, 1, 3, 2, 4)
    latents = latents.reshape(batch_size, channels, height * 2, width * 2)
    return latents

def unpatchify_latents_v2(latents):
    """Pipeline 的版本"""
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), height * 2, width * 2)
    return latents

# 测试
torch.manual_seed(42)
test = torch.randn(1, 128, 64, 64, device=device)

result_v1 = unpatchify_latents_v1(test)
result_v2 = unpatchify_latents_v2(test)

print(f"Input: {test.shape}")
print(f"V1 (mine): {result_v1.shape}")
print(f"V2 (pipeline): {result_v2.shape}")

print(f"\n两个版本的差异:")
print(f"  Mean diff: {(result_v1 - result_v2).abs().mean().item():.8f}")
print(f"  Max diff: {(result_v1 - result_v2).abs().max().item():.8f}")
print(f"  形状相同: {result_v1.shape == result_v2.shape}")

# 检查具体值的差异
print(f"\nV1 mean: {result_v1.mean().item():.6f}")
print(f"V2 mean: {result_v2.mean().item():.6f}")

# 验证 patchify-unpatchify 的循环一致性
def patchify_latents(latents):
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
    return latents

# 原始
orig = torch.randn(1, 32, 128, 128, device=device)
print(f"\n原始: {orig.shape}")

# Patchify
patched = patchify_latents(orig)
print(f"Patchified: {patched.shape}")

# Unpatchify V1
unpatched_v1 = unpatchify_latents_v1(patched)
print(f"Unpatchified V1: {unpatched_v1.shape}")
print(f"V1 循环误差: {(orig - unpatched_v1).abs().mean().item():.8f}")

# Unpatchify V2
unpatched_v2 = unpatchify_latents_v2(patched)
print(f"Unpatchified V2: {unpatched_v2.shape}")
print(f"V2 循环误差: {(orig - unpatched_v2).abs().mean().item():.8f}")

# 检查具体位置的差异
print(f"\n特定位置检查 (orig[0,0,0,0]): {orig[0,0,0,0].item():.6f}")
print(f"V1[0,0,0,0]: {unpatched_v1[0,0,0,0].item():.6f}")
print(f"V2[0,0,0,0]: {unpatched_v2[0,0,0,0].item():.6f}")
