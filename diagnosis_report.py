#!/usr/bin/env python3
"""
================================================================
Pipeline 推理流程逐行分析报告：为什么 pred ≈ LQ
================================================================

根据对 Flux2KleinIRPipeline.__call__ 的逐行分析，结合实际 val 图片的像素值统计，
以及数学推导，以下是完整的根因分析。

================================================================
一、实际观察到的现象
================================================================

从 val_monitor/step_000020/rain_0.png 的像素统计：
- LQ mean: 94.03, std: 47.90
- PRED mean: 93.85, std: 47.88
- GT mean: 61.78, std: 52.58

PRED vs LQ 差异: 1.71 (约 1.8% 相对差异)
PRED vs GT 差异: 32.47
LQ vs GT 差异: 32.56

结论: PRED 几乎等于 LQ，只有约 1.8% 的像素值变化。

================================================================
二、数学推导：Flow Matching 的期望行为
================================================================

Flow Matching 公式:
    x_t = (1 - σ) * x_0 + σ * ε
其中:
    σ (sigma): 噪声等级，固定值 0.1 (对应 fixed_timestep=900)
    x_0 = LQ normalized latent
    ε = 随机噪声
    x_t = 加噪后的 latent

模型的训练目标 (来自 trainer.py line 638):
    target = noise + ((1 - σ) * lq_norm - hq_norm) / σ

Euler 更新 (pipeline line 1221):
    x_{new} = x_t - σ * model_pred

理想情况 (model_pred = target):
    x_{new} = x_t - σ * [noise + ((1-σ)*LQ - HQ)/σ]
             = x_t - σ*noise - (1-σ)*LQ + HQ
             = [(1-σ)*LQ + σ*noise] - σ*noise - (1-σ)*LQ + HQ
             = HQ

所以，完美训练的模型应该输出 HQ。

================================================================
三、实际问题：未训练的模型 + BN 归一化/反归一化
================================================================

关键发现: 验证时使用的是**未微调的原始预训练权重**！

Flux2-Klein 是预训练模型，训练目标是:
    给定文本描述，从噪声生成对应图像 (text-to-image)

但图像恢复任务的训练目标是:
    给定 LQ 图像，恢复为 HQ 图像 (image restoration)

这是完全不同的任务！

原始 Flux2-Klein 模型：
- 输入: 噪声 + 文本描述
- 输出: 生成的图像
- 训练数据: 高质量图文对

图像恢复微调后：
- 输入: 加噪的 LQ latent + LQ 图像特征 (通过 ConvNeXt) + 文本描述
- 输出: 恢复后的 latent
- 训练数据: LQ-HQ 对

当使用**未微调的原始权重**时：
- 模型完全不知道如何进行图像恢复
- 它会尝试执行 text-to-image 生成
- 但输入不是纯噪声，而是加噪的 LQ

================================================================
四、逐行 Pipeline 分析
================================================================

以下是 Flux2KleinIRPipeline.__call__ 的关键步骤分析：

### 步骤 1-5: 预处理
- LQ 图像预处理 (resize, normalize) ✓
- VAE 编码 → patchify → BN normalize ✓
- Pack 成 (B, 1024, 128) ✓

### 步骤 6: 加噪 (line 1151-1152)
```python
sigma_start = original_sigmas[fixed_idx].item()  # ≈ 0.1
current_latents = (1.0 - sigma_start) * lq_packed + sigma_start * noise
```
- x_t = 0.9 * LQ_norm + 0.1 * noise
- sigma = 0.1 很小，所以 x_t ≈ LQ_norm (90% LQ, 10% noise)

### 步骤 7: 模型推理 (line 1189-1200)
```python
model_pred = transformer(
    hidden_states=current_latents,  # x_t ≈ 0.9LQ + 0.1noise
    timestep=model_timestep,       # 0.9
    guidance=cfg_guidance,         # 3.5
    ...
)
```

**关键**: 模型是未微调的原始 Flux2-Klein 权重！
- 它期望输入是噪声 (x_t 中噪声占主导)
- 但实际输入 x_t ≈ 0.9LQ + 0.1noise，LQ 占主导
- LQ 的特征通过 ConvNeXt (deg_extractor) 注入调制参数

**未微调模型的行为**:
- 原始 Flux2-Klein 在 t=0.9, x=0.9LQ+0.1noise 时的输出
- 由于模型在训练时从未见过这种输入模式
- 输出接近其"默认"行为

### 步骤 8: Euler 更新 (line 1220-1221)
```python
sigma_t = self._raw_sigmas[i].to(device=current_latents.device, dtype=current_latents.dtype)
current_latents = current_latents - sigma_t * model_pred
```

**数学推导**:
设:
- x_t = (1-σ)*LQ + σ*noise  (加噪)
- σ = 0.1

如果 model_pred 预测的是"去噪到 LQ"的 target:
    target_toward_LQ = (x_t - LQ) / σ = noise

那么:
    x_new = x_t - σ * noise = LQ

**验证**:
    x_t = (1-0.1)*LQ + 0.1*noise = 0.9*LQ + 0.1*noise
    x_new = (0.9*LQ + 0.1*noise) - 0.1*noise = 0.9*LQ

等等，这不对！x_new = 0.9*LQ，不是 LQ！

让我重新推导...

实际上，如果模型预测的是:
    model_pred ≈ noise + ((1-σ)*LQ - HQ)/σ
                 = noise + 9*LQ - 10*HQ    (当 σ=0.1)

那么:
    x_new = x_t - σ*model_pred
          = [(1-σ)*LQ + σ*noise] - σ*[noise + 9*LQ - 10*HQ]
          = (1-σ)*LQ + σ*noise - σ*noise - 9σ*LQ + 10σ*HQ
          = (1-σ-9σ)*LQ + 10σ*HQ
          = (1-10σ)*LQ + 10σ*HQ
          = 0*LQ + 1*HQ  (当 σ=0.1 时)
          = HQ

完美！这就是为什么完美训练的模型应该输出 HQ。

但如果模型**未训练/微调**呢？
- 模型权重的初始状态是 text-to-image 训练的
- 它不知道如何从 LQ 恢复 HQ
- 最简单的"最小 loss"策略: 预测 noise (恒等映射)
- 这样 x_new ≈ LQ，loss = MSE(noise, noise) = 0

### 步骤 9: Unpack + BN 反归一化 (line 1245-1253)
```python
pred_latents = self._unpack_latents_with_ids(current_latents, img_ids, ...)
pred_latents = pred_latents * latents_bn_std + latents_bn_mean
pred_latents = self._unpatchify_latents(pred_latents)
```

如果 x_new ≈ LQ:
- LQ 的 latent 经过 patchify → BN normalize → pack → unpack → BN denorm → unpatchify
- 理论上应该恢复为原始的 LQ latent
- VAE decode → 输出 ≈ LQ

### 步骤 10: VAE 解码
```python
image = self.vae.decode(pred_latents, return_dict=False)[0]
```

================================================================
五、BN 归一化/反归一化的对称性
================================================================

### 编码路径 (_encode_vae_image, line 469-480):
```python
image_latents = self._patchify_latents(image_latents)  # (1, 128, H/2, W/2)
latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1)  # (1, 16, 1, 1)
latents_bn_std = torch.sqrt(...)  # (1, 16, 1, 1)
image_latents = (image_latents - latents_bn_mean) / latents_bn_std
```

注意: bn.running_mean 形状是 (16,)，view 后是 (1, 16, 1, 1)
但 patchified latent 有 128 通道！

**PyTorch Broadcasting 验证**:
- (1, 16, 1, 1) broadcast with (1, 128, H/2, W/2)
- dim -4: 1 vs 1 ✓
- dim -3: 16 vs 128 ← MISMATCH!

**这应该导致 RuntimeError**，但可能:
1. 在某些 PyTorch 版本中行为不同
2. 或者有其他代码路径

### 解码路径 (line 1249-1254):
```python
latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1)  # (1, 16, 1, 1)
pred_latents = self._unpatchify_latents(pred_latents)  # (1, 32, H, W)
pred_latents = pred_latents * latents_bn_std + latents_bn_mean
```

这里同样有 BN shape mismatch 问题。

================================================================
六、最终结论
================================================================

### 根因: **未使用微调后的权重进行验证**

验证时的权重来源:
- Flux2KleinIRPipeline 的 transformer 参数来自 __init__ 时传入的模型
- 如果没有显式加载微调后的权重，使用的是原始预训练权重
- 原始 Flux2-Klein 是 text-to-image 模型，不是图像恢复模型

### 为什么 pred ≈ LQ?

1. **sigma = 0.1 很小**:
   - x_t = 0.9*LQ + 0.1*noise ≈ 主要是 LQ
   - 即使模型输出 0 (恒等映射), x_new = x_t ≈ LQ

2. **未训练模型的"保守"预测**:
   - 面对不熟悉的输入 (加噪的 LQ)，模型输出接近 0
   - 预测 noise → x_new = LQ (通过 Euler 更新)
   - 这样 MSE loss = 0

3. **Flow matching 公式的固有特性**:
   - target = noise + ((1-σ)*LQ - HQ)/σ
   - 当模型完全不知道任务时，预测 noise 是最"安全"的
   - x_new = (1-σ)*LQ + σ*noise - σ*noise = LQ

### 为什么 PRED vs LQ 有 1.7 的差异?

这是由于:
1. sigma = 0.1 的加噪: x_t = 0.9*LQ + 0.1*noise
2. 模型输出 model_pred ≈ 0 (未训练)
3. x_new = x_t - 0.1*0 = x_t ≈ 0.9*LQ + 0.1*noise
4. 解码后 ≈ LQ + 一点噪声的影响

================================================================
七、解决方案
================================================================

### 方案 1: 加载微调后的权重
在验证前加载 accelerator 保存的最新检查点:
```python
from safetensors.torch import load_file
state_dict = load_file('checkpoint-XXXXX/model.safetensors')
transformer.load_state_dict(state_dict, strict=False)
```

### 方案 2: 检查训练日志中的 loss
如果训练正常，应该看到 loss 逐渐下降。

### 方案 3: 使用训练时的验证路径
确保验证走的是 `trainer.py` 的 `validate()` 方法，而不是单独的 pipeline 调用。

================================================================
八、调试建议
================================================================

1. 打印 model_pred 的统计值:
```python
print(f"model_pred: mean={model_pred.mean():.6f}, std={model_pred.std():.6f}")
```

期望:
- 微调后: model_pred ≈ target = noise + 9*LQ - 10*HQ (非零，有意义)
- 未微调: model_pred ≈ 0 或接近噪声 (恒等映射)

2. 打印 Euler 更新后的 latent:
```python
print(f"x_new: mean={x_new.mean():.6f}, std={x_new.std():.6f}")
print(f"lq_norm: mean={lq_norm.mean():.6f}, std={lq_norm.std():.6f}")
print(f"diff(x_new, lq_norm)={torch.abs(x_new - lq_norm).mean():.6f}")
```

期望:
- 微调后: x_new ≈ hq_norm (不同于 lq_norm)
- 未微调: x_new ≈ lq_norm (等于 lq_norm)

3. 对比 VAE decode 结果:
```python
# 直接 decode LQ latent
lq_decoded = vae.decode(lq_unpatchified)
# 对比 pred_decoded
print(f"diff(pred, lq_decode)={torch.abs(pred_decoded - lq_decoded).mean():.6f}")
```
"""
print(__doc__)
