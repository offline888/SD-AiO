#!/usr/bin/env python3
"""
调试脚本：追踪 Pipeline 推理每一步的 latent 值，找出为什么 pred ≈ LQ
"""
import sys
import os
sys.path.insert(0, '/home/yhmi/All_in_one/src')

import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

torch.manual_seed(42)
device = 'cuda:0'

# ============================================================
# 1. 加载模型
# ============================================================
print("=" * 70)
print("STEP 1: 加载模型")
print("=" * 70)

from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
from src.flux2 import Flux2Transformer2DModel
from transformers import T5Tokenizer, T5EncoderModel
from src.flux2.pipelines.flux2_klein import Flux2KleinIRPipeline

MODEL_PATH = '/home/yhmi/data/model/flux.2-klein'
DTYPE = torch.bfloat16

# VAE
vae = AutoencoderKLFlux2.from_pretrained(MODEL_PATH, subfolder='vae', torch_dtype=DTYPE)
vae.to(device).eval()
print(f"VAE loaded. config.block_out_channels={vae.config.block_out_channels}")
print(f"VAE bn.running_mean: shape={vae.bn.running_mean.shape}, mean={vae.bn.running_mean.mean().item():.4f}")
print(f"VAE bn.running_var: shape={vae.bn.running_var.shape}, mean={vae.bn.running_var.mean().item():.4f}")

# Transformer
transformer = Flux2Transformer2DModel.from_pretrained(MODEL_PATH, subfolder='transformer', torch_dtype=DTYPE)
transformer.to(device).eval()
print(f"Transformer loaded. in_channels={transformer.config.in_channels}, out_channels={transformer.out_channels}")

# Scheduler
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_PATH, subfolder='scheduler')
scheduler.to(device)

# Text encoder
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH, subfolder='tokenizer')
text_encoder = T5EncoderModel.from_pretrained(MODEL_PATH, subfolder='text_encoder')
text_encoder.to(device).eval()

# ============================================================
# 2. 加载测试图片
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: 加载测试图片")
print("=" * 70)

LQ_DIR = "/home/yhmi/data/patches/10Rain/LQ_train"
GT_DIR = "/home/yhmi/data/patches/10Rain/GT"
lq_files = sorted([f for f in os.listdir(LQ_DIR) if f.endswith(('.jpg', '.png', '.JPG', '.PNG'))])
lq_path = os.path.join(LQ_DIR, lq_files[0])
gt_path = os.path.join(GT_DIR, lq_files[0])

lq_pil = Image.open(lq_path).convert('RGB').resize((512, 512))
gt_pil = Image.open(gt_path).convert('RGB').resize((512, 512))

# ImageNet normalization (same as pipeline)
IMGNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMGNET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

lq_t = TF.to_tensor(lq_pil).sub_(IMGNET_MEAN).div_(IMGNET_STD).unsqueeze(0).to(device, DTYPE)
gt_t = TF.to_tensor(gt_pil).sub_(IMGNET_MEAN).div_(IMGNET_STD).unsqueeze(0).to(device, DTYPE)

lq_np = np.array(lq_pil) / 255.0
gt_np = np.array(gt_pil) / 255.0

print(f"lq_pil size: {lq_pil.size}, lq_t shape: {lq_t.shape}")
print(f"GT size: {gt_pil.size}, gt_t shape: {gt_t.shape}")

# ============================================================
# 3. 手动执行 pipeline 的每个步骤
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: 手动追踪 pipeline 每个步骤")
print("=" * 70)

# ---------- 3a. VAE 编码 (与 pipeline._encode_vae_image 相同) ----------
def patchify_latents(latents):
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
    return latents

def pack_latents(latents):
    batch_size, num_channels, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)
    return latents

def prepare_latent_ids(latents: torch.Tensor):
    batch_size, _, height, width = latents.shape
    t = torch.arange(1)
    h = torch.arange(height)
    w = torch.arange(width)
    l = torch.arange(1)
    latent_ids = torch.cartesian_prod(t, h, w, l)
    latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)
    return latent_ids

def unpack_latents_with_ids(x, x_ids, height=None, width=None):
    x_list = []
    for data, pos in zip(x, x_ids):
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)
        h = height if height is not None else torch.max(h_ids) + 1
        w = width if width is not None else torch.max(w_ids) + 1
        flat_ids = h_ids * w + w_ids
        _, ch = data.shape
        out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)
        out = out.view(h, w, ch).permute(2, 0, 1)
        x_list.append(out)
    return torch.stack(x_list, dim=0)

def unpatchify_latents(latents):
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), height * 2, width * 2)
    return latents

# Encode LQ
with torch.no_grad():
    lq_encoded = vae.encode(lq_t)
    lq_latent_raw = lq_encoded.latent_dist.mode()
    print(f"VAE encode output latent: {lq_latent_raw.shape}")  # (1, 32, 64, 64)

    gt_encoded = vae.encode(gt_t)
    gt_latent_raw = gt_encoded.latent_dist.mode()
    print(f"GT VAE encode: {gt_latent_raw.shape}")

# Patchify
lq_patchified = patchify_latents(lq_latent_raw)
gt_patchified = patchify_latents(gt_latent_raw)
print(f"After patchify: LQ={lq_patchified.shape}, GT={gt_patchified.shape}")  # (1, 128, 32, 32)

# BN normalize
bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device)
bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(device)
lq_norm = (lq_patchified - bn_mean) / bn_std
gt_norm = (gt_patchified - bn_mean) / bn_std

print(f"\nBN stats: mean in [{bn_mean.min().item():.4f}, {bn_mean.max().item():.4f}]")
print(f"BN stats: std in [{bn_std.min().item():.4f}, {bn_std.max().item():.4f}]")
print(f"LQ patchified: mean={lq_patchified.mean().item():.4f}, std={lq_patchified.std().item():.4f}")
print(f"LQ BN-normalized: mean={lq_norm.mean().item():.4f}, std={lq_norm.std().item():.4f}")
print(f"GT BN-normalized: mean={gt_norm.mean().item():.4f}, std={gt_norm.std().item():.4f}")

# ---------- 3b. Pack ----------
lq_packed = pack_latents(lq_norm)
gt_packed = pack_latents(gt_norm)
print(f"\nAfter pack: LQ={lq_packed.shape}, GT={gt_packed.shape}")  # (1, 1024, 128)

# ---------- 3c. 准备 timestep 和噪声 ----------
fixed_timestep = 900
num_inference_steps = 1

# 保存原始 sigmas
scheduler._original_sigmas = scheduler.sigmas.clone()
original_sigmas = scheduler._original_sigmas
sigma_start = original_sigmas[fixed_timestep].item()
model_timestep = fixed_timestep / 1000.0

print(f"\nfixed_timestep={fixed_timestep}, sigma_start={sigma_start:.6f}")
print(f"model_timestep={model_timestep}")

# 加噪
torch.manual_seed(42)
noise = torch.randn_like(lq_packed)
current_latents = (1.0 - sigma_start) * lq_packed + sigma_start * noise

print(f"\n加噪后 current_latents: mean={current_latents.mean().item():.4f}, std={current_latents.std().item():.4f}")
print(f"lq_packed: mean={lq_packed.mean().item():.4f}, std={lq_packed.std().item():.4f}")
print(f"noise: mean={noise.mean().item():.4f}, std={noise.std().item():.4f}")

# ---------- 3d. 准备 text embeds ----------
prompt = "remove rain from this image"
inputs = tokenizer(prompt, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
prompt_embeds = text_encoder(
    input_ids=inputs.input_ids.to(device),
    attention_mask=inputs.attention_mask.to(device)
).last_hidden_state.to(DTYPE)
print(f"\nprompt_embeds shape: {prompt_embeds.shape}")  # (1, 120, 4096)

# Text IDs
def prepare_text_ids(x):
    B, L, _ = x.shape
    out_ids = []
    for i in range(B):
        t = torch.arange(1)
        h = torch.arange(1)
        w = torch.arange(1)
        l = torch.arange(L)
        coords = torch.cartesian_prod(t, h, w, l)
        out_ids.append(coords)
    return torch.stack(out_ids)

text_ids = prepare_text_ids(prompt_embeds).to(device)
print(f"text_ids shape: {text_ids.shape}")  # (1, 120, 4)

# ---------- 3e. 准备 img_ids (latent position IDs) ----------
lq_packed_B, seq_len, C = lq_packed.shape
lq_reshaped_for_ids = lq_patchified  # (1, 128, 32, 32)
img_ids = prepare_latent_ids(lq_reshaped_for_ids).to(device)
print(f"img_ids shape: {img_ids.shape}")  # (1, 1024, 4)

# ---------- 3f. 准备 deg_emb (degradation embedding) ----------
# 用 conv stem 的输出作为 deg_emb (与 modulation 相同的处理)
deg_extractor = transformer.double_stream_modulation_img.conv_stem_s1
deg_extractor.eval()

with torch.no_grad():
    lq_for_deg = lq_t.to(DTYPE)
    deg_feat = deg_extractor(lq_for_deg)
    deg_feat = transformer.double_stream_modulation_img.conv_time_mod1(deg_feat, torch.zeros(1, 6144, device=device, dtype=DTYPE))
    deg_feat = transformer.double_stream_modulation_img.conv_down1_s2(deg_feat)
    deg_feat = transformer.double_stream_modulation_img.conv_time_mod2(deg_feat, torch.zeros(1, 6144, device=device, dtype=DTYPE))
    deg_feat = transformer.double_stream_modulation_img.conv_down2_s3(deg_feat)
    deg_feat = transformer.double_stream_modulation_img.conv_time_mod3(deg_feat, torch.zeros(1, 6144, device=device, dtype=DTYPE))
    deg_emb_raw = transformer.double_stream_modulation_img.feat_proj(deg_feat)
    deg_emb = deg_emb_raw.permute(0, 2, 3, 1).reshape(1, -1, 3 * transformer.double_stream_modulation_img.dim)
    deg_emb = deg_emb.mean(dim=(1, 2), keepdim=True).permute(0, 2, 1)  # (1, 1, 768)
    print(f"deg_emb shape: {deg_emb.shape}")

# ---------- 3g. 设置 scheduler ----------
deno_sigmas = np.array([sigma_start])
scheduler.set_timesteps(
    num_inference_steps=num_inference_steps,
    device=device,
    sigmas=deno_sigmas,
    mu=2.052,  # approximate
)
scheduler.set_begin_index(0)
timesteps = scheduler.timesteps
raw_sigmas = torch.from_numpy(deno_sigmas).to(dtype=torch.float32, device=device)

print(f"\nScheduler timesteps: {timesteps}")
print(f"Scheduler sigmas: {scheduler.sigmas}")
print(f"raw_sigmas: {raw_sigmas}")

# ---------- 3h. 合并 text_ids + deg_emb token ----------
deg_txt_id = text_ids[:, :1, :].clone()
text_ids_with_deg = torch.cat([deg_txt_id, text_ids], dim=1)
print(f"\ntext_ids_with_deg shape: {text_ids_with_deg.shape}")  # (1, 121, 4)

# ---------- 3i. 模型推理 ----------
guidance = torch.tensor([3.5], device=device, dtype=DTYPE)
timestep_tensor = torch.full((1,), model_timestep, device=device, dtype=DTYPE)

print(f"\n模型输入:")
print(f"  current_latents: {current_latents.shape}, dtype={current_latents.dtype}")
print(f"  timestep: {timestep_tensor}")
print(f"  guidance: {guidance}")
print(f"  prompt_embeds: {prompt_embeds.shape}")
print(f"  text_ids_with_deg: {text_ids_with_deg.shape}")
print(f"  img_ids: {img_ids.shape}")
print(f"  deg_emb: {deg_emb.shape}")

# 检查 lq_tensor 的格式
lq_tensor_cat = lq_t  # (1, 3, 512, 512)

with torch.no_grad():
    model_pred = transformer(
        hidden_states=current_latents,
        timestep=timestep_tensor,
        guidance=guidance,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids_with_deg,
        img_ids=img_ids,
        deg_emb=deg_emb,
        lq_tensor=lq_tensor_cat,
        return_dict=False,
    )[0]

print(f"\n模型输出 model_pred: {model_pred.shape}")
print(f"model_pred: mean={model_pred.mean().item():.6f}, std={model_pred.std().item():.6f}")
print(f"model_pred: min={model_pred.min().item():.6f}, max={model_pred.max().item():.6f}")
print(f"model_pred vs 0: abs_diff={(model_pred - 0).abs().mean().item():.6f}")
print(f"model_pred vs noise (packed): abs_diff={(model_pred - noise).abs().mean().item():.6f}")

# ---------- 3j. Euler 更新 ----------
sigma_t = raw_sigmas[0].to(current_latents.dtype)
print(f"\nEuler 更新: sigma_t={sigma_t.item():.6f}")
latents_after_euler = current_latents - sigma_t * model_pred

print(f"去噪后: mean={latents_after_euler.mean().item():.4f}, std={latents_after_euler.std().item():.4f}")
print(f"lq_packed: mean={lq_packed.mean().item():.4f}, std={lq_packed.std().item():.4f}")

# 关键：去噪后与 LQ 的差异
diff_to_lq = (latents_after_euler - lq_packed).abs().mean().item()
diff_current_lq = (current_latents - lq_packed).abs().mean().item()
print(f"\ndiff(去噪后, LQ) = {diff_to_lq:.6f}")
print(f"diff(加噪后, LQ) = {diff_current_lq:.6f}")
print(f"model_pred 贡献: sigma_t * model_pred mean = {(sigma_t * model_pred).abs().mean().item():.6f}")

# ---------- 3k. 检查 model_pred 的实际含义 ----------
# 如果 flow matching target = (x_GT - x_LQ) / sigma, 而 model 预测 target
# 那么 x_{t-1} = x_t - sigma * target = x_t - (x_t - x_GT) = x_GT
# 
# 但如果 model_pred ≈ (x_t - x_LQ) / sigma（即预测的是"去噪到 LQ"的 target）
# 那么 x_{t-1} = x_t - sigma * ((x_t - x_LQ)/sigma) = x_LQ

print("\n" + "=" * 70)
print("分析: model_pred 预测的是什么?")
print("=" * 70)

# 期望的 flow matching target: (x_t - x_GT) / sigma
flow_target = (current_latents - lq_packed) / sigma_t  # 注意：这里用 lq_packed 作为 GT
print(f"flow target (toward LQ): mean={flow_target.mean().item():.4f}, std={flow_target.std().item():.4f}")
print(f"model_pred: mean={model_pred.mean().item():.4f}, std={model_pred.std().item():.4f}")
print(f"model_pred vs flow_target: {(model_pred - flow_target).abs().mean().item():.6f}")

# 如果模型完美预测了 target_to_LQ
perfect_denoise_to_lq = current_latents - sigma_t * flow_target
print(f"\n如果 model_pred = (x_t - x_LQ)/sigma, 去噪后 = {perfect_denoise_to_lq.mean():.4f} (应该等于 LQ)")
print(f"diff(完美去噪, LQ) = {(perfect_denoise_to_lq - lq_packed).abs().mean().item():.10f}")

# ---------- 3l. 解码流程 ----------
print("\n" + "=" * 70)
print("STEP 4: 解码流程 (unpack -> denorm -> unpatchify -> VAE decode)")
print("=" * 70)

pred_latents_after_euler = latents_after_euler  # (1, 1024, 128) packed

# Unpack
latent_height = 32
latent_width = 32
pred_unpacked = unpack_latents_with_ids(
    pred_latents_after_euler, img_ids, latent_height // 2, latent_width // 2
)
print(f"Unpacked: {pred_unpacked.shape}")  # (1, 128, 16, 16)

# BN denormalize
bn_mean_flat = vae.bn.running_mean.view(1, -1, 1, 1).to(pred_unpacked.device, pred_unpacked.dtype)
bn_std_flat = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
    pred_unpacked.device, pred_unpacked.dtype
)
pred_denorm = pred_unpacked * bn_std_flat + bn_mean_flat
print(f"After BN denorm: {pred_denorm.shape}")

# Unpatchify
pred_unpatchified = unpatchify_latents(pred_denorm)
print(f"After unpatchify: {pred_unpatchified.shape}")  # (1, 32, 32, 32)

# VAE decode
with torch.no_grad():
    decoded = vae.decode(pred_unpatchified.to(DTYPE), return_dict=False)[0]
    decoded_np = decoded[0].permute(1, 2, 0).cpu().float().numpy()
    # 反 ImageNet normalize
    decoded_np = decoded_np * np.array(IMGNET_STD.view(3).tolist()) + np.array(IMGNET_MEAN.view(3).tolist())
    decoded_np = np.clip(decoded_np, 0, 1)

print(f"\ndecoded shape: {decoded_np.shape}, mean={decoded_np.mean():.4f}")

# 对比
diff_pred_lq = np.abs(decoded_np - lq_np).mean()
diff_pred_gt = np.abs(decoded_np - gt_np).mean()
print(f"pred vs LQ: mean_diff={diff_pred_lq:.6f}")
print(f"pred vs GT: mean_diff={diff_pred_gt:.6f}")

# ---------- 3m. 用原始预训练权重 (无微调) 的情况 ----------
print("\n" + "=" * 70)
print("STEP 5: 分析 - 如果加载了微调后的权重会怎样?")
print("=" * 70)

# 查找微调后的权重
checkpoint_dir = '/home/yhmi/data/output/flux2_convnext_ft_3'
import os
model_files = []
for root, dirs, files in os.walk(checkpoint_dir):
    for f in files:
        if f.endswith('.safetensors') or f.endswith('.bin'):
            model_files.append(os.path.join(root, f))

print(f"Found {len(model_files)} model files in {checkpoint_dir}")
for mf in sorted(model_files)[:10]:
    print(f"  {mf}")

# ============================================================
# 6. 完整的端到端 Pipeline 测试
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: 完整 Pipeline 端到端测试")
print("=" * 70)

# 构造 pipeline (不使用 deg_extractor 的情况)
pipeline = Flux2KleinIRPipeline(
    vae=vae,
    transformer=transformer,
    scheduler=scheduler,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    deg_extractor=None,
)
pipeline.to(device, DTYPE)

# 使用原始权重推理
with torch.no_grad():
    result = pipeline(
        lq_image=lq_pil,
        prompt_embeds=prompt_embeds,
        num_inference_steps=1,
        guidance_scale=3.5,
        fixed_timestep=900,
        output_type="np",
    )
    pred_img = result.images[0]

print(f"\nPipeline 输出:")
print(f"  pred mean={pred_img.mean():.4f}, LQ mean={lq_np.mean():.4f}, GT mean={gt_np.mean():.4f}")
print(f"  pred vs LQ: {np.abs(pred_img - lq_np).mean():.6f}")
print(f"  pred vs GT: {np.abs(pred_img - gt_np).mean():.6f}")

# 保存对比图
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(lq_np)
axes[0].set_title(f'LQ\nmean={lq_np.mean():.3f}')
axes[0].axis('off')
axes[1].imshow(pred_img)
axes[1].set_title(f'PRED\nmean={pred_img.mean():.3f}')
axes[1].axis('off')
axes[2].imshow(gt_np)
axes[2].set_title(f'GT\nmean={gt_np.mean():.3f}')
axes[2].axis('off')
plt.tight_layout()
plt.savefig('/home/yhmi/debug_pipeline_comparison.png', dpi=100, bbox_inches='tight')
print(f"\n对比图已保存到: /home/yhmi/debug_pipeline_comparison.png")

# ============================================================
# 7. 关键诊断
# ============================================================
print("\n" + "=" * 70)
print("DIAGNOSIS SUMMARY")
print("=" * 70)

print(f"""
1. model_pred 统计:
   - mean={model_pred.mean().item():.6f}
   - std={model_pred.std().item():.6f}
   - abs_mean={model_pred.abs().mean().item():.6f}
   - vs zero: {(model_pred - 0).abs().mean().item():.6f}

2. 如果 model_pred ≈ 0:
   - Euler 步: x_new = x_t - sigma * 0 = x_t ≈ LQ (因为 sigma 很小)
   - 解码后 ≈ LQ

3. 如果 model_pred ≈ (x_t - x_LQ)/sigma:
   - Euler 步: x_new = x_t - sigma * ((x_t - x_LQ)/sigma) = x_LQ
   - 解码后 = LQ

4. 当前 model_pred 与 flow target toward LQ 的差异:
   {(model_pred - (current_latents - lq_packed) / sigma_t).abs().mean().item():.6f}

5. 结论: model_pred 接近 {((current_latents - lq_packed) / sigma_t).abs().mean().item():.4f} 而非 0
   意味着模型正在预测"去噪到 LQ"而非"去噪到 GT"
""")

# 保存详细的 trace 数据
import json
trace_data = {
    "model_pred_mean": float(model_pred.mean().item()),
    "model_pred_std": float(model_pred.std().item()),
    "model_pred_abs_mean": float(model_pred.abs().mean().item()),
    "lq_packed_mean": float(lq_packed.mean().item()),
    "lq_packed_std": float(lq_packed.std().item()),
    "current_latents_mean": float(current_latents.mean().item()),
    "latents_after_euler_mean": float(latents_after_euler.mean().item()),
    "sigma_start": float(sigma_start),
    "diff_after_euler_to_lq": float(diff_to_lq),
    "pred_vs_lq_np": float(diff_pred_lq),
    "pred_vs_gt_np": float(diff_pred_gt),
}

with open('/home/yhmi/debug_trace.json', 'w') as f:
    json.dump(trace_data, f, indent=2)
print(f"Trace data saved to: /home/yhmi/debug_trace.json")
