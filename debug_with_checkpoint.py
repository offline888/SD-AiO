#!/usr/bin/env python3
"""
Debug script: properly replicate the inference pipeline to find pred ≈ LQ root cause.
"""
import sys, os
sys.path.insert(0, '/home/yhmi/All_in_one/src')

import torch
import numpy as np

torch.manual_seed(42)
device = 'cuda:0'

print("=" * 80)
print("STEP 1: Build inference pipeline (matching src/inference.py)")
print("=" * 80)

from src.flux2.pipelines.flux2_klein import Flux2KleinIRPipeline
from src.models import FLUX2ModulationV2, DegFeatExtractor
import argparse

pipe = Flux2KleinIRPipeline.from_pretrained(
    '/home/yhmi/data/model/flux.2-klein',
    torch_dtype=torch.bfloat16,
)

pipe.vae = pipe.vae.to(dtype=torch.bfloat16)
pipe.transformer = pipe.transformer.to(dtype=torch.bfloat16)

print(f"Original double_stream_modulation_img type: {type(pipe.transformer.double_stream_modulation_img)}")
print(f"Has conv_stem_s1: {hasattr(pipe.transformer.double_stream_modulation_img, 'conv_stem_s1')}")

orig = pipe.transformer.double_stream_modulation_img
if not isinstance(orig, FLUX2ModulationV2) or not hasattr(orig, 'conv_stem_s1'):
    print("Replacing modulation with ConvNeXt version...")
    new_mod = FLUX2ModulationV2(
        dim=orig.linear.in_features,
        mod_param_sets=2,
        bias=orig.linear.bias is not None,
        use_block_emb=True,
        use_conv=True,
        use_vae=False,
        vae_path='/home/yhmi/data/model/flux.2-klein',
    ).to(device='cpu', dtype=torch.bfloat16)
    new_mod.linear.load_state_dict(orig.linear.state_dict())
    pipe.transformer.double_stream_modulation_img = new_mod

    orig_single = pipe.transformer.single_stream_modulation
    new_mod_single = FLUX2ModulationV2(
        dim=orig_single.linear.in_features,
        mod_param_sets=1,
        bias=orig_single.linear.bias is not None,
        use_block_emb=True,
        use_conv=True,
        use_vae=False,
        vae_path='/home/yhmi/data/model/flux.2-klein',
    ).to(device='cpu', dtype=torch.bfloat16)
    new_mod_single.linear.load_state_dict(orig_single.linear.state_dict())
    pipe.transformer.single_stream_modulation = new_mod_single
    print("Replaced both double and single stream modulation")
else:
    print("Already has ConvNeXt modulation, skipping replacement")

ckpt_path = '/home/yhmi/data/output/flux2_convnext_ft_3/checkpoint-2000/pytorch_model/mp_rank_00_model_states.pt'
print(f"\nLoading checkpoint from: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
raw_state = ckpt['module']

loaded_count = 0
for k, v in raw_state.items():
    k_clean = k
    if k_clean.startswith('module.'):
        k_clean = k_clean[7:]
    if k_clean in pipe.transformer.state_dict():
        pipe.transformer.state_dict()[k_clean].copy_(v)
        loaded_count += 1

print(f"Loaded {loaded_count} tensors into transformer")

mod_img = pipe.transformer.double_stream_modulation_img
for name, param in sorted(mod_img.named_parameters(), key=lambda x: x[0]):
    if 'conv_stem' in name or 'block_embed' in name:
        print(f"  Loaded: {name} mean={param.mean().item():.6f}, std={param.std().item():.6f}")
        break

pipe.deg_extractor = DegFeatExtractor(
    inner_dim=pipe.transformer.inner_dim,
    num_deg_types=5,
    weight_dtype=torch.bfloat16,
    args=argparse.Namespace(
        degradation_classifier_path=None,
        dino_type='vits14',
    ),
    deg_embedding=None,
)
pipe.transformer.register_parameter("deg_embedding", pipe.deg_extractor.deg_embedding)

if 'module.deg_embedding' in raw_state:
    pipe.transformer.state_dict()['deg_embedding'].copy_(raw_state['module.deg_embedding'])
    print(f"Loaded deg_embedding: mean={pipe.transformer.state_dict()['deg_embedding'].mean().item():.6f}")

pipe.to(device, torch.bfloat16)
pipe.eval()
print(f"\nPipeline ready on {device}")

print("\n" + "=" * 80)
print("STEP 2: Test inference")
print("=" * 80)

from PIL import Image
import torchvision.transforms.functional as TF

lq_dir = "/home/yhmi/data/patches/10Rain/LQ_train"
lq_files = sorted([f for f in os.listdir(lq_dir) if f.endswith(('.jpg', '.png', '.JPG', '.PNG'))])
lq_path = os.path.join(lq_dir, lq_files[0])
gt_path = os.path.join("/home/yhmi/data/patches/10Rain/GT", lq_files[0])

lq_pil = Image.open(lq_path).convert('RGB').resize((512, 512))
gt_pil = Image.open(gt_path).convert('RGB').resize((512, 512))

with torch.no_grad():
    result = pipe(
        lq_image=lq_pil,
        prompt="remove rain from this image",
        num_inference_steps=1,
        guidance_scale=3.5,
        fixed_timestep=900,
        output_type="np",
    )
    pred_img = result.images[0]

lq_np = np.array(lq_pil) / 255.0
gt_np = np.array(gt_pil) / 255.0
diff_lq = np.abs(pred_img - lq_np).mean()
diff_gt = np.abs(pred_img - gt_np).mean()
print(f"\nResult:")
print(f"  pred vs LQ: mean_diff={diff_lq:.6f}")
print(f"  pred vs GT: mean_diff={diff_gt:.6f}")
print(f"  pred mean={pred_img.mean():.4f}, LQ mean={lq_np.mean():.4f}, GT mean={gt_np.mean():.4f}")

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(lq_np)
axes[0].set_title(f'LQ (mean={lq_np.mean():.3f})')
axes[1].imshow(pred_img)
axes[1].set_title(f'PRED (mean={pred_img.mean():.3f})')
axes[2].imshow(gt_np)
axes[2].set_title(f'GT (mean={gt_np.mean():.3f})')
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.savefig('/home/yhmi/debug_with_checkpoint.png', dpi=100)
print("\nSaved to /home/yhmi/debug_with_checkpoint.png")

print("\n" + "=" * 80)
print("STEP 3: Manual trace")
print("=" * 80)

from src.flux2.pipelines.latent_utils import pack_latents, patchify_latents, prepare_latent_ids, unpack_latents_with_ids
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

IMGNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMGNET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

lq_t = TF.to_tensor(lq_pil).unsqueeze(0) / 255.0
lq_t_norm = (lq_t - IMGNET_MEAN) / IMGNET_STD
lq_t_bf16 = lq_t_norm.to(device, torch.bfloat16)

with torch.no_grad():
    lq_latent_raw = pipe.vae.encode(lq_t_bf16).latent_dist.mode()
    gt_t = TF.to_tensor(gt_pil).unsqueeze(0) / 255.0
    gt_t_norm = (gt_t - IMGNET_MEAN) / IMGNET_STD
    gt_t_bf16 = gt_t_norm.to(device, torch.bfloat16)
    gt_latent_raw = pipe.vae.encode(gt_t_bf16).latent_dist.mode()

bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(device)
bn_std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(device)

lq_patchified = patchify_latents(lq_latent_raw)
lq_norm = (lq_patchified - bn_mean) / bn_std
lq_packed = pack_latents(lq_norm)

gt_patchified = patchify_latents(gt_latent_raw)
gt_norm = (gt_patchified - bn_mean) / bn_std

print(f"LQ packed: mean={lq_packed.mean().item():.6f}, std={lq_packed.std().item():.6f}")
print(f"GT normalized: mean={gt_norm.mean().item():.6f}, std={gt_norm.std().item():.6f}")

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained('/home/yhmi/data/model/flux.2-klein', subfolder='scheduler')
original_sigmas = scheduler.sigmas.clone()
fixed_idx = 900
sigma_start = original_sigmas[fixed_idx].item()
model_timestep = fixed_idx / 1000.0
print(f"\nfixed_idx={fixed_idx}, sigma_start={sigma_start:.6f}, model_timestep={model_timestep:.3f}")

torch.manual_seed(42)
noise = torch.randn_like(lq_packed) * 0.3
current_latents = (1.0 - sigma_start) * lq_packed + sigma_start * noise
print(f"Noised: mean={current_latents.mean().item():.6f}, std={current_latents.std().item():.6f}")

deg_emb = pipe.deg_extractor(lq_t_bf16).unsqueeze(1)
print(f"deg_emb: mean={deg_emb.mean().item():.6f}, std={deg_emb.std().item():.6f}")

def prepare_text_ids(x):
    B, L, _ = x.shape
    coords = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(L))
    return coords.unsqueeze(0)

prompt = "remove rain from this image"
prompt_embeds, text_ids = pipe.encode_prompt(
    prompt=prompt, prompt_embeds=None, device=device,
    num_images_per_prompt=1, max_sequence_length=512, text_encoder_out_layers=(9, 18, 27))

img_ids = prepare_latent_ids(lq_patchified).to(device)
deg_txt_id = text_ids[:, :1, :].clone()
text_ids_with_deg = torch.cat([deg_txt_id, text_ids], dim=1)

cfg_guidance = torch.tensor([3.5], device=device, dtype=torch.bfloat16)

deno_sigmas = np.array([sigma_start])
scheduler.set_timesteps(num_inference_steps=1, device=device, sigmas=deno_sigmas, mu=2.052)
raw_sigmas = torch.from_numpy(deno_sigmas).to(dtype=torch.float32, device=device)

print("\nRunning transformer (cond pass)...")
with torch.no_grad():
    model_pred = pipe.transformer(
        hidden_states=current_latents,
        timestep=torch.tensor([model_timestep], device=device, dtype=torch.bfloat16),
        guidance=cfg_guidance,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids_with_deg,
        img_ids=img_ids,
        deg_emb=deg_emb,
        lq_tensor=lq_t_bf16,
        return_dict=False,
    )[0]

print(f"\nModel output (cond):")
print(f"  shape: {model_pred.shape}")
print(f"  mean: {model_pred.mean().item():.8f}")
print(f"  std: {model_pred.std().item():.8f}")
print(f"  abs_mean: {model_pred.abs().mean().item():.8f}")

print("\nRunning transformer (uncond pass)...")
neg_prompt_embeds, neg_text_ids = pipe.encode_prompt(
    prompt="", prompt_embeds=None, device=device,
    num_images_per_prompt=1, max_sequence_length=512, text_encoder_out_layers=(9, 18, 27))
neg_deg_ti = torch.cat([neg_text_ids[:, :1, :].clone(), neg_text_ids], dim=1)

with torch.no_grad():
    neg_model_pred = pipe.transformer(
        hidden_states=current_latents,
        timestep=torch.tensor([model_timestep], device=device, dtype=torch.bfloat16),
        guidance=None,
        encoder_hidden_states=neg_prompt_embeds,
        txt_ids=neg_deg_ti,
        img_ids=img_ids,
        deg_emb=deg_emb,
        lq_tensor=lq_t_bf16,
        return_dict=False,
    )[0]

print(f"Uncond output: mean={neg_model_pred.mean().item():.8f}, std={neg_model_pred.std().item():.8f}")
model_pred = neg_model_pred + 3.5 * (model_pred - neg_model_pred)
print(f"CFG applied: mean={model_pred.mean().item():.8f}, std={model_pred.std().item():.8f}")

sigma_t = raw_sigmas[0].item()
latents_after_euler = current_latents - sigma_t * model_pred
print(f"\nEuler update: sigma_t={sigma_t:.6f}")
print(f"  After Euler: mean={latents_after_euler.mean().item():.8f}")
print(f"  vs LQ_packed: diff={(latents_after_euler - lq_packed).abs().mean().item():.8f}")

latent_h, latent_w = 32, 32
pred_unpacked = unpack_latents_with_ids(latents_after_euler, img_ids, latent_h, latent_w)
pred_denorm = pred_unpacked * bn_std + bn_mean

with torch.no_grad():
    decoded = pipe.vae.decode(pred_denorm.to(dtype=pipe.vae.dtype), return_dict=False)[0]

print(f"\nDecoded image: shape={decoded.shape}")
print(f"  mean={decoded.mean().item():.6f}, std={decoded.std().item():.6f}")

decoded_vis = decoded.squeeze(0).permute(1, 2, 0).cpu().float()
decoded_vis = decoded_vis * IMGNET_STD.view(1, 1, 3) + IMGNET_MEAN.view(1, 1, 3)
decoded_vis = decoded_vis.clamp(0, 1).numpy()

diff_decoded_lq = np.abs(decoded_vis - lq_np).mean()
diff_decoded_gt = np.abs(decoded_vis - gt_np).mean()
print(f"\nDecoded vs LQ: {diff_decoded_lq:.6f}")
print(f"Decoded vs GT: {diff_decoded_gt:.6f}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
