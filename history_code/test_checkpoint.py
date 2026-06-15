import argparse
import gc
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio as psnr_fn
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.dataset import PairedDataset2
from src.networks.degnet import DegNet_DINO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

def _prepare_latent_image_ids(latents: torch.Tensor) -> torch.Tensor:
    batch_size, _, height, width = latents.shape
    device = latents.device

    t = torch.arange(1, device=device)
    h = torch.arange(height // 2, device=device)
    w = torch.arange(width // 2, device=device)
    layer = torch.arange(1, device=device)

    latent_ids = torch.cartesian_prod(t, h, w, layer)
    return latent_ids.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

def _pack_latents(latents: torch.Tensor, batch_size: int, height: int, width: int) -> torch.Tensor:
    num_channels = latents.shape[1]
    latents = latents.view(batch_size, num_channels, height, 2, width, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(batch_size, height * width, num_channels * 4)

def _unpack_latents_with_ids(x: torch.Tensor, x_ids: torch.Tensor, latent_channels: int) -> torch.Tensor:
    batch_size = x.shape[0]

    H_packed = int(torch.max(x_ids[:, :, 1]).item()) + 1
    W_packed = int(torch.max(x_ids[:, :, 2]).item()) + 1

    x = x.view(batch_size, H_packed, W_packed, latent_channels, 2, 2)
    x = x.permute(0, 3, 1, 4, 2, 5)

    return x.reshape(batch_size, latent_channels, H_packed * 2, W_packed * 2)

@torch.no_grad()
def encode_images(pixels: torch.Tensor, vae: nn.Module, weight_dtype: torch.dtype) -> torch.Tensor:
    pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
    return pixel_latents.to(weight_dtype)

def _flux2_transformer_forward():
    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs=None,
        return_dict: bool = True,
        deg_emb: torch.Tensor = None,
    ):
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        num_txt_tokens = encoder_hidden_states.shape[1]
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        temb = self.time_guidance_embed(timestep, guidance)
        if deg_emb is not None:
            temb = temb + deg_emb

        double_stream_mod_img = self.double_stream_modulation_img(temb)
        double_stream_mod_txt = self.double_stream_modulation_txt(temb)
        single_stream_mod = self.single_stream_modulation(temb)

        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        freqs_cos_img, freqs_sin_img = self.pos_embed(img_ids)
        freqs_cos_txt, freqs_sin_txt = self.pos_embed(txt_ids)
        concat_rotary_emb = (
            torch.cat([freqs_cos_txt, freqs_cos_img], dim=0),
            torch.cat([freqs_sin_txt, freqs_sin_img], dim=0),
        )

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_img=double_stream_mod_img,
                temb_mod_txt=double_stream_mod_txt,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for block in self.single_transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=None,
                temb_mod=single_stream_mod,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
        hidden_states = hidden_states[:, num_txt_tokens:, ...]
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    Flux2Transformer2DModel.forward = patched_forward


def _collate_fn(examples):
    return {
        "pixel_values": torch.stack([e["pixel_values"] for e in examples]),
        "conditioning_pixel_values": torch.stack([e["conditioning_pixel_values"] for e in examples]),
        "dataset_indices": torch.tensor([ex["dataset_idx"] for ex in examples], dtype=torch.long),
    }

class ConditionTester:
    def __init__(self, args):
        self.args = args
        self.config = OmegaConf.load(args.config)

        self.num_classes = self.config.network.get("num_classes", 4)

        self.vae = None
        self.transformer = None
        self.noise_scheduler = None
        self.deg_classifier = None
        self.weight_dtype = None
        self.prompt_embed_cache = {}
        self._class_embedding_U_full = None

        self.output_dir = None

    def build_models(self):
        model_path = self.config.network.pretrained_model_name_or_path
        precisions = {"fp16": torch.float16, "bf16": torch.bfloat16}
        self.weight_dtype = precisions.get(
            self.config.accelerator.get("mixed_precision", "bf16"), torch.float32
        )

        # Load VAE
        self.vae = AutoencoderKLFlux2.from_pretrained(
            model_path, subfolder="vae",
            revision=self.config.network.get("revision"),
            variant=self.config.network.get("variant"),
            torch_dtype=torch.float32,
        )
        self.vae.requires_grad_(False)

        # Load Transformer
        self.transformer = Flux2Transformer2DModel.from_pretrained(
            model_path, subfolder="transformer",
            revision=self.config.network.get("revision"),
            variant=self.config.network.get("variant"),
            torch_dtype=self.weight_dtype,
        )
        self.transformer.requires_grad_(False)

        inner_dim = self.transformer.inner_dim
        self.transformer.register_parameter(
            "class_embedding_U",
            nn.Parameter(torch.randn(self.num_classes, inner_dim), requires_grad=True)
        )
        nn.init.orthogonal_(self.transformer.class_embedding_U)

        deg_cls_path = self.config.network.get("degradation_classifier_path")
        self.deg_classifier = DegNet_DINO(
            dino_type=self.config.network.get("dino_type", None),
            num_types=self.num_classes
        )
        self.deg_classifier.load_state_dict(
            torch.load(deg_cls_path, map_location="cpu"),
            strict=False
        )

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )

        self.vae.to(dtype=torch.float32)
        self.transformer.to(dtype=self.weight_dtype)
        self.deg_classifier.to(dtype=self.weight_dtype)
        self.deg_classifier.eval()

        self._class_embedding_U_full = self.transformer.class_embedding_U.data.to(
            device=torch.device("cuda"), dtype=self.weight_dtype
        )

    def load_checkpoint(self, ckpt_path):
        model_path = self.ckpt_path = ckpt_path

        metadata_path = os.path.join(ckpt_path, "metadata.pt")
        if os.path.exists(metadata_path):
            metadata = torch.load(metadata_path, map_location="cpu")
            self.global_step = metadata.get("global_step", 0)
            self.global_iter = metadata.get("global_iter", 0)
            logger.info(f"Loaded metadata: step={self.global_step}, iter={self.global_iter}")

        safetensors_path = os.path.join(ckpt_path, "model.safetensors")
        if not os.path.exists(safetensors_path):
            raise FileNotFoundError(f"Checkpoint not found: {safetensors_path}")

        state_dict = load_file(safetensors_path)
        missing, unexpected = self.transformer.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded transformer from {safetensors_path}")
        if missing:
            logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    def setup_output(self):
        ckpt_name = os.path.basename(self.ckpt_path.rstrip('/'))
        exp_name = os.path.basename(os.path.dirname(self.ckpt_path))
        if self.args.output_dir:
            self.output_dir = self.args.output_dir
        else:
            self.output_dir = os.path.join(
                os.path.dirname(self.ckpt_path),
                f"test_vis_{ckpt_name}"
            )
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")

    def setup_prompt_embeddings(self):
        embed_dir = self.config.data.get("embed_dir", "./cached_embeddings")
        for root, dirs, files in os.walk(embed_dir):
            for fname in files:
                if not fname.endswith("_embeds.pt"):
                    continue
                fpath = os.path.join(root, fname)
                data = torch.load(fpath, map_location="cpu", weights_only=True)
                ds_idx = data["dataset_idx"]
                self.prompt_embed_cache[ds_idx] = (data["prompt_embeds"], data["text_ids"])
        logger.info(f"Loaded {len(self.prompt_embed_cache)} prompt embeddings from {embed_dir}")

    def extract_deg_feat(self, lq_images):
        with torch.no_grad():
            logits = self.deg_classifier(lq_images)
            probs = F.softmax(logits, dim=-1)[:, :, 0]
        deg_feat = probs.detach().to(dtype=self.weight_dtype) @ self._class_embedding_U_full
        return deg_feat

    def get_prompt_embeds(self, dataset_indices):
        prompt_embeds = torch.cat([
            self.prompt_embed_cache[int(i)][0].to(dtype=self.weight_dtype)
            for i in dataset_indices.tolist()
        ], dim=0)
        text_ids = self.prompt_embed_cache[int(dataset_indices[0].item())][1].to(dtype=self.weight_dtype)
        return prompt_embeds, text_ids

    def get_sigmas(self, timesteps, n_dim, dtype):
        sigmas = self.noise_scheduler.sigmas.to(device=self.vae.device, dtype=dtype)
        schedule = self.noise_scheduler.timesteps.to(device=self.vae.device, dtype=timesteps.dtype)
        sigma = sigmas[torch.searchsorted(schedule, timesteps)].flatten()
        while sigma.ndim < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    @torch.no_grad()
    def predict(self, lq_latents, batch, fixed_timestep, guidance_scale):
        batchsize = lq_latents.shape[0]
        device = lq_latents.device

        noise = torch.randn_like(lq_latents)
        timesteps = torch.full((batchsize,), fixed_timestep, device=device, dtype=self.weight_dtype)
        sigmas = self.get_sigmas(timesteps, lq_latents.ndim, lq_latents.dtype)

        # Flow matching: noisy = (1-sigma)*clean + sigma*noise
        noisy_input = (1.0 - sigmas) * lq_latents + sigmas * noise

        h, w = noisy_input.shape[2], noisy_input.shape[3]
        packed_input = _pack_latents(noisy_input, batchsize, h // 2, w // 2)
        latent_image_ids = _prepare_latent_image_ids(noisy_input)

        prompt_embeds, text_ids = self.get_prompt_embeds(batch["dataset_indices"])

        guidance = None
        if self.transformer.config.guidance_embeds:
            guidance = torch.full((batchsize,), guidance_scale, device=device, dtype=self.weight_dtype)

        deg_feat = self.extract_deg_feat(batch["conditioning_pixel_values"])

        model_pred = self.transformer(
            hidden_states=packed_input,
            timestep=timesteps.float() / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            deg_emb=deg_feat,
            return_dict=False,
        )[0]

        model_pred_unpacked = _unpack_latents_with_ids(model_pred, latent_image_ids, lq_latents.shape[1])

        # Denoise: x_pred = noisy - sigma * pred
        x_pred = noisy_input - sigmas * model_pred_unpacked

        return x_pred, model_pred

    def run(self):
        logger.info("=" * 60)
        logger.info("Flux2 Image Restoration Tester")
        logger.info("=" * 60)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {device}")

        # Setup
        self.setup_output()
        _flux2_transformer_forward()

        # Build models
        logger.info("Building models...")
        self.build_models()
        self.vae = self.vae.to(device).eval()
        self.transformer = self.transformer.to(device).eval()

        # Load checkpoint
        logger.info(f"Loading checkpoint: {self.args.checkpoint}")
        self.load_checkpoint(self.args.checkpoint)

        # Setup embeddings
        self.setup_prompt_embeddings()

        # Build datasets
        resolution = self.config.data.resolution
        datasets_cfg = self.config.data.get("datasets", {})
        val_datasets = []
        val_dataset_names = []

        for ds_name, ds_cfg in datasets_cfg.items():
            if ds_name.startswith("ValDataset"):
                dataset = PairedDataset2(
                    lq_path=ds_cfg.lq_path,
                    hq_path=ds_cfg.hq_path,
                    resolution=resolution,
                    prompt=ds_cfg.get("prompt", ""),
                    dataset_idx=len(val_datasets),
                )
                val_datasets.append(dataset)
                val_dataset_names.append(ds_name)
                logger.info(f"  Added dataset: {ds_name} ({len(dataset)} samples)")

        if not val_datasets:
            logger.error("No validation datasets found!")
            return

        # Dataloader
        val_cfg = self.config.data.dataloader.get("val", {})
        batch_size = val_cfg.get("batch_size", 1)

        val_dataloaders = [
            DataLoader(
                ds,
                shuffle=False,
                collate_fn=_collate_fn,
                batch_size=batch_size,
                num_workers=val_cfg.get("num_workers", 4),
                pin_memory=True,
                persistent_workers=False,
                drop_last=False,
            )
            for ds in val_datasets
        ]

        # Inference settings
        guidance_scale = self.args.guidance_scale
        fixed_timestep = self.args.fixed_timestep
        max_samples = self.args.max_samples

        # Metrics
        psnr_fn_local = PeakSignalNoiseRatio(data_range=1.0).to(device)
        ssim_fn_local = StructuralSimilarityIndexMeasure(data_range=1.0, data_dtype=torch.float32).to(device)

        # Run inference on each dataset
        for ds_idx, (ds_name, dataloader) in enumerate(zip(val_dataset_names, val_dataloaders)):
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Testing: {ds_name}")
            logger.info(f"{'=' * 50}")

            ds_output_dir = os.path.join(self.output_dir, ds_name)
            os.makedirs(ds_output_dir, exist_ok=True)

            ssim_total, psnr_total = 0.0, 0.0
            count = 0

            pbar = tqdm(dataloader, desc=ds_name)
            for batch in pbar:
                if max_samples and count >= max_samples:
                    break

                pixel_values = batch["pixel_values"].to(device)
                cond_pixel_values = batch["conditioning_pixel_values"].to(device)
                batch["dataset_indices"] = batch["dataset_indices"].to(device)

                # Encode LQ to latents
                lq_latents = encode_images(cond_pixel_values, self.vae, self.weight_dtype)

                # Predict
                x_pred, model_pred = self.predict(
                    lq_latents, batch, fixed_timestep, guidance_scale
                )

                # Decode
                latents_to_decode = x_pred
                if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor is not None:
                    latents_to_decode = (latents_to_decode / self.vae.config.scaling_factor) + self.vae.config.shift_factor

                generated = self.vae.decode(latents_to_decode.to(dtype=torch.float32)).sample

                pred_tensor = (generated / 2 + 0.5).clamp(0, 1).to(device)
                gt_tensor = (pixel_values + 1.0) / 2.0

                # Compute metrics
                psnr_val = psnr_fn_local(pred_tensor, gt_tensor)
                ssim_val = ssim_fn_local(pred_tensor, gt_tensor)

                psnr_total += psnr_val.item()
                ssim_total += ssim_val.item()
                count += 1

                # Save visualizations (first N samples)
                if count <= self.args.num_vis_samples:
                    gen_img = Image.fromarray(
                        (pred_tensor[0].cpu().float().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    ).convert("RGB")
                    gen_img.save(os.path.join(ds_output_dir, f"pred_{count:04d}.png"))

                    gt_img = Image.fromarray(
                        (gt_tensor[0].cpu().float().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    ).convert("RGB")
                    gt_img.save(os.path.join(ds_output_dir, f"hq_{count:04d}.png"))

                    lq_img = Image.fromarray(
                        (((cond_pixel_values[0].cpu() + 1.0) / 2.0).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    ).convert("RGB")
                    lq_img.save(os.path.join(ds_output_dir, f"lq_{count:04d}.png"))

                pbar.set_postfix({
                    "PSNR": f"{psnr_val.item():.2f}",
                    "SSIM": f"{ssim_val.item():.4f}",
                    "n": count,
                })

                # Cleanup
                del pred_tensor, gt_tensor, generated, latents_to_decode
                del model_pred, x_pred, lq_latents
                del pixel_values, cond_pixel_values
                gc.collect()
                torch.cuda.empty_cache()

            avg_psnr = psnr_total / count if count > 0 else 0
            avg_ssim = ssim_total / count if count > 0 else 0

            logger.info(f"  {ds_name} — PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}, Samples: {count}")

            # Save metrics
            metrics_file = os.path.join(ds_output_dir, "metrics.txt")
            with open(metrics_file, "w") as f:
                f.write(f"Dataset: {ds_name}\n")
                f.write(f"Checkpoint: {self.args.checkpoint}\n")
                f.write(f"Guidance Scale: {guidance_scale}\n")
                f.write(f"Fixed Timestep: {fixed_timestep}\n")
                f.write(f"Samples: {count}\n")
                f.write(f"PSNR: {avg_psnr:.4f}\n")
                f.write(f"SSIM: {avg_ssim:.4f}\n")

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Test completed. Results saved to: {self.output_dir}")
        logger.info(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Test Flux2 Image Restoration")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint folder")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--guidance_scale", type=float, default=3.0, help="Guidance scale")
    parser.add_argument("--fixed_timestep", type=int, default=500, help="Fixed timestep for inference")
    parser.add_argument("--max_samples", type=int, default=100, help="Max samples per dataset")
    parser.add_argument("--num_vis_samples", type=int, default=5, help="Number of visualization samples")
    args = parser.parse_args()

    tester = ConditionTester(args)
    tester.run()


if __name__ == "__main__":
    main()
