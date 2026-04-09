import argparse
import copy
import logging
import math
import os
import shutil
import sys
from pathlib import Path

import cv2
import diffusers.utils.logging
import lpips
import numpy as np
import piq
import swanlab
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKLFlux2,
    FlowMatchEulerDiscreteScheduler,
    Flux2KleinPipeline,
)
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinIRPipeline
from diffusers.models.transformers import Flux2Transformer2DModel
from diffusers.models.transformers.transformer_flux2 import Flux2Modulation
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_loss_weighting_for_sd3, free_memory
from diffusers.utils import check_min_version
from omegaconf import OmegaConf
from PIL import Image
from src.data.dataset import PairedDataset
from src.networks.degnet import DegNet_DINO
from torch.utils.data import ConcatDataset
from tqdm.auto import tqdm
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM
from accelerate import DistributedDataParallelKwargs

check_min_version("0.38.0.dev0")
logger = logging.getLogger(__name__)

def log_once(msg, accelerator=None):
    if accelerator is None or accelerator.is_main_process:
        print(msg, flush=True)
        sys.stdout.flush()

_lpips_model = None

def get_lpips_model(device="cuda"):
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net="alex").eval().to(device)
    return _lpips_model

def compute_metrics(pred_np: np.ndarray, gt_np: np.ndarray, lpips_model=None) -> dict:
    pred_t = torch.from_numpy(pred_np).permute(2, 0, 1).float() / 255.0  # [3, H, W], [0,1]
    gt_t = torch.from_numpy(gt_np).permute(2, 0, 1).float() / 255.0

    # SSIM: data_range=1.0 since inputs are [0,1]
    ssim_val = piq.ssim(pred_t.unsqueeze(0), gt_t.unsqueeze(0), data_range=1.0).item()

    # PSNR: data_range=1.0
    psnr_val = piq.psnr(pred_t.unsqueeze(0), gt_t.unsqueeze(0), data_range=1.0,
                         reduction="mean").item()

    # LPIPS: lpips expects inputs in [-1, 1], shape [B, 3, H, W]
    lpips_t = lpips_model if lpips_model is not None else get_lpips_model()
    lpips_device = next(lpips_t.parameters()).device
    pred_lpips = pred_t.to(lpips_device) * 2.0 - 1.0
    gt_lpips = gt_t.to(lpips_device) * 2.0 - 1.0
    lpips_val = lpips_t(pred_lpips.unsqueeze(0), gt_lpips.unsqueeze(0)).item()

    return {"ssim": ssim_val, "psnr": psnr_val, "lpips": lpips_val}

class FLUX2ModulationV2(nn.Module):
    def __init__(
        self,
        dim: int,
        mod_param_sets: int = 2,
        bias: bool = False,
        n_blocks: int = 56,
    ):
        super().__init__()
        self.dim = dim
        self.mod_param_sets = mod_param_sets
        self.n_blocks = n_blocks

        self.linear = nn.Linear(dim, dim * 3 * mod_param_sets, bias=bias)
        self.act_fn = nn.SiLU()

        # 8 frequencies is enough to capture the main patterns in the block index
        if n_blocks is not None:
            self.num_freqs = 8
            self.block_proj = nn.Sequential(
                nn.Linear(self.num_freqs * 2, dim // 2, bias=bias),
                nn.SiLU(),
                nn.Linear(dim // 2, dim * 3 * mod_param_sets, bias=bias),
            )

        # use convnext to extract features
        self.convnext = torchvision.models.convnext_small(
            weights=torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1,
        )
        self.up_conv = nn.ConvTranspose2d(768, 768, kernel_size=4, stride=2, padding=1)
        self.feat_proj = nn.Linear(768, dim * 3 * mod_param_sets)

    def fourier_encode(self, block_idx: int | torch.Tensor, target_dtype: torch.dtype = None) -> torch.Tensor:
        if isinstance(block_idx, torch.Tensor):
            block_idx = (
                block_idx.item() if block_idx.numel() == 1 else block_idx.float()
            )
        else:
            block_idx = torch.tensor(
                block_idx, device=next(self.parameters()).device, dtype=torch.float32
            )

        freqs = 2 ** torch.arange(
            self.num_freqs, device=block_idx.device, dtype=block_idx.dtype
        )
        angs = freqs * math.pi * block_idx.unsqueeze(-1)
        result = torch.cat([torch.sin(angs), torch.cos(angs)], dim=-1)
        
        # Cast to target_dtype if provided
        if target_dtype is not None:
            result = result.to(target_dtype)
        return result

    def forward(
        self,
        lq_tensor: torch.Tensor,  # [B, 3, H, W] 低质量 RGB 图像
        temb: torch.Tensor,
        block_idx: int | None = None,
    ) -> torch.Tensor:

        B = temb.size(0)

        mod = self.act_fn(temb)
        mod = self.linear(mod)

        # Cast lq_tensor to temb.dtype to avoid dtype mismatch with convnext and subsequent layers
        lq_tensor_dtype = temb.dtype
        if lq_tensor.dtype != lq_tensor_dtype:
            lq_tensor = lq_tensor.to(lq_tensor_dtype)
        
        feat = self.convnext.features(lq_tensor)  # [B,768,H//32,W//32]
        feat = self.up_conv(feat)  # [B,768,H//16,W//16]
        feat = feat.flatten(2).permute(0, 2, 1).view(-1, 768)  # [B*H*W//256,768]
        feat = self.feat_proj(feat)  # [B*H*W//256,dim * 3 * mod_param_sets]
        
        # Ensure feat is in temb.dtype for subsequent operations
        if feat.dtype != temb.dtype:
            feat = feat.to(temb.dtype)
            
        feat = feat.view(
            B, 3 * self.mod_param_sets, -1, self.dim
        )  # [B, 3 * mod_param_sets, H*W//256, dim] -> [B, 3 * mod_param_sets, seq_len ,dim]

        if block_idx is not None:
            # Use temb's dtype for block embedding to avoid dtype mismatch
            block_emb = self.block_proj(self.fourier_encode(block_idx, target_dtype=temb.dtype))
            mod = mod + block_emb  # [B, dim * 3 * mod_param_sets]

        mod = mod.view(
            B, 3 * self.mod_param_sets, 1, -1
        )  # [B, 3 * mod_param_sets, dim]
        mod = mod + feat  # [B, 3 * mod_param_sets, H*W//256, dim]

        return mod

    @staticmethod
    def split(mod: torch.Tensor, mod_param_sets: int) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
        if mod.ndim == 2:
            mod = mod.unsqueeze(1)
            mod_params = torch.chunk(mod, 3 * mod_param_sets, dim=-1)
            return tuple(mod_params[3 * i : 3 * (i + 1)] for i in range(mod_param_sets))
        elif mod.ndim == 4:
            # [B,3*mod_param_sets,seq,dim]
            mod.params = torch.chunk(mod, 3 * mod_param_sets, dim=1)
            return tuple(mod_params[3 * i : 3 * (i + 1)] for i in range(mod_param_sets))

def log_validation(
    unwrapped_transformer,
    vae,
    weight_dtype,
    args,
    accelerator,
    pipeline_args,
    epoch,
    global_step=0,
    is_final_validation=False,
):
    unwrapped_transformer.eval()

    def get_sigmas_fixed(sigmas_tensor, n_dim, dtype):
        sigmas_tensor = sigmas_tensor.float()
        schedule = sigmas_tensor
        while True:
            schedule = schedule[:-1]
            if len(schedule) <= n_dim:
                break
        return schedule.float().to(dtype=dtype)

    def gather_list(lst):
        if accelerator.num_processes == 1:
            return lst
        gathered = [None] * accelerator.num_processes
        torch.distributed.all_gather_object(gathered, lst)
        out = []
        for g in gathered:
            out.extend(g)
        return out

    is_main = accelerator.is_main_process
    phase = "test" if is_final_validation else "validation"

    local_ssim, local_psnr, local_lpips = [], [], []
    local_images = []

    num_vis = min(args.num_validation_images, 4)
    desc = f"[GPU{accelerator.process_index}] Val"
    pbar = tqdm(pipeline_args, desc=desc, disable=not is_main, file=sys.stdout, leave=True)

    guidance_scale = args.guidance_scale
    fixed_timestep = args.fixed_timestep

    for val_idx, batch in enumerate(pbar):
        lq_pixels = batch["lq_pixel_values"].to(accelerator.device)
        hq_pixels = batch["hq_pixel_values"].to(accelerator.device)
        batch_size = lq_pixels.shape[0]

        lq_latents = vae.encode(lq_pixels.to(vae.dtype)).latent_dist.sample().to(weight_dtype)
        bs, c, h, w = lq_latents.shape

        packed = lq_latents.view(bs, c, h, 2, w, 2).permute(0, 2, 4, 1, 3, 5).reshape(bs, h * w, c * 4)
        img_ids = torch.cartesian_prod(
            torch.arange(1, device=packed.device),
            torch.arange(h // 2, device=packed.device),
            torch.arange(w // 2, device=packed.device),
            torch.arange(1, device=packed.device),
        ).unsqueeze(0).expand(bs, -1, -1).contiguous()

        timesteps = torch.full((batch_size,), fixed_timestep, device=packed.device, dtype=weight_dtype)
        sigmas = get_sigmas_fixed(timesteps, lq_latents.ndim, weight_dtype)

        noise = torch.randn_like(packed)
        noisy_input = (1.0 - sigmas) * packed + sigmas * noise

        guidance = torch.full((batch_size,), guidance_scale, device=packed.device, dtype=weight_dtype)

        deg_emb = None
        if hasattr(args, "num_deg_types"):
            deg_emb = unwrapped_transformer.extract_deg_feat(lq_pixels, num_deg_types=args.num_deg_types)
            deg_emb = deg_emb.unsqueeze(1)

        prompt_embeds = batch["prompt_embeds"].to(packed.device) if "prompt_embeds" in batch else None
        txt_ids = torch.zeros((batch_size, 1, 3), device=packed.device, dtype=torch.long)
        if prompt_embeds is None:
            prompt_embeds = torch.zeros((batch_size, 512, 4096), device=packed.device, dtype=weight_dtype)

        with torch.no_grad():
            model_pred = unwrapped_transformer(
                hidden_states=noisy_input,
                timestep=timesteps.float() / 1000,
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                txt_ids=txt_ids,
                img_ids=img_ids,
                deg_emb=deg_emb,
                return_dict=False,
            )[0]

        model_pred_unpacked = model_pred.view(bs, h, w, c, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(bs, c, h * 2, w * 2)
        x_pred = packed - sigmas.view(bs, 1, 1, 1) * model_pred_unpacked

        if hasattr(vae.config, "shift_factor") and vae.config.shift_factor is not None:
            x_pred = x_pred / vae.config.scaling_factor + vae.config.shift_factor

        recon = vae.decode(x_pred.to(dtype=torch.float32)).sample
        pred_t = (recon / 2 + 0.5).clamp(0, 1)
        gt_t = (hq_pixels + 1.0) / 2.0

        lpips_model = get_lpips_model(accelerator.device)
        pred_np = (pred_t[0].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
        gt_np = (gt_t[0].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)

        metrics = compute_metrics(pred_np, gt_np, lpips_model=lpips_model)
        local_ssim.append(metrics["ssim"])
        local_psnr.append(metrics["psnr"])
        local_lpips.append(metrics["lpips"])
        local_images.append(Image.fromarray(pred_np))

        if val_idx >= num_vis - 1:
            break

    all_ssim = gather_list(local_ssim)
    all_psnr = gather_list(local_psnr)
    all_lpips = gather_list(local_lpips)

    avg_ssim = float(np.mean(all_ssim)) if all_ssim else 0.0
    avg_psnr = float(np.mean(all_psnr)) if all_psnr else 0.0
    avg_lpips = float(np.mean(all_lpips)) if all_lpips else 0.0

    if is_main:
        log_once(
            f"[{phase}] Epoch {epoch} — SSIM: {avg_ssim:.4f}  PSNR: {avg_psnr:.2f}  LPIPS: {avg_lpips:.4f}",
            accelerator,
        )
        for tracker in accelerator.trackers:
            tracker.writer.add_scalar(f"{phase}/ssim", avg_ssim, epoch)
            tracker.writer.add_scalar(f"{phase}/psnr", avg_psnr, epoch)
            tracker.writer.add_scalar(f"{phase}/lpips", avg_lpips, epoch)

        images = local_images[:args.num_validation_images]
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images(phase, np_images, epoch, dataformats="NHWC")
            elif tracker.name == "swanlab":
                tracker.log(
                    {phase: [swanlab.Image(img, caption=f"{i}") for i, img in enumerate(images)]},
                    step=accelerator.global_step,
                )

    torch.distributed.barrier()

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Flux.2 single-step image restoration fine-tuning."
    )

    # Model
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files (e.g. fp16).",
    )

    # Data
    parser.add_argument(
        "--datasets_config",
        type=str,
        default=None,
        required=True,
        help="Path to a YAML file listing datasets.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Resolution for input images. All images will be resized to this resolution.",
    )

    # Training
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=1
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps. Overrides num_train_epochs if provided.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint every X steps.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Maximum number of checkpoints to keep.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help='Resume training from a previous checkpoint, or "latest" for the last available.',
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps before a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Use gradient checkpointing to save memory at the expense of slower backward passes.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after any warmup).",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="Guidance scale for the model (used at inference time during validation).",
    )
    parser.add_argument(
        "--fixed_timestep",
        type=int,
        default=300,
        help="Fixed timestep index for noise injection (determines sigma_start).",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=1,
        help="Number of denoising steps in pipeline inference.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses for data loading.",
    )

    # Optimizer
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help='Supported: "AdamW" and "prodigy".',
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)

    # LR Scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="consine",
        help='Choose between "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup".',
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)

    # Degradation Classifier
    parser.add_argument(
        "--degradation_classifier_path",
        type=str,
        default=None,
        help="Path to the degradation classifier checkpoint (.pt file).",
    )
    parser.add_argument(
        "--dino_type",
        type=str,
        default=None,
        help="DINO model type for degradation classifier.",
    )

    # Validation
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=2,
        help="Number of images to generate per validation set.",
    )

    # Output / Logging
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux2-image-restoration",
        help="The output directory for checkpoints and logs.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Log directory.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help='"tensorboard" (default), "swanlab", or "all".',
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Allow TF32 on Ampere GPUs.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Mixed precision type.",
    )
    parser.add_argument("--local_rank", type=int, default=-1)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def collate_fn(examples):
    hq_pixel_values = torch.stack([e["hq_pixel_values"] for e in examples]).float()
    lq_pixel_values = torch.stack([e["lq_pixel_values"] for e in examples]).float()
    prompts = [e["instance_prompt"] for e in examples]
    dataset_indices = torch.tensor(
        [e["dataset_idx"] for e in examples], dtype=torch.long
    )

    return {
        "hq_pixel_values": hq_pixel_values,
        "lq_pixel_values": lq_pixel_values,
        "prompts": prompts,
        "dataset_indices": dataset_indices,
    }


class DegFeatExtractor:
    def __init__(self, transformer, num_deg_types, weight_dtype, device, args, accelerator=None):
        self.weight_dtype = weight_dtype
        self.args = args
        inner_dim = transformer.inner_dim

        self._class_embedding_U = nn.Parameter(
            torch.randn(num_deg_types, inner_dim, device=device, dtype=weight_dtype)
        )
        nn.init.orthogonal_(self._class_embedding_U.to(torch.float32))
        self._class_embedding_U.data = self._class_embedding_U.data.to(weight_dtype)
        transformer.register_parameter("class_embedding_U", self._class_embedding_U)
        log_once(
            f"Registered trainable class_embedding_U with shape {self._class_embedding_U.shape}",
            accelerator,
        )

        self.deg_classifier = None
        if args.degradation_classifier_path is not None:
            log_once("Loading degradation classifier...", accelerator)
            self.deg_classifier = DegNet_DINO(
                dino_type=args.dino_type,
                num_types=num_deg_types,
            )
            self.deg_classifier.load_state_dict(
                torch.load(args.degradation_classifier_path, map_location="cpu"),
                strict=False,
            )
            self.deg_classifier.to(device, dtype=weight_dtype).eval()
            log_once("Degradation classifier loaded successfully", accelerator)

    def __call__(self, lq_images: torch.Tensor) -> torch.Tensor:
        if self.deg_classifier is None:
            raise RuntimeError(
                "No degradation classifier loaded; set --degradation_classifier_path"
            )
        with torch.no_grad():
            logits = self.deg_classifier(lq_images)
            probs = F.softmax(logits, dim=-1)[:, :, 0]
        deg_feat = probs.to(dtype=self.weight_dtype) @ self._class_embedding_U.to(probs.device)
        return deg_feat


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    log_once(accelerator.state, accelerator)
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = Qwen2TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    vae = AutoencoderKLFlux2.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        variant=args.variant,
    )
    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(accelerator.device)
    latents_bn_std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
    ).to(accelerator.device)

    transformer = Flux2Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        variant=args.variant,
        torch_dtype=weight_dtype,
    )

    text_encoder = Qwen3ForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        variant=args.variant,
    )
    text_encoder.requires_grad_(False)

    text_encoding_pipeline = Flux2KleinPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=None,
        transformer=None,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=None,
        variant=args.variant,
    )

    cfg = OmegaConf.load(args.datasets_config)
    train_datasets_cfg = [cfg[k] for k in cfg.keys() if k.startswith("Train")]
    num_deg_types = 4
    log_once(f"Detected {num_deg_types} training datasets (degradation types)", accelerator)

    deg_extractor = DegFeatExtractor(
        transformer,
        num_deg_types,
        weight_dtype,
        accelerator.device,
        args,
    )
    
    # replace the img diuble stream modulation with FLUX2ModulationV2
    modulation_names = ["double_stream_modulation_img"]
    for mod_name in modulation_names:
        if hasattr(transformer, mod_name) and isinstance(
            getattr(transformer, mod_name), Flux2Modulation
        ):
            original_module = getattr(transformer, mod_name)
            n_blocks = transformer.config.num_layers
            new_module = FLUX2ModulationV2(
                dim=original_module.linear.in_features,
                mod_param_sets=original_module.mod_param_sets,
                bias=original_module.linear.bias is not None,
                n_blocks=n_blocks,
            )
            new_module.linear.load_state_dict(original_module.linear.state_dict())
            setattr(transformer, mod_name, new_module)
            log_once(
                f"Replaced {mod_name} with FLUX2ModulationV2 (n_blocks={n_blocks})",
                accelerator,
            )

    transformer.requires_grad_(False)
    transformer.class_embedding_U.requires_grad_(True)
    vae.requires_grad_(False)

    for mod_name in modulation_names:
        if hasattr(transformer, mod_name):
            mod = getattr(transformer, mod_name)
            for sub_name in ["convnext", "up_conv", "feat_proj", "block_proj"]:
                if hasattr(mod, sub_name):
                    getattr(mod, sub_name).requires_grad_(True)
                    log_once(f"Unlocked {mod_name}.{sub_name} for training", accelerator)
            if hasattr(mod, "linear"):
                mod.linear.requires_grad_(False)
                log_once(f"Frozen {mod_name}.linear", accelerator)

    vae.to(device=accelerator.device, dtype=weight_dtype)
    transformer.to(device=accelerator.device, dtype=weight_dtype)
    text_encoder.to(device=accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.optimizer.lower() not in ("adamw", "prodigy"):
        log_once(f"Unsupported optimizer: {args.optimizer}. Defaulting to AdamW.", accelerator)
        args.optimizer = "adamw"

    params_to_optimize = [
        {
            "params": [p for p in transformer.parameters() if p.requires_grad and p is not deg_extractor._class_embedding_U],
            "lr": args.learning_rate,
        },
        {
            "params": [deg_extractor._class_embedding_U],
            "lr": args.learning_rate,
        },
    ]
    # status the number of parameters to optimize
    total_params = 0
    for p in params_to_optimize:
        total_params += sum(x.numel() for x in p["params"])
    log_once(f"Total number of parameters to optimize: {total_params}", accelerator)

    if args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    else:
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, install: `pip install prodigyopt`")
        optimizer = prodigyopt.Prodigy(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    train_datasets = []
    val_datasets = []
    val_prompts = []

    for key in cfg.keys():
        ds_cfg = cfg[key]
        is_train = key.startswith("Train")
        is_val = key.startswith("Val")
        if not is_train and not is_val:
            continue

        ds = PairedDataset(
            lq_path=str(ds_cfg.lq_path),
            hq_path=str(ds_cfg.hq_path),
            resolution=args.resolution,
            prompt=str(ds_cfg.prompt),
            dataset_idx=len(train_datasets) if is_train else len(val_datasets),
        )

        if is_train:
            train_datasets.append(ds)
            log_once(
                f"Loaded train dataset '{key}': lq={ds_cfg.lq_path}, hq={ds_cfg.hq_path}, "
                f"prompt='{ds_cfg.prompt}'",
                accelerator,
            )
        elif is_val:
            val_datasets.append(ds)
            val_prompts.append(str(ds_cfg.prompt))
            log_once(
                f"Loaded val dataset '{key}': lq={ds_cfg.lq_path}, hq={ds_cfg.hq_path}, "
                f"prompt='{ds_cfg.prompt}'",
                accelerator,
            )

    if not train_datasets:
        raise ValueError(
            f"No training datasets found in {args.datasets_config}. "
            "Expected keys starting with 'Train'."
        )

    train_dataset = ConcatDataset(train_datasets)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    def compute_text_embeddings(prompt, pipeline):
        with torch.no_grad():
            prompt_embeds, text_ids = pipeline.encode_prompt(
                prompt=prompt,
                max_sequence_length=512,
            )
        return prompt_embeds, text_ids

    dataset_prompt_embeds_cache = {}
    dataset_text_ids_cache = {}
    for ds_idx, ds_cfg in enumerate(train_datasets_cfg):
        prompt_str = ds_cfg.prompt
        with torch.no_grad():
            embeds, tids = compute_text_embeddings(prompt_str, text_encoding_pipeline)
        dataset_prompt_embeds_cache[ds_idx] = embeds
        dataset_text_ids_cache[ds_idx] = tids

    val_dataloaders_list = None
    if val_datasets and val_prompts:
        val_dataloaders_list = []
        for ds, val_prompt in zip(val_datasets, val_prompts):
            sampler = torch.utils.data.DistributedSampler(
                ds,
                num_replicas=accelerator.num_processes,
                rank=accelerator.process_index,
                shuffle=False,
            )
            val_loader = torch.utils.data.DataLoader(
                ds,
                batch_size=1,
                num_workers=args.dataloader_num_workers,
                sampler=sampler,
            )
            val_dataloaders_list.append((val_prompt, val_loader))

    del text_encoder, tokenizer, text_encoding_pipeline
    free_memory()

    num_warmup_steps = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_dataloader_sharded = math.ceil(
            len(train_dataloader) / accelerator.num_processes
        )
        num_update_steps_per_epoch = math.ceil(
            len_dataloader_sharded / args.gradient_accumulation_steps
        )
        num_training_steps = (
            args.num_train_epochs
            * accelerator.num_processes
            * num_update_steps_per_epoch
        )
    else:
        num_training_steps = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(
            "flux2-image-restoration",
            config=vars(args),
        )

    def save_model_hook(models, weights, output_dir):
        transformer_model = None
        for model in models:
            m = accelerator.unwrap_model(model)
            if isinstance(m, Flux2Transformer2DModel):
                transformer_model = m
                break
            if hasattr(model, "_orig_mod") and isinstance(
                model._orig_mod, Flux2Transformer2DModel
            ):
                transformer_model = model._orig_mod
                break

        if transformer_model is None:
            raise ValueError("No Flux2Transformer2DModel found in models")

        if accelerator.is_main_process and weights:
            weights.pop()

        if accelerator.is_main_process:
            state_dict = transformer_model.state_dict()
            modulation_state_dict = {
                k: v.to("cpu")
                for k, v in state_dict.items()
                if "double_stream_modulation" in k or "class_embedding_U" in k
            }
            torch.save(
                modulation_state_dict,
                os.path.join(output_dir, "modulation_weights.pt"),
            )
            log_once(
                f"Saved {len(modulation_state_dict)} modulation parameters to {output_dir}/modulation_weights.pt",
                accelerator,
            )

    def load_model_hook(models, input_dir):
        transformer_ = None
        for model in models:
            m = accelerator.unwrap_model(model)
            if isinstance(m, Flux2Transformer2DModel):
                transformer_ = m
                break
            if hasattr(model, "_orig_mod") and isinstance(
                model._orig_mod, Flux2Transformer2DModel
            ):
                transformer_ = model._orig_mod
                break
        if transformer_ is None:
            raise ValueError("No Flux2Transformer2DModel found in models")

        ckpt_path = os.path.join(input_dir, "modulation_weights.pt")
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location="cpu")
            log_once(f"Loaded modulation weights from {ckpt_path}", accelerator)
            transformer_.load_state_dict(state_dict, strict=False)
            log_once(f"Restored {len(state_dict)} modulation parameters", accelerator)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)

        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches per epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Total batch size = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = sorted(
                [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                key=lambda x: int(x.split("-")[1]),
            )
            path = dirs[-1] if dirs else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' not found. Starting fresh.",
                only_main_process=True,
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]

            with accelerator.accumulate(models_to_accumulate):
                dataset_indices = batch["dataset_indices"].to(accelerator.device)
                bsz = dataset_indices.shape[0]
                unique_ds = dataset_indices.unique()
                if unique_ds.numel() == 1:
                    ds_idx = unique_ds.item()
                    pe = dataset_prompt_embeds_cache[ds_idx].to(accelerator.device)
                    ti = dataset_text_ids_cache[ds_idx].to(accelerator.device)
                    prompt_embeds = pe.repeat(bsz, 1, 1)
                    text_ids = ti.repeat(bsz, 1, 1)
                else:
                    all_pes, all_tis = [], []
                    for idx in range(bsz):
                        ds_idx = dataset_indices[idx].item()
                        pe = dataset_prompt_embeds_cache[ds_idx].to(accelerator.device)
                        ti = dataset_text_ids_cache[ds_idx].to(accelerator.device)
                        all_pes.append(pe)
                        all_tis.append(ti)
                    prompt_embeds = torch.cat(all_pes, dim=0)
                    text_ids = torch.cat(all_tis, dim=0)

                lq_pixel_values = batch["lq_pixel_values"].to(dtype=weight_dtype)
                hq_pixel_values = batch["hq_pixel_values"].to(dtype=weight_dtype)


                with torch.no_grad():
                    model_input = vae.encode(lq_pixel_values).latent_dist.mode()
                    hq_latent = vae.encode(hq_pixel_values).latent_dist.mode()

                model_input = Flux2KleinIRPipeline._patchify_latents(model_input)
                model_input = (model_input - latents_bn_mean) / latents_bn_std

                hq_target =  Flux2KleinIRPipeline._patchify_latents(hq_latent)
                hq_target = (hq_target - latents_bn_mean) / latents_bn_std

                model_input_ids = Flux2KleinIRPipeline._prepare_latent_ids(model_input).to(
                    device=model_input.device
                )

                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                if args.fixed_timestep is not None:
                    # 300 < 1000
                    fixed_idx = min(
                        args.fixed_timestep, len(noise_scheduler.timesteps) - 1
                    )
                    timesteps = torch.full(
                        (bsz,),
                        fixed_idx,
                        dtype=torch.long,
                        device=model_input.device,
                    )
                else:
                    raise NotImplementedError(
                        "Random timestep sampling is not implemented for single-step IR."
                    )

                sigmas = get_sigmas(
                    timesteps, n_dim=model_input.ndim, dtype=model_input.dtype
                )
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                packed_noisy_model_input = Flux2KleinIRPipeline._pack_latents(
                    noisy_model_input
                )

                orig_input_shape = packed_noisy_model_input.shape
                orig_input_ids_shape = model_input_ids.shape

                packed_noisy_model_input = torch.cat(
                    [packed_noisy_model_input], dim=1
                )
                model_input_ids = torch.cat(
                    [model_input_ids], dim=1
                )
                guidance = torch.full(
                    [bsz], args.guidance_scale, device=accelerator.device, dtype=weight_dtype
                )

                deg_token = deg_extractor(lq_pixel_values).unsqueeze(1)
                #print(f"\n========== Training Loop - Transformer Input ==========")
                #print(f"[TRAIN] packed_noisy_model_input: {packed_noisy_model_input.shape}")
                #print(f"[TRAIN] timesteps: {timesteps.shape}, values: {timesteps[:5]}")
                #print(f"[TRAIN] guidance: {guidance.shape}")
                #print(f"[TRAIN] prompt_embeds: {prompt_embeds.shape}")
                deg_txt_id = text_ids[:, :1, :].clone()
                deg_txt_ids = torch.cat([deg_txt_id, text_ids], dim=1)
                #print(f"[TRAIN] deg_txt_ids: {deg_txt_ids.shape}")
                #print(f"[TRAIN] model_input_ids: {model_input_ids.shape}")
                #print(f"[TRAIN] deg_token: {deg_token.shape}")
                #print(f"[TRAIN] lq_pixel_values: {lq_pixel_values.shape}")
                
                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=deg_txt_ids,
                    img_ids=model_input_ids,
                    deg_emb=deg_token,
                    return_dict=False,
                    lq_tensor=lq_pixel_values,
                )[0]

                model_pred = model_pred[:, : orig_input_shape[1], :]
                model_input_ids_trimmed = model_input_ids[
                    :, : orig_input_ids_shape[1], :
                ]
                model_pred = Flux2KleinIRPipeline._unpack_latents_with_ids(
                    model_pred, model_input_ids_trimmed
                )

                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme="none", sigmas=sigmas
                )
                # lq_latent  -> hq_latent
                target = noise - hq_target
                loss = torch.mean(
                    (
                        weighting.float() * (model_pred.float() - target.float()) ** 2
                    ).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        transformer.parameters(), args.max_grad_norm
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if args.checkpoints_total_limit is not None:
                        checkpoints = sorted(
                            [
                                d
                                for d in os.listdir(args.output_dir)
                                if d.startswith("checkpoint")
                            ],
                            key=lambda x: int(x.split("-")[1]),
                        )
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = (
                                len(checkpoints) - args.checkpoints_total_limit + 1
                            )
                            removing = checkpoints[:num_to_remove]
                            log_once(
                                f"Removing {len(removing)} old checkpoints "
                                f"(limit={args.checkpoints_total_limit})",
                                accelerator,
                            )
                            for rc in removing:
                                shutil.rmtree(os.path.join(args.output_dir, rc))

                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}"
                    )
                    accelerator.save_state(save_path)
                    log_once(f"Saved state to {save_path}", accelerator)

                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            pass  # validation disabled
            # if val_dataloaders_list is not None and epoch % args.validation_epochs == 0:
            #     unwrapped_transformer = accelerator.unwrap_model(transformer)
            #     for val_idx, (_, val_loader) in enumerate(val_dataloaders_list):
            #         log_validation(
            #             unwrapped_transformer=unwrapped_transformer,
            #             vae=vae,
            #             weight_dtype=weight_dtype,
            #             args=args,
            #             accelerator=accelerator,
            #             pipeline_args=val_loader,
            #             epoch=epoch,
            #             global_step=global_step,
            #             is_final_validation=False,
            #         )

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        unwrapped_transformer = accelerator.unwrap_model(transformer)
        state_dict = accelerator.get_state_dict(unwrapped_transformer)
        modulation_state_dict = {
            k.replace("transformer.", ""): v.to("cpu")
            for k, v in state_dict.items()
            if k.startswith("transformer.")
            and (
                "double_stream_modulation" in k
                or "class_embedding_U" in k
            )
        }
        torch.save(
            modulation_state_dict,
            os.path.join(args.output_dir, "modulation_weights.pt"),
        )
        log_once(
            f"Saved modulation weights to {args.output_dir}/modulation_weights.pt",
            accelerator,
        )

        # Final validation
        # if val_dataloaders_list is not None and args.num_validation_images > 0:
        #     unwrapped_transformer = accelerator.unwrap_model(transformer)
        #     for val_idx, (_, val_loader) in enumerate(val_dataloaders_list):
        #         log_validation(
        #             unwrapped_transformer=unwrapped_transformer,
        #             vae=vae,
        #             weight_dtype=weight_dtype,
        #             args=args,
        #             accelerator=accelerator,
        #             pipeline_args=val_loader,
        #             epoch=-1,
        #             global_step=global_step,
        #             is_final_validation=True,
        #         )

        accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)