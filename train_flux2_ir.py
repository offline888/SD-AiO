import argparse
import ast
import logging
import math
import os
import shutil
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.models
import torchvision.transforms.functional as TF
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKLFlux2,
    FlowMatchEulerDiscreteScheduler,
    Flux2KleinIRPipeline,
)
from diffusers.models.transformers import Flux2Transformer2DModel
from diffusers.models.transformers.transformer_flux2 import Flux2Modulation
from diffusers.optimization import get_scheduler
from diffusers.models.embeddings import (
    TimestepEmbedding,
    Timesteps,
)
from diffusers.training_utils import compute_loss_weighting_for_sd3, free_memory
from diffusers.utils import check_min_version
from diffusers.utils.logging import (
    set_verbosity_error,
    set_verbosity_info,
)
from diffusers.utils.torch_utils import is_compiled_module
from omegaconf import OmegaConf
from PIL import Image
from src.data.dataset import PairedDataset
from src.networks.degnet import DegNet_DINO
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

check_min_version("0.38.0.dev0")
logger = logging.getLogger(__name__)

_printed: set = set()

def log_once(message, accelerator=None):
    global _printed
    key = str(message)
    if key in _printed:
        return
    _printed.add(key)
    if accelerator is None or accelerator.is_main_process:
        print(message, flush=True)


def compute_text_embeddings(prompt, pipeline):
    with torch.no_grad():
        prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=prompt,
            max_sequence_length=512,
        )
    return prompt_embeds, text_ids


class EfficientConvProj(nn.Module):
    """
    Flow: 1x1(expand) -> DWConv(3x3) -> 1x1(project) -> residual
    """
    def __init__(self, in_ch: int, out_ch: int, bottleneck_dim: int | None = None):
        super().__init__()
        if bottleneck_dim is None:
            bottleneck_dim = max(16, min(in_ch, out_ch // 64))
        self.bottleneck_dim = bottleneck_dim

        self.expand = nn.Conv2d(in_ch, bottleneck_dim, kernel_size=1)
        self.dwconv = nn.Conv2d(
            bottleneck_dim, bottleneck_dim, kernel_size=3,
            padding=1, groups=bottleneck_dim
        )
        self.act = nn.SiLU()
        self.project = nn.Conv2d(bottleneck_dim, out_ch, kernel_size=1)

        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bottleneck: in_ch -> bottleneck_dim
        x = self.expand(x)
        # Depthwise: spatial mixing on bottleneck_dim channels
        x = self.dwconv(x)
        x = self.act(x)
        # Project: bottleneck_dim -> out_ch
        x = self.project(x)
        return x


class TimeModulator(nn.Module):
    def __init__(self, in_channels: int, time_emb_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, in_channels * 2))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        scale_shift = self.mlp(temb)  # [B, 2 * C]
        scale, shift = scale_shift.chunk(2, dim=1)
        scale = scale.view(-1, x.size(1), 1, 1)
        shift = shift.view(-1, x.size(1), 1, 1)
        return x * (1 + scale) + shift

class FLUX2ModulationV2(nn.Module):
    def __init__(
        self,
        dim: int,
        mod_param_sets: int = 2,
        bias: bool = False,
        use_block_emb: bool = True,
        use_conv: bool = True,
        use_vae: bool = False,
        vae_path: str = "",
    ):
        super().__init__()

        self.dim = dim
        self.mod_param_sets = mod_param_sets
        self.use_block_emb = use_block_emb
        self.use_conv = use_conv
        self.use_vae = use_vae

        self.act_fn = nn.SiLU()
        self.linear = nn.Linear(dim, 3 * mod_param_sets * dim, bias=bias)

        if self.use_block_emb:
            self.block_proj = Timesteps(
                num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0
            )
            self.block_embedder = TimestepEmbedding(
                in_channels=256, time_embed_dim=dim, sample_proj_bias=bias
            )

        if self.use_conv and not self.use_vae:
            convnext = torchvision.models.convnext_small(
                weights=torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1
            )

            # ConvNeXt-Small 前3个 stage: S1(96ch) -> S2(192ch) -> S3(384ch)
            # 每个 stage 后接 TimeModulator，最后 S3 输出空间分辨率 H/16×W/16
            # seq = H/16 * W/16，与 Transformer img_ids token 数对齐
            self.conv_stem_s1 = convnext.features[:2]  # out_channels = 96
            self.conv_down1_s2 = convnext.features[2:4]  # out_channels = 192
            self.conv_down2_s3 = convnext.features[4:6]  # out_channels = 384

            self.conv_time_mod1 = TimeModulator(in_channels=96, time_emb_dim=dim)
            self.conv_time_mod2 = TimeModulator(in_channels=192, time_emb_dim=dim)
            self.conv_time_mod3 = TimeModulator(in_channels=384, time_emb_dim=dim)

            # Plan A: Bottleneck(expand) -> DWConv(3x3) -> 1x1(project)
            # bottleneck_dim=384: expand=147.5K, dwconv=3.5K, project=14.1M => ~14.2M
            # vs naive 1x1: 14.1M | vs original Linear: 114M
            self.feat_proj = EfficientConvProj(
                in_ch=384,
                out_ch=3 * mod_param_sets * dim,
                bottleneck_dim=384,
            )

        elif self.use_vae:
            self.vae = AutoencoderKLFlux2.from_pretrained(vae_path, subfolder="vae")
            self.vae.requires_grad_(False)
            self.vae.eval()

            vae_out_channels = self.vae.config.block_out_channels
            latent_dim = self.vae.config.latent_channels

            # 每个 down_block 之后各一个 TimeModulator，对应通道数为 block_out_channels
            # down_block[0]: 128ch -> vae_time_mods[0]: 128ch
            # down_block[1]: 256ch -> vae_time_mods[1]: 256ch
            # down_block[2]: 512ch -> vae_time_mods[2]: 512ch
            # down_block[3]: 512ch -> vae_time_mods[3]: 512ch
            self.vae_time_mods = nn.ModuleList(
                [
                    TimeModulator(in_channels=ch, time_emb_dim=dim)
                    for ch in vae_out_channels
                ]
            )
            # mid_block 之后单独一个 TimeModulator
            self.vae_mid_time_mod = TimeModulator(
                in_channels=vae_out_channels[-1], time_emb_dim=dim
            )

            self.vae_proj = EfficientConvProj(
                in_ch=latent_dim,
                out_ch=3 * mod_param_sets * dim,
                bottleneck_dim=32,
            )

    @staticmethod
    def _pack_latents(latents):
        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels, height * width).permute(
            0, 2, 1
        )
        return latents

    def forward(
        self,
        lq_tensor: torch.Tensor,
        temb: torch.Tensor,
        block_idx: int | None = None,
    ) -> torch.Tensor:
        B = temb.size(0)

        # Detect the target dtype from any of our linear weights (these are
        # cast to bf16/fp16 before prepare() and stay that way).
        w_dtype = self.linear.weight.dtype

        if self.use_block_emb and block_idx is not None:
            if isinstance(block_idx, int):
                block_tensor = torch.tensor([block_idx], dtype=torch.long, device=temb.device)
            else:
                block_tensor = block_idx.long().to(device=temb.device)
            bemb = self.block_proj(block_tensor)
            # Cast to bf16 so matmuls use the bf16 paths inside block_embedder
            # (weights are bf16, input must match).
            bemb = self.block_embedder(bemb.to(dtype=w_dtype))
            # temb is float32 from Flux2TimestepGuidanceEmbeddings; cast it.
            temb = (temb.to(dtype=w_dtype) + bemb)

        mod_time = self.act_fn(temb)
        mod_time = self.linear(mod_time)
        mod = mod_time.unsqueeze(1)

        if lq_tensor is not None:
            if self.use_conv and not self.use_vae:
                x = self.conv_stem_s1(lq_tensor)
                x = self.conv_time_mod1(x, temb)
                x = self.conv_down1_s2(x)
                x = self.conv_time_mod2(x, temb)
                x = self.conv_down2_s3(x)
                x = self.conv_time_mod3(x, temb)

                lq_mod = self.feat_proj(x)   # [B, 3*2*6144, H/16, W/16]
                lq_mod = lq_mod.permute(0, 2, 3, 1).reshape(B, -1, 3 * self.mod_param_sets * self.dim)
                mod = mod + lq_mod

            elif self.use_vae:
                encoder = self.vae.encoder
                x = encoder.conv_in(lq_tensor)

                for i, down_block in enumerate(encoder.down_blocks):
                    x = down_block(x)
                    if i < len(self.vae_time_mods):
                        x = self.vae_time_mods[i](x, temb)

                x = encoder.mid_block(x)
                x = self.vae_mid_time_mod(x, temb)

                x = encoder.conv_norm_out(x)
                x = encoder.conv_act(x)
                x = encoder.conv_out(x)
                if self.vae.quant_conv is not None:
                    x = self.vae.quant_conv(x)

                lq_feat, _ = torch.chunk(x, 2, dim=1)  # [B, latent_dim=32, H/16, W/16]
                lq_mod = self.vae_proj(lq_feat)       # [B, 3*2*dim H/16, W/16]
                lq_mod = lq_mod.permute(0, 2, 3, 1).reshape(B, -1, 3 * self.mod_param_sets * self.dim)
                mod = mod + lq_mod
        return mod

    @staticmethod
    def split(
        mod: torch.Tensor, mod_param_sets: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
        if mod.ndim == 2:
            mod = mod.unsqueeze(1)
            mod_params = torch.chunk(mod, 3 * mod_param_sets, dim=-1)
            return tuple(mod_params[3 * i : 3 * (i + 1)] for i in range(mod_param_sets))
        elif mod.ndim == 3:
            mod_params = torch.chunk(mod, 3 * mod_param_sets, dim=-1)
            return tuple(mod_params[3 * i : 3 * (i + 1)] for i in range(mod_param_sets))
        else:
            raise RuntimeError(f"mod dim is not 2 or 3: {mod.ndim}")


def parse_timestep(value):
    if value.startswith("[") and value.endswith("]"):
        try:
            return ast.literal_eval(value)
        except Exception:
            raise argparse.ArgumentTypeError(f"Invalid list format: {value}")
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer: {value}")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Flux.2 single-step image restoration fine-tuning."
    )

    # Model
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, default=None, required=True
    )
    parser.add_argument("--datasets_config", type=str, default=None, required=True)
    parser.add_argument("--resolution", type=int, default=512)

    # Training
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--save_checkpointing_steps", type=int, default=500)
    parser.add_argument(
        "--val_monitor_steps",
        type=int,
        default=None,
        help="Set to 0 to disable in-training visual monitoring.",
    )
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--fixed_timestep", type=parse_timestep, default=100)
    parser.add_argument("--num_inference_steps", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    # Optimizer
    parser.add_argument(
        "--optimizer", type=str, default="AdamW", choices=["AdamW", "prodigy"]
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # LR Scheduler
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)

    # Degradation Classifier
    parser.add_argument("--degradation_classifier_path", type=str, default=None, required=True)
    parser.add_argument("--dino_type", type=str, default=None)
    parser.add_argument("--num_deg_types", type=int, default=4)

    # Modulation backbone type
    parser.add_argument(
        "--mod_lq_type",
        type=str,
        default="convnext",
        choices=["convnext", "vae"],
        help="Backbone type for FLUX2ModulationV2: 'convnext' (ConvNeXt-Small) or 'vae' (dedicated VAE encoder)",
    )

    # Accelerate config
    parser.add_argument("--output_dir", type=str, default="flux2-image-restoration")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="swanlab")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument(
        "--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"]
    )
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args(input_args)
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args

def collate_fn(examples):
    return {
        "hq_pixel_values": torch.stack([e["hq_pixel_values"] for e in examples]),
        "lq_pixel_values": torch.stack([e["lq_pixel_values"] for e in examples]),
        "prompts": [e["prompt"] for e in examples],
        "dataset_indices": torch.tensor(
            [e["dataset_idx"] for e in examples], dtype=torch.long
        ),
        "deg_types": [e["deg_type"] for e in examples],
    }

class DegFeatExtractor(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        num_deg_types: int,
        weight_dtype: torch.dtype,
        args: argparse.Namespace,
        deg_embedding: nn.Parameter | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        inner_dim = transformer.inner_dim
        if deg_embedding is not None:
            self.deg_embedding = deg_embedding 
        else:
            self.deg_embedding=nn.Parameter(torch.randn(num_deg_types, inner_dim))
            nn.init.orthogonal_(self.deg_embedding)
        
        transformer.register_parameter("deg_embedding", self.deg_embedding)

        self.weight_dtype = weight_dtype
        self.deg_classifier = DegNet_DINO(
            dino_type=args.dino_type,
            num_types=num_deg_types,
        )
        state_dict = torch.load(args.degradation_classifier_path, map_location="cpu")
        self.deg_classifier.load_state_dict(state_dict, strict=False)
        self.deg_classifier.requires_grad_(False).eval()
        self.deg_classifier.to(device=device or torch.device("cpu"))

    def forward(self, lq_images: torch.Tensor) -> torch.Tensor:
        logits = self.deg_classifier(lq_images)
        deg_probs = torch.softmax(logits, dim=-1)[:, :, 0].to(dtype=self.weight_dtype)
        embedding = self.deg_embedding.to(device=lq_images.device, dtype=self.weight_dtype)
        return deg_probs @ embedding

def validate(
    pipeline,
    val_items: list,
    guidance_scale: float,
    fixed_timestep: int,
    device: torch.device,
    output_dir: str,
    global_step: int,
):
    os.makedirs(output_dir, exist_ok=True)
    step_dir = os.path.join(output_dir, f"step_{global_step:06d}")
    os.makedirs(step_dir, exist_ok=True)

    pipeline.eval()

    IMGNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    IMGNET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    with torch.no_grad():
        for item in val_items:
            lq_tensor = item["lq_pixel_values"].to(device)
            hq_tensor = item["hq_pixel_values"].to(device)

            lq_pil = TF.to_pil_image((lq_tensor * IMGNET_STD + IMGNET_MEAN).clamp(0, 1))

            pred_output = pipeline(
                lq_image=lq_pil,
                prompt_embeds=item["prompt_embeds"].unsqueeze(0).to(device),
                num_inference_steps=1,
                guidance_scale=guidance_scale,
                fixed_timestep=fixed_timestep,
                output_type="np",
                return_dict=True,
            )
            pred_np = pred_output.images[0]
            pred_tensor = torch.from_numpy(pred_np).permute(2, 0, 1)

            lq_vis = (lq_tensor.to(device) * IMGNET_STD + IMGNET_MEAN).clamp(0, 1).cpu()
            hq_vis = (hq_tensor.float().to(device) * IMGNET_STD + IMGNET_MEAN).clamp(0, 1).cpu()
            pred_vis = pred_tensor.clamp(0, 1)

            grid = torch.cat([lq_vis, pred_vis, hq_vis], dim=2)
            dataset_label = item.get("label", "unknown")
            Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).astype("uint8")).save(
                os.path.join(step_dir, f"{dataset_label}.png")
            )

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
        set_verbosity_info()
    else:
        set_verbosity_error()

    if isinstance(args.fixed_timestep, list):
        args.fixed_timestep = args.fixed_timestep[0]
    if not isinstance(args.fixed_timestep, int):
        args.fixed_timestep = int(args.fixed_timestep)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = Qwen2TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )

    weight_dtype = torch.bfloat16

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
    )
    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(accelerator.device)
    latents_bn_std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
    ).to(accelerator.device)

    transformer = Flux2Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
    )

    text_encoder = Qwen3ForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
    )

    text_encoding_pipeline = Flux2KleinIRPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=None,
        transformer=None,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=None,
    )

    num_deg_types = args.num_deg_types
    deg_extractor = DegFeatExtractor(
        transformer=transformer,
        num_deg_types=num_deg_types,
        weight_dtype=weight_dtype,
        args=args,
        deg_embedding=None,
        device=accelerator.device,
    )

    # Replace img double-stream modulation with FLUX2ModulationV2
    modulation_names = ["double_stream_modulation_img"]
    use_conv = (args.mod_lq_type == "convnext")
    use_vae = (args.mod_lq_type == "vae")

    for mod_name in modulation_names:
        if hasattr(transformer, mod_name) and isinstance(
            getattr(transformer, mod_name), Flux2Modulation
        ):
            orig = getattr(transformer, mod_name)
            new_mod = FLUX2ModulationV2(
                dim=orig.linear.in_features,
                mod_param_sets=orig.mod_param_sets,
                bias=orig.linear.bias is not None,
                use_block_emb=True,
                use_conv=use_conv,
                use_vae=use_vae,
                vae_path=args.pretrained_model_name_or_path,
            )
            new_mod.linear.load_state_dict(orig.linear.state_dict())
            setattr(transformer, mod_name, new_mod)
            log_once(f"Replaced {mod_name} with FLUX2ModulationV2 ({args.mod_lq_type})", accelerator)

    # Freeze all, unfreeze trainable submodules
    transformer.requires_grad_(False)
    if hasattr(transformer, "class_embedding_U"):
        transformer.class_embedding_U.requires_grad_(True)
    for mod_name in modulation_names:
        if hasattr(transformer, mod_name):
            mod = getattr(transformer, mod_name)

            # Shared modules (always unlock)
            for sub in ["block_proj", "block_embedder"]:
                if hasattr(mod, sub):
                    getattr(mod, sub).requires_grad_(True)
                    log_once(f"Unlocked {mod_name}.{sub}", accelerator)

            # ConvNeXt path
            if use_conv:
                for sub in [
                    "conv_stem_s1", "conv_down1_s2", "conv_down2_s3",
                    "conv_time_mod1", "conv_time_mod2", "conv_time_mod3",
                    "feat_proj",
                ]:
                    if hasattr(mod, sub):
                        getattr(mod, sub).requires_grad_(True)
                        log_once(f"Unlocked {mod_name}.{sub}", accelerator)

            # VAE path
            if use_vae:
                for sub in [
                    "vae_time_mods", "vae_mid_time_mod", "vae_proj",
                ]:
                    if hasattr(mod, sub):
                        getattr(mod, sub).requires_grad_(True)
                        log_once(f"Unlocked {mod_name}.{sub}", accelerator)
                # Freeze entire VAE (including encoder), only time_mods and proj are trainable
                for n, p in mod.vae.named_parameters():
                    p.requires_grad_(False)
                log_once(f"Frozen {mod_name}.vae (all params)", accelerator)

            # Freeze the linear (original weights copied, not trained)
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
        log_once(
            f"Unsupported optimizer: {args.optimizer}. Defaulting to AdamW.",
            accelerator,
        )
        args.optimizer = "adamw"

    # Unlock deg_embedding (registered on transformer by DegFeatExtractor)
    if hasattr(transformer, "deg_embedding"):
        transformer.deg_embedding.requires_grad_(True)
        log_once("Unlocked transformer.deg_embedding", accelerator)

    params_to_optimize = [
        {
            "params": [
                p
                for p in transformer.parameters()
                if p.requires_grad
            ],
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
    data_config = OmegaConf.load(args.datasets_config)
    train_datasets_cfg = [
        data_config[k] for k in data_config.keys() if k.startswith("Train")
    ]

    train_datasets = []
    for key in data_config.keys():
        if not key.startswith("Train"):
            continue
        ds_cfg = data_config[key]
        dataset = PairedDataset(
            lq_path=str(ds_cfg.lq_path),
            hq_path=str(ds_cfg.hq_path),
            resolution=args.resolution,
            prompt=str(ds_cfg.prompt),
            dataset_idx=len(train_datasets),
            deg_type=str(ds_cfg.deg_type),
            enlarge_ratio=float(ds_cfg.enlarge_ratio),
        )
        train_datasets.append(dataset)
        log_once(
            f"Loaded train dataset '{key}': lq={ds_cfg.lq_path}, hq={ds_cfg.hq_path}, "
            f"prompt='{ds_cfg.prompt}'",
            accelerator,
        )

    if not train_datasets:
        raise ValueError(
            f"No training datasets found in {args.datasets_config}. Expected keys starting with 'Train'."
        )

    # Load train dataloader
    train_dataset = ConcatDataset(train_datasets)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    # Pre-compute text embeddings for each train dataset
    prompt_embeds_cache, text_ids_cache = {}, {}
    for idx, cfg in enumerate(train_datasets_cfg):
        with torch.no_grad():
            embeds, tids = compute_text_embeddings(cfg.prompt, text_encoding_pipeline)
        prompt_embeds_cache[idx] = embeds.to(accelerator.device)
        text_ids_cache[idx] = tids.to(accelerator.device)

    val_monitor_steps = args.val_monitor_steps if args.val_monitor_steps is not None else args.save_checkpointing_steps
    monitor_items = []
    if val_monitor_steps > 0 and accelerator.is_main_process:
        for ds_idx, (ds, cfg) in enumerate(zip(train_datasets, train_datasets_cfg)):
            hq_path, lq_path = ds.pairs[0]  # fixed first pair
            hq_pil = ds._load_image(hq_path)
            lq_pil = ds._load_image(lq_path)
            lq_t = ds.transforms(lq_pil)  # ImageNet norm
            hq_t = ds.transforms(hq_pil)  # ImageNet norm
            monitor_items.append(
                {
                    "label": cfg.deg_type,
                    "lq_pixel_values": lq_t,
                    "hq_pixel_values": hq_t.clone(),
                    "prompt_embeds": prompt_embeds_cache[ds_idx].squeeze(
                        0
                    ),  # [seq, dim]
                    "text_ids": text_ids_cache[ds_idx].squeeze(0),
                }
            )
            log_once(
                f"[ValMonitor] ds={ds_idx} ({cfg.deg_type}): lq={lq_path}",
                accelerator,
            )
        monitor_output_dir = os.path.join(args.output_dir, "val_monitor")
        log_once(
            f"[ValMonitor] Will snapshot every {val_monitor_steps} steps to {monitor_output_dir}",
            accelerator,
        )
    else:
        monitor_output_dir = None

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

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Custom hooks so accelerator.save_state() serializes only trainable params
    def save_model_hook(models, weights, output_dir):
        transformer_model = None
        for model in models:
            m = unwrap_model(model)
            if isinstance(m, Flux2Transformer2DModel):
                transformer_model = model
                break
        if transformer_model is None:
            raise ValueError("No Flux2Transformer2DModel found in models")
        if weights:
            weights.pop()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(transformer_model)
            trainable = {
                k: v.to("cpu") for k, v in state_dict.items() if v.requires_grad
            }
            if not trainable:
                return
            torch.save(trainable, os.path.join(output_dir, "modulation_weights.pt"))
            log_once(
                f"Saved {len(trainable)} tensors ({sum(v.numel() for v in trainable.values()):,} params) "
                f"to {output_dir}/modulation_weights.pt",
                accelerator,
            )

    def load_model_hook(models, input_dir):
        transformer_ = None
        while models:
            model = models.pop()
            m = unwrap_model(model)
            if isinstance(m, Flux2Transformer2DModel):
                transformer_ = m
                break
        if transformer_ is None:
            raise ValueError("No Flux2Transformer2DModel found in models")
        ckpt_path = os.path.join(input_dir, "modulation_weights.pt")
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            missing, unexpected = transformer_.load_state_dict(state_dict, strict=False)
            log_once(
                f"Restored {len(state_dict)} tensors ({sum(v.numel() for v in state_dict.values()):,} params) "
                f"from {ckpt_path}",
                accelerator,
            )
            if missing:
                log_once(f"  missing: {missing}", accelerator)
            if unexpected:
                log_once(f"  unexpected: {unexpected}", accelerator)

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
            log_once(
                f"Checkpoint '{args.resume_from_checkpoint}' not found. Starting fresh.",
                accelerator,
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            log_once(f"Resuming from checkpoint {path}", accelerator)
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
                    pe = prompt_embeds_cache[ds_idx].to(accelerator.device)
                    ti = text_ids_cache[ds_idx].to(accelerator.device)
                    prompt_embeds = pe.repeat(bsz, 1, 1)
                    text_ids = ti.repeat(bsz, 1, 1)
                else:
                    all_pes, all_tis = [], []
                    for idx in range(bsz):
                        ds_idx = dataset_indices[idx].item()
                        pe = prompt_embeds_cache[ds_idx].to(accelerator.device)
                        ti = text_ids_cache[ds_idx].to(accelerator.device)
                        all_pes.append(pe)
                        all_tis.append(ti)
                    prompt_embeds = torch.cat(all_pes, dim=0)
                    text_ids = torch.cat(all_tis, dim=0)

                lq_pixel_values = batch["lq_pixel_values"].to(device=accelerator.device, dtype=weight_dtype)
                hq_pixel_values = batch["hq_pixel_values"].to(device=accelerator.device, dtype=weight_dtype)

                with torch.no_grad():
                    model_input = vae.encode(lq_pixel_values).latent_dist.mode()
                    hq_latent = vae.encode(hq_pixel_values).latent_dist.mode()

                model_input = Flux2KleinIRPipeline._patchify_latents(model_input)
                model_input = (model_input - latents_bn_mean) / latents_bn_std
                hq_target = Flux2KleinIRPipeline._patchify_latents(hq_latent)
                hq_target = (hq_target - latents_bn_mean) / latents_bn_std
                model_input_ids = Flux2KleinIRPipeline._prepare_latent_ids(
                    model_input
                ).to(model_input.device)

                noise = torch.randn_like(model_input)

                if args.fixed_timestep is not None:
                    fixed_idx = min(
                        args.fixed_timestep, len(noise_scheduler.timesteps) - 1
                    )
                    timesteps = torch.full(
                        (bsz,), fixed_idx, dtype=torch.long, device=model_input.device
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

                guidance = torch.full(
                    (bsz,),
                    args.guidance_scale,
                    device=accelerator.device,
                    dtype=weight_dtype,
                )

                deg_token = deg_extractor(lq_pixel_values).unsqueeze(1)
                deg_txt_id = text_ids[:, :1, :].clone()
                deg_txt_ids = torch.cat([deg_txt_id, text_ids], dim=1)

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
                    model_pred,
                    model_input_ids_trimmed,
                )

                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme="none", sigmas=sigmas
                )
                target = noise - hq_target
                # weighting is 4D [1,16,P,1]; model_pred is 4D [B,C,H,W]; target is 4D [B,C,H,W]
                # Flatten spatial dims of weighting to match model_pred/target
                loss = torch.mean(
                    (
                        weighting.float() * (model_pred.float() - target.float()) ** 2
                    ).reshape(model_pred.shape[0], -1),
                    1,
                ).mean()

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

                if global_step % args.save_checkpointing_steps == 0:
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

                # In-training val_monitor snapshot
                if val_monitor_steps > 0 and global_step % val_monitor_steps == 0:
                    unwrapped_transformer = unwrap_model(transformer)
                    val_pipeline = Flux2KleinIRPipeline(
                        vae=vae,
                        transformer=unwrapped_transformer,
                        scheduler=noise_scheduler,
                        text_encoder=None,
                        tokenizer=None,
                    )
                    val_pipeline.deg_extractor = deg_extractor
                    val_pipeline.to(device=accelerator.device, dtype=weight_dtype)
                    validate(
                        pipeline=val_pipeline,
                        val_items=monitor_items,
                        guidance_scale=args.guidance_scale,
                        fixed_timestep=args.fixed_timestep,
                        device=accelerator.device,
                        output_dir=monitor_output_dir,
                        global_step=global_step,
                    )
                    log_once(
                        f"[ValMonitor] saved snapshot at step={global_step}",
                        accelerator,
                    )

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
            # validation disabled
            pass

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        unwrapped_transformer = accelerator.unwrap_model(transformer)
        state_dict = accelerator.get_state_dict(unwrapped_transformer)
        trainable_dict = {
            k: v.to("cpu") for k, v in state_dict.items() if v.requires_grad
        }
        if trainable_dict:
            path = os.path.join(args.output_dir, "modulation_weights.pt")
            torch.save(trainable_dict, path)
            log_once(
                f"Saved {len(trainable_dict)} tensors ({sum(v.numel() for v in trainable_dict.values()):,} params) "
                f"to {path}",
                accelerator,
            )

        for entry in os.scandir(args.output_dir):
            if entry.is_dir() and entry.name.startswith("checkpoint"):
                for name in ["class_embedding_U.pt", "trainable_weights.pt"]:
                    p = os.path.join(entry.path, name)
                    if os.path.exists(p):
                        os.remove(p)
                        log_once(f"Cleaned legacy: {p}", accelerator)

        # final validation disabled
        accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)