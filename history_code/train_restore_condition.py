import argparse
import gc
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxImg2ImgPipeline,
    FluxTransformer2DModel,
)
from diffusers.models.normalization import AdaLayerNormZero
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import ChainDataset, DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm

import swanlab

from src.data.dataset import InterleavedShuffleDataset, PairedDataset
from src.networks.degnet import DegNet_DINO


def encode_images(
    pixels: torch.Tensor,
    vae: torch.nn.Module,
    weight_dtype: torch.dtype
) -> torch.Tensor:
    pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
    pixel_latents = (
        pixel_latents - vae.config.shift_factor
    ) * vae.config.scaling_factor
    return pixel_latents.to(weight_dtype)


class ConditionTrainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.config = OmegaConf.load(args.config)

        self.exp_name = self.config.get("name", "flux_condition_finetune")
        self.exp_name = f"{self.exp_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        set_seed(self.config.seed)

        self.num_classes: int = self.config.network.get("num_classes", 4)
        self.current_epoch: int = 0
        self.global_step: int = 0

        self.accelerator: Optional[Accelerator] = None
        self.is_main: bool = False
        self.unwrap_fn: Optional[callable] = None

        self.vae: Optional[AutoencoderKL] = None
        self.transformer: Optional[FluxTransformer2DModel] = None
        self.noise_scheduler: Optional[FlowMatchEulerDiscreteScheduler] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.train_dataloader: Optional[DataLoader] = None
        self.val_dataloader: Optional[DataLoader] = None
        self.weight_dtype: Optional[torch.dtype] = None
        self.vae_scale_factor: Optional[int] = None

        self.train_datasets: List[Any] = []
        self.val_datasets: List[Any] = []
        self.prompt_embed_cache: Dict[int, Tuple[Any, Any, Any]] = {}
        self._fixed_vis_samples: Optional[List[Dict[str, Any]]] = None

        self.deg_classifier = DegNet_DINO(
            dino_type=self.config.network.get("dino_type", None),
            num_types=self.num_classes
        )
        self.deg_classifier.load_state_dict(
            torch.load(self.config.network.degradation_classifier_path, map_location="cpu"),
            strict=False
        )

        self.output_dir: Optional[str] = None
        self.ckpt_dir: Optional[str] = None
        self.log_dir: Optional[str] = None

        self.logger = logging.getLogger(__name__)

    def setup_dist(self) -> None:
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=self.config.accelerator.get(
                "find_unused_parameters", False
            )
        )

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.train.get(
                "gradient_accumulation_steps", 4
            ),
            mixed_precision=self.config.accelerator.get("mixed_precision", "bf16"),
            log_with="swanlab",
            project_dir=self.config.accelerator.get("project_dir", "./experiments"),
            kwargs_handlers=[ddp_kwargs],
        )

        self.is_main = self.accelerator.is_main_process
        self.unwrap_fn = self.accelerator.unwrap_model

    def init_logger(self) -> None:
        self.output_dir = os.path.join(
            self.config.accelerator.get("project_dir", "./experiments"), self.exp_name
        )
        self.ckpt_dir = os.path.join(self.output_dir, "checkpoints")

        if self.is_main:
            os.makedirs(self.ckpt_dir, exist_ok=True)
            config_save_path = os.path.join(self.output_dir, "config.yaml")
            OmegaConf.save(self.config, config_save_path)

        swanlab_config = OmegaConf.to_container(self.config)
        self.log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        swanlab_config["log_dir"] = self.log_dir

        init_kwargs = {
            "swanlab": {
                "experiment_name": self.exp_name,
                "log_dir": self.log_dir,
            }
        }
        self.accelerator.init_trackers(
            project_name=self.config.logging.swanlab_project,
            config=swanlab_config,
            init_kwargs=init_kwargs,
        )

    def build_models(self) -> None:
        model_path = self.config.network.pretrained_model_name_or_path

        precisions = {"fp16": torch.float16, "bf16": torch.bfloat16}
        self.weight_dtype = precisions.get(
            self.config.accelerator.get("mixed_precision", "bf16"), torch.float32
        )

        self.vae = AutoencoderKL.from_pretrained(
            model_path,
            subfolder="vae",
            revision=self.config.network.get("revision"),
            variant=self.config.network.get("variant"),
            torch_dtype=torch.float32,
        )
        self.vae.requires_grad_(False)
        self.vae.enable_tiling()

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.transformer = FluxTransformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            revision=self.config.network.get("revision"),
            variant=self.config.network.get("variant"),
            torch_dtype=self.weight_dtype,
        )
        self.transformer.requires_grad_(False)

        self.transformer.register_parameter(
            "class_embedding_U",
            nn.Parameter(torch.randn(self.num_classes, 768), requires_grad=True)
        )
        nn.init.orthogonal_(self.transformer.class_embedding_U)
        self._class_embedding_U_cpu = self.transformer.class_embedding_U.data.clone().cpu()

        self._unfreeze_adaln_layers()

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path,
            subfolder="scheduler",
        )

        self.fixed_timestep = self.config.train.get("fixed_timestep", None)

        if self.config.network.get("gradient_checkpointing", False):
            self.transformer.enable_gradient_checkpointing()
        self.vae.to(dtype=torch.float32)
        self.transformer.to(dtype=self.weight_dtype)
        if self.config.accelerator.get("allow_tf32", False):
            torch.backends.cuda.matmul.allow_tf32 = True

    def _unfreeze_adaln_layers(self) -> None:
        """
        解冻 AdaLN 中 SiLU 之后的 Linear 层（shift/scale/gate 投影）

        AdaLayerNormZero: Linear(dim, 6*dim) -> 输出 shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        """
        unfreeze_type = self.config.network.get("unfreeze_adaln_type", "adaln")

        if unfreeze_type == "single":
            return

        adaln_linear_count = 0
        for name, module in self.transformer.named_modules():
            if isinstance(module, AdaLayerNormZero):
                module.linear.requires_grad_(True)
                adaln_linear_count += 1

        self.logger.info(f"[AdaLN] unfreeze AdaLayerNormZero: {adaln_linear_count} layers")

    def _get_param_name(self, param: nn.Parameter) -> str:
        for name, p in self.transformer.named_parameters():
            if p is param:
                return name
        return ""

    def extract_deg_feat(self, lq_images: torch.Tensor) -> torch.Tensor:
        self.deg_classifier.to(self.accelerator.device)
        self.deg_classifier.eval()

        with torch.no_grad():
            logits = self.deg_classifier(lq_images)
            # 逐行softmax
            probs = F.softmax(logits, dim=-1)[:, :, 0]

        probs = probs.detach().clone().to(dtype=self.weight_dtype)
        class_emb_U = self._class_embedding_U_full
        deg_feat = probs @ class_emb_U

        self.deg_classifier.cpu()

        return deg_feat

    def build_dataloader(self) -> None:
        resolution = self.config.data.resolution
        datasets_cfg = self.config.data.get("datasets", {})

        train_datasets = []
        val_datasets = []

        for ds_idx, (ds_name, ds_cfg) in enumerate(datasets_cfg.items()):
            is_val = ds_name.startswith("ValDataset")
            prompt = ds_cfg.get("prompt", "")
            dataset = PairedDataset(
                lq_path=ds_cfg.lq_path,
                hq_path=ds_cfg.hq_path,
                resolution=resolution,
                prompt=prompt,
                dataset_idx=ds_idx,
                enlarge_ratio=ds_cfg.get("enlarge_ratio", 1.0),
            )
            if is_val:
                val_datasets.append(dataset)
            else:
                train_datasets.append(dataset)

        self.train_datasets = train_datasets
        self.val_datasets = val_datasets

        train_loader_cfg = self.config.data.dataloader.train
        batch_size = train_loader_cfg.batch_size

        if len(train_datasets) > 1:
            self.train_dataloader = DataLoader(
                InterleavedShuffleDataset(
                    train_datasets, buffer_size=500, seed=self.config.seed
                ),
                shuffle=False,
                collate_fn=self.collate_fn,
                batch_size=batch_size,
                num_workers=train_loader_cfg.get("num_workers", 2),
                pin_memory=train_loader_cfg.get("pin_memory", True),
                persistent_workers=train_loader_cfg.get("persistent_workers", True),
                drop_last=train_loader_cfg.get("drop_last", True),
            )
        else:
            self.train_dataloader = DataLoader(
                train_datasets[0],
                collate_fn=self.collate_fn,
                batch_size=batch_size,
                num_workers=train_loader_cfg.get("num_workers", 2),
                pin_memory=train_loader_cfg.get("pin_memory", True),
                persistent_workers=train_loader_cfg.get("persistent_workers", True),
                drop_last=train_loader_cfg.get("drop_last", True),
            )

        if len(val_datasets) > 0:
            val_loader_cfg = self.config.data.dataloader.val
            self.val_dataloader = DataLoader(
                ChainDataset(val_datasets),
                shuffle=False,
                collate_fn=self.collate_fn,
                batch_size=val_loader_cfg.get("batch_size", 1),
                num_workers=val_loader_cfg.get("num_workers", 2),
                pin_memory=val_loader_cfg.get("pin_memory", True),
                persistent_workers=val_loader_cfg.get("persistent_workers", True),
                drop_last=val_loader_cfg.get("drop_last", False),
            )
        else:
            self.val_dataloader = None

    @staticmethod
    def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        pixel_values = torch.stack([e["pixel_values"] for e in examples])
        cond_pixel_values = torch.stack(
            [e["conditioning_pixel_values"] for e in examples]
        )
        dataset_indices = torch.tensor(
            [ex["dataset_idx"] for ex in examples], dtype=torch.long
        )
        return {
            "pixel_values": pixel_values,
            "conditioning_pixel_values": cond_pixel_values,
            "dataset_indices": dataset_indices,
        }

    def setup_optimization(self) -> None:
        opt_cfg = self.config.train.optim

        decay_params = []
        no_decay_params = []

        for name, param in self.transformer.named_parameters():
            if not param.requires_grad:
                continue
            if "class_embedding_U" in name:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        trainable_params_set = set(decay_params) | set(no_decay_params)
        all_trainable = {p for p in self.transformer.parameters() if p.requires_grad}
        assert trainable_params_set == all_trainable, (
            f"Parameter mismatch: trainable={len(all_trainable)}, grouped={len(trainable_params_set)}"
        )

        adaln_params = [p for p in no_decay_params if ".linear." in self._get_param_name(p)]
        mlp_params = [p for p in no_decay_params if ".linear." not in self._get_param_name(p)]
        self.logger.info(
            f"[Training parameters] class_embedding_U: {len(decay_params)} parameters, "
            f"AdaLN Linear: {len(adaln_params)} parameters, "
            f"MLP (not frozen): {len(mlp_params)} parameters"
        )

        wd = opt_cfg.get("weight_decay", 0.01)
        param_groups = [
            {"params": decay_params, "weight_decay": wd},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=opt_cfg.get("lr", 1e-4),
            betas=(opt_cfg.get("beta1", 0.9), opt_cfg.get("beta2", 0.999)),
            eps=opt_cfg.get("epsilon", 1e-8),
        )

        scheduler_cfg = self.config.train.scheduler
        num_training_steps = (
            len(self.train_dataloader) // self.accelerator.gradient_accumulation_steps
        ) * self.config.train.num_train_epochs

        scheduler_kwargs = {
            "optimizer": self.optimizer,
            "num_warmup_steps": scheduler_cfg.get("lr_warmup_steps", 100),
            "num_training_steps": num_training_steps,
            "num_cycles": scheduler_cfg.get("lr_num_cycles", 1),
        }

        lr_end = scheduler_cfg.get("lr_end")
        if lr_end is not None:
            scheduler_kwargs["lr_end"] = lr_end

        self.scheduler = get_scheduler(scheduler_cfg.type, **scheduler_kwargs)

        self.transformer, self.optimizer, self.train_dataloader, self.scheduler = (
            self.accelerator.prepare(
                self.transformer, self.optimizer, self.train_dataloader, self.scheduler
            )
        )

        self._class_embedding_U_full = self._class_embedding_U_cpu.to(
            self.accelerator.device, dtype=self.weight_dtype
        )

        if not self.config.network.get("offload", False):
            self.vae = self.vae.to(self.accelerator.device)

        self._precompute_prompt_embeddings()

        gc.collect()
        torch.cuda.empty_cache()

    def _precompute_prompt_embeddings(self) -> None:
        embed_dir = self.config.data.get("embed_dir", "./cached_embeddings")

        datasets = self.train_datasets + self.val_datasets
        unique_dataset_idx = set([ds.dataset_idx for ds in datasets])

        for ds_idx in unique_dataset_idx:
            if ds_idx in self.prompt_embed_cache:
                continue

            load_path = os.path.join(embed_dir, f"dataset_{ds_idx}_embeds.pt")
            if not os.path.exists(load_path):
                raise FileNotFoundError(
                    f"找不到预计算的 prompt embedding 文件: {load_path}，"
                    "请先运行 embedding 提取脚本。"
                )

            loaded_data = torch.load(load_path, map_location="cpu", weights_only=True)

            self.prompt_embed_cache[ds_idx] = (
                loaded_data["prompt_embeds"],
                loaded_data["pooled_prompt_embeds"],
                loaded_data["text_ids"]
            )

    def _get_prompt_embeds_from_cache(
        self, dataset_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.accelerator.device

        prompt_embeds_list = []
        pooled_list = []
        text_ids_list = []

        for idx in dataset_indices.tolist():
            p_embed, pooled, txt_ids = self.prompt_embed_cache[int(idx)]
            prompt_embeds_list.append(p_embed.to(device, dtype=self.weight_dtype))
            pooled_list.append(pooled.to(device, dtype=self.weight_dtype))
            text_ids_list.append(txt_ids)

        prompt_embeds = torch.stack(prompt_embeds_list)
        pooled_prompt_embeds = torch.stack(pooled_list)
        text_ids = text_ids_list[0].to(device, dtype=self.weight_dtype)

        return prompt_embeds, pooled_prompt_embeds, text_ids

    def training_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        with self.accelerator.accumulate(self.transformer):
            if self.config.network.get("offload", False):
                self.vae.to(self.accelerator.device)

            hq_latents = encode_images(batch["pixel_values"], self.vae, self.weight_dtype)
            lq_latents = encode_images(batch["conditioning_pixel_values"], self.vae, self.weight_dtype)

            if self.config.network.get("offload", False):
                self.vae.cpu()
                torch.cuda.empty_cache()

            batchsize = lq_latents.shape[0]
            noise = torch.randn_like(
                lq_latents, device=lq_latents.device, dtype=self.weight_dtype
            )

            if self.fixed_timestep is not None:
                timesteps = torch.full(
                    (batchsize,),
                    self.fixed_timestep,
                    device=lq_latents.device,
                    dtype=lq_latents.dtype,
                )
                sigmas = self._get_sigmas(
                    timesteps, n_dim=lq_latents.ndim, dtype=lq_latents.dtype
                )
            else:
                density = compute_density_for_timestep_sampling(
                    weighting_scheme=self.config.loss.get("weighting_scheme", "none"),
                    batch_size=batchsize,
                    logit_mean=self.config.loss.get("logit_mean", 0.0),
                    logit_std=self.config.loss.get("logit_std", 1.0),
                    mode_scale=self.config.loss.get("mode_scale", 1.29),
                )
                indices = (density * self.noise_scheduler.config.num_train_timesteps).long()
                timesteps = self.noise_scheduler.timesteps[indices].to(
                    device=lq_latents.device, dtype=lq_latents.dtype
                )
                sigmas = self._get_sigmas(
                    timesteps, n_dim=lq_latents.ndim, dtype=lq_latents.dtype
                )

            noisy_input = (1.0 - sigmas) * hq_latents + sigmas * noise

            packed_input = FluxImg2ImgPipeline._pack_latents(
                noisy_input,
                batch_size=batchsize,
                num_channels_latents=noisy_input.shape[1],
                height=noisy_input.shape[2],
                width=noisy_input.shape[3],
            )

            latent_image_ids = FluxImg2ImgPipeline._prepare_latent_image_ids(
                batchsize,
                noisy_input.shape[2] // 2,
                noisy_input.shape[3] // 2,
                self.accelerator.device,
                self.weight_dtype,
            )

            if self.unwrap_fn(self.transformer).config.guidance_embeds:
                guidance = torch.full(
                    (batchsize,),
                    self.config.train.get("guidance_scale", 3.0),
                    device=noisy_input.device,
                    dtype=self.weight_dtype,
                )
            else:
                guidance = None

            prompt_embeds, pooled_prompt_embeds, text_ids = self._get_prompt_embeds_from_cache(
                batch["dataset_indices"]
            )

            if self.config.train.get("proportion_empty_prompts", 0) > 0:
                if torch.rand(1).item() < self.config.train.proportion_empty_prompts:
                    prompt_embeds.zero_()
                    pooled_prompt_embeds.zero_()

            lq_images_for_deg = batch["conditioning_pixel_values"].to(self.accelerator.device)
            deg_feat = self.extract_deg_feat(lq_images_for_deg)
            pooled_prompt_embeds = pooled_prompt_embeds + deg_feat

            model_pred = self.transformer(
                hidden_states=packed_input,
                timestep=timesteps / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]

            model_pred = FluxImg2ImgPipeline._unpack_latents(
                model_pred,
                height=noisy_input.shape[2] * self.vae_scale_factor,
                width=noisy_input.shape[3] * self.vae_scale_factor,
                vae_scale_factor=self.vae_scale_factor,
            )

            weighting = compute_loss_weighting_for_sd3(
                weighting_scheme=self.config.loss.get("weighting_scheme", "none"),
                sigmas=sigmas,
            )

            target = noise - hq_latents
            loss = torch.mean(
                (
                    weighting.float() * (model_pred.float() - target.float()) ** 2
                ).reshape(target.shape[0], -1),
            ).mean()

            self.accelerator.backward(loss)

            if self.accelerator.sync_gradients:
                max_grad_norm = self.config.train.get("max_grad_norm", 1.0)
                self.accelerator.clip_grad_norm_(
                    self.transformer.parameters(), max_grad_norm
                )

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            if self.accelerator.sync_gradients:
                self.global_step += 1
                log_interval = self.config.logging.get("log_interval", 10)
                if self.global_step % log_interval == 0 and self.is_main:
                    self.accelerator.log(
                        {
                            "train/loss": loss.item(),
                            "train/lr": self.optimizer.param_groups[0]["lr"],
                        },
                        step=self.global_step,
                    )

            return loss

    def _get_sigmas(
        self,
        timesteps: torch.Tensor,
        n_dim: int = 4,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        sigmas = self.noise_scheduler.sigmas.to(
            device=self.accelerator.device, dtype=dtype
        )
        schedule_timesteps = self.noise_scheduler.timesteps.to(self.accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    @torch.no_grad()
    def validation(self) -> Optional[Dict[str, Any]]:
        if self.val_dataloader is None:
            return None

        if self.config.network.get("offload", False):
            self.vae.to(self.accelerator.device)

        guidance_scale = self.config.val.get("guidance_scale", 3.0)
        fixed_val_timestep = self.config.val.get("fixed_val_timestep", 300)

        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.accelerator.device)
        psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(self.accelerator.device)

        all_ssim = []
        all_psnr = []
        image_logs = []

        to_tensor = transforms.ToTensor()

        num_vis_samples = self.config.val.get("num_vis_samples", 4)

        # Use fixed vis samples if already cached
        use_cached_vis = self._fixed_vis_samples is not None and len(self._fixed_vis_samples) > 0
        vis_batch_indices = None
        if use_cached_vis:
            vis_batch_indices = {cand["batch_idx"] for cand in self._fixed_vis_samples}

        if self.is_main:
            vis_dir = os.path.join(self.output_dir, f"val_vis_iter_{self.global_step}")
            os.makedirs(vis_dir, exist_ok=True)

        max_val_samples = self.config.val.get("max_val_samples")

        # Track vis candidates from different datasets (for first-time caching)
        vis_candidates: List[Dict[str, Any]] = []
        seen_datasets = set()

        for batch_idx, batch in enumerate(self.val_dataloader):
            if max_val_samples and batch_idx >= max_val_samples:
                break

            pixel_values = batch["pixel_values"].to(self.accelerator.device)
            cond_pixel_values = batch["conditioning_pixel_values"].to(
                self.accelerator.device
            )
            dataset_idx = int(batch["dataset_indices"][0].item())

            # Collect vis sample from first occurrence of each dataset
            if not use_cached_vis and dataset_idx not in seen_datasets and len(vis_candidates) < num_vis_samples:
                seen_datasets.add(dataset_idx)
                vis_candidates.append({"batch_idx": batch_idx, "batch": batch})

            cond_img_pil = transforms.ToPILImage()(cond_pixel_values[0].cpu()).convert("RGB")
            lq_latents = encode_images(cond_pixel_values, self.vae, self.weight_dtype)

            noise = torch.randn_like(lq_latents)
            timesteps = torch.full(
                (1,),
                fixed_val_timestep,
                device=lq_latents.device,
                dtype=self.weight_dtype,
            )
            sigmas = self._get_sigmas(
                timesteps, n_dim=lq_latents.ndim, dtype=lq_latents.dtype
            )

            noisy_input = (1.0 - sigmas) * lq_latents + sigmas * noise

            prompt_embeds, pooled_prompt_embeds, text_ids = self._get_prompt_embeds_from_cache(
                batch["dataset_indices"]
            )

            latent_image_ids = FluxImg2ImgPipeline._prepare_latent_image_ids(
                1,
                lq_latents.shape[2] // 2,
                lq_latents.shape[3] // 2,
                self.accelerator.device,
                self.weight_dtype,
            )

            packed_input = FluxImg2ImgPipeline._pack_latents(
                noisy_input,
                batch_size=1,
                num_channels_latents=noisy_input.shape[1],
                height=noisy_input.shape[2],
                width=noisy_input.shape[3],
            )

            if self.unwrap_fn(self.transformer).config.guidance_embeds:
                guidance = torch.full(
                    (1,),
                    guidance_scale,
                    device=self.accelerator.device,
                    dtype=self.weight_dtype,
                )
            else:
                guidance = None

            deg_feat = self.extract_deg_feat(cond_pixel_values)
            pooled_prompt_embeds = pooled_prompt_embeds + deg_feat

            model_pred = self.unwrap_fn(self.transformer)(
                hidden_states=packed_input,
                timestep=timesteps.float() / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]

            # Reconstruct: x_pred = x_t - sigma * model_pred
            x_pred = noisy_input - sigmas * model_pred

            model_pred_unpacked = FluxImg2ImgPipeline._unpack_latents(
                x_pred,
                height=lq_latents.shape[2] * self.vae_scale_factor,
                width=lq_latents.shape[3] * self.vae_scale_factor,
                vae_scale_factor=self.vae_scale_factor,
            )

            model_pred_unpacked = model_pred_unpacked / self.vae.config.scaling_factor + self.vae.config.shift_factor
            generated = self.vae.decode(model_pred_unpacked.to(dtype=torch.float32)).sample
            generated = (generated / 2 + 0.5).clamp(0, 1)
            generated = generated.cpu().float().permute(0, 2, 3, 1).numpy()[0]
            generated = (generated * 255).astype(np.uint8)
            generated = Image.fromarray(generated).convert("RGB")

            pred_tensor = (
                to_tensor(generated).unsqueeze(0).to(self.accelerator.device)
            )
            gt_tensor = pixel_values[0:1].to(self.accelerator.device)

            ssim_val = ssim_metric(pred_tensor, gt_tensor).item()
            psnr_val = psnr_metric(pred_tensor, gt_tensor).item()

            all_ssim.append(ssim_val)
            all_psnr.append(psnr_val)

            # Save visualization images - use fixed samples from different datasets
            is_vis_sample = False
            if use_cached_vis:
                is_vis_sample = any(
                    cand["batch_idx"] == batch_idx for cand in self._fixed_vis_samples
                )
            else:
                is_vis_sample = any(
                    cand["batch_idx"] == batch_idx for cand in vis_candidates
                )

            if self.is_main and is_vis_sample:
                # Determine sample index for naming
                if use_cached_vis:
                    sample_idx = next(
                        i for i, cand in enumerate(self._fixed_vis_samples)
                        if cand["batch_idx"] == batch_idx
                    )
                else:
                    sample_idx = next(
                        i for i, cand in enumerate(vis_candidates)
                        if cand["batch_idx"] == batch_idx
                    )

                # Get GT image
                gt_img = (pixel_values[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                gt_img = Image.fromarray(gt_img).convert("RGB")

                # Save LQ, pred, HQ as separate images
                lq_path = os.path.join(vis_dir, f"sample_{sample_idx:03d}_LQ.png")
                pred_path = os.path.join(vis_dir, f"sample_{sample_idx:03d}_pred.png")
                hq_path = os.path.join(vis_dir, f"sample_{sample_idx:03d}_HQ.png")

                cond_img_pil.save(lq_path)
                generated.save(pred_path)
                gt_img.save(hq_path)

                image_logs.append(
                    {
                        "validation_image": cond_img_pil,
                        "images": [generated],
                        "validation_prompt": self.val_datasets[0].prompt if self.val_datasets else "",
                        "dataset_idx": dataset_idx,
                    }
                )

        # Cache fixed vis samples after first validation
        if self._fixed_vis_samples is None and len(vis_candidates) > 0:
            self._fixed_vis_samples = vis_candidates

        avg_ssim = sum(all_ssim) / len(all_ssim) if all_ssim else None
        avg_psnr = sum(all_psnr) / len(all_psnr) if all_psnr else None

        if self.is_main:
            val_metrics = {}
            if avg_ssim is not None:
                val_metrics["validation/SSIM"] = avg_ssim
            if avg_psnr is not None:
                val_metrics["validation/PSNR"] = avg_psnr

            if val_metrics:
                self.accelerator.log(val_metrics, step=self.global_step)

        for log in image_logs:
            self.accelerator.log(
                {"validation/input": swanlab.Image(log["validation_image"], caption=log.get("validation_prompt", ""))},
                step=self.global_step,
            )
            self.accelerator.log(
                {"validation/output": swanlab.Image(log["images"][0], caption=log.get("validation_prompt", ""))},
                step=self.global_step,
            )

        free_memory()

        if avg_ssim is not None or avg_psnr is not None:
            return {"image_logs": image_logs, "ssim": avg_ssim, "psnr": avg_psnr}
        return image_logs

    def save_ckpt(self, is_best: bool = False, non_blocking: bool = True) -> None:
        unwrapped = self.unwrap_fn(self.transformer)

        filename = "best_model" if is_best else f"step_{self.global_step}"
        save_path = os.path.join(self.ckpt_dir, filename)

        class_emb_state = unwrapped.class_embedding_U.state_dict()

        os.makedirs(save_path, exist_ok=True)

        def _do_save():
            class_emb_path = os.path.join(save_path, "class_embedding_U.pt")
            torch.save(class_emb_state, class_emb_path)

        if non_blocking:
            import threading
            t = threading.Thread(target=_do_save, name=f"ckpt-save-{self.global_step}")
            t.start()
        else:
            _do_save()

    def train(self) -> None:
        self.setup_dist()
        self.init_logger()
        self.build_models()
        self.build_dataloader()
        self.setup_optimization()

        self.transformer.train()

        num_epochs = self.config.train.num_train_epochs
        steps_per_epoch = len(self.train_dataloader)
        total_steps = num_epochs * steps_per_epoch

        val_freq_iters = self.config.val.get("val_freq_iters", 1000)
        save_freq_iters = self.config.train.get("save_freq_iters", 5000)

        if self.is_main:
            pbar = tqdm(
                total=total_steps,
                disable=not self.is_main,
                desc="Training",
            )

        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1

            for batch in self.train_dataloader:
                loss = self.training_step(batch)

                if self.is_main:
                    pbar.set_postfix(
                        {
                            "iter": self.global_step,
                            "loss": f"{loss.item():.4f}",
                            "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                        }
                    )
                    pbar.update(1)

                # Validation based on iter frequency
                if (
                    self.val_dataloader is not None
                    and self.global_step % val_freq_iters == 0
                    and self.global_step > 0
                ):
                    self.transformer.eval()
                    image_logs = self.validation()
                    self.transformer.train()

                    if self.is_main and image_logs:
                        print("\n" + "=" * 60)
                        print(f"Validation @ iter {self.global_step}")
                        print("=" * 60)
                        if image_logs.get("ssim") is not None:
                            print(f"SSIM: {image_logs['ssim']:.4f} | PSNR: {image_logs['psnr']:.2f}")
                        print("=" * 60)

                # Save checkpoint based on iter frequency
                if (
                    self.is_main
                    and self.global_step % save_freq_iters == 0
                    and self.global_step > 0
                ):
                    self.save_ckpt()

        if self.is_main:
            pbar.close()
            self.save_ckpt(is_best=True)

        self.accelerator.end_training()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    trainer = ConditionTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
