import argparse
from copy import copy
import logging
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import diffusers
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKLFlux2,
    FlowMatchEulerDiscreteScheduler,
    Flux2KleinPipeline,
    Flux2Transformer2DModel,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import is_compiled_module
from omegaconf import OmegaConf
from PIL import Image
from src.data.dataset import PairedDataset2
from src.networks.degnet import DegNet_DINO
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics.functional import peak_signal_noise_ratio as psnr_fn
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
from tqdm import tqdm

@torch.no_grad()
def encode_images(pixels: torch.Tensor, vae: nn.Module, weight_dtype: torch.dtype) -> torch.Tensor:
    pixel_latents = vae.encode(pixels).latent_dist.sample()
    return pixel_latents.to(weight_dtype)

class Trainer:
    def __init__(self, args: argparse.Namespace) -> None:

        self.args = args
        self.config = OmegaConf.load(args.config)
        self.config_path = args.config 
        self.exp_name = f"{self.config.get('name', 'flux2_condition_finetune')}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        set_seed(self.config.seed)

        self.num_classes: int = self.config.network.get("num_classes", 4)
        self.deg_names: List[str] = self.config.network.get("deg_names", ["Clean", "Haze", "Rain", "Lowlight"])
        
        self.global_step: int = 0
        self.global_iter: int = 0  
        self.is_main: bool = False
        self.accelerator=None

        # bulid model
        self.vae=None
        self.transformer=None
        self.noise_scheduler=None
        self.optimizer=None
        self.scheduler=None
        self.weight_dtype=None
        self.deg_classifier=None

        self.train_dataloader=None
        self.val_dataloader=None
        self.train_datasets=[]
        self.val_datasets=[]
        self.prompt_embed_cache={}

        self.best_metrics={}

        self.deg_classifier = DegNet_DINO(
            dino_type=self.config.network.get("dino_type", None),
            num_types=self.num_classes
        )
        self.deg_classifier.load_state_dict(
            torch.load(self.config.network.degradation_classifier_path, map_location="cpu"),
            strict=False
        )

        self.output_dir=None
        self.ckpt_dir=None

    def setup_dist(self) -> None:
        # setup the distributed data parallel
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=self.config.accelerator.get("find_unused_parameters", False)
        )

        accelerator_project_config = ProjectConfiguration(project_dir=self.config.accelerator.project_dir)
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.accelerator.get("gradient_accumulation_steps", 8),
            mixed_precision=self.config.accelerator.get("mixed_precision", "bf16"),
            log_with=self.config.accelerator.get("log_with", "swanlab"),
            project_config=accelerator_project_config,
            kwargs_handlers=[ddp_kwargs],
        )

        self.is_main = self.accelerator.is_main_process

        if self.config.accelerator.get("allow_tf32", False):
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True

    def _unwrap_transformer(self):
        model = self.accelerator.unwrap_model(self.transformer)
        return model._orig_mod if is_compiled_module(model) else model

    def init_logger(self) -> None:
        self.output_dir = os.path.join(self.config.accelerator.project_dir, self.exp_name)
        self.ckpt_dir = os.path.join(self.output_dir, "checkpoints")

        if self.is_main:
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(self.ckpt_dir, exist_ok=True)
            OmegaConf.save(self.config, os.path.join(self.output_dir, "config.yaml"))
            shutil.copy2(os.path.abspath(__file__), os.path.join(self.output_dir, os.path.basename(__file__)))

        tracker_config = vars(self.config).copy()
        self.accelerator.init_trackers(
            self.exp_name,
            config=tracker_config,
        )

    def build_models(self) -> None:
        # load the model from the model_path
        model_path = self.config.network.get("model_path", None)
        if model_path is None:
            raise ValueError("model_path is required")
            
        self.weight_dtype = self.config.accelerator.get("mixed_precision", "bf16")
        if self.weight_dtype == "bf16":
            self.weight_dtype = torch.bfloat16
        elif self.weight_dtype == "fp16":
            self.weight_dtype = torch.float16
        else:
            self.weight_dtype = torch.float32
        
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path,subfolder="scheduler",
            torch_dtype=self.weight_dtype,
        )
        noise_scheduler_copy = copy.deepcopy(noise_scheduler)
        self.noise_scheduler = noise_scheduler_copy

        # load vae with float32 (recommended)
        self.vae = AutoencoderKLFlux2.from_pretrained(
            model_path, subfolder="vae",
            torch_dtype=torch.float32,
        )
        self.vae.requires_grad_(False)
        # VAE BN statistics for latent normalization (used in training and inference)
        self.hq_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(self.accelerator.device)
        self.hq_bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
        ).to(self.accelerator.device)

        self.lpips_loss = lpips.LPIPS(net="vgg").to(self.accelerator.device)
        self.lpips_loss.requires_grad_(False)

        self.transformer = Flux2Transformer2DModel.from_pretrained(
            model_path, subfolder="transformer",
            torch_dtype=self.weight_dtype,
        )
        self.transformer.requires_grad_(False)

        inner_dim = self.transformer.inner_dim

        self.transformer.register_parameter(
            "class_embedding_U",
            nn.Parameter(torch.randn(self.num_classes, inner_dim), requires_grad=True)
        )
        nn.init.orthogonal_(self.transformer.class_embedding_U)

        # move to cpu to prevent problems when training on multiple GPUs
        self._class_embedding_U_cpu = self.transformer.class_embedding_U.data.clone().cpu()

        unfreeze_type = self.config.network.get("unfreeze_adaln_type", "all")

        ds_img_lin = self.transformer.double_stream_modulation_img.linear
        ds_txt_lin = self.transformer.double_stream_modulation_txt.linear
        ss_lin     = self.transformer.single_stream_modulation.linear

        ds_img_lin.requires_grad_(False)
        ds_txt_lin.requires_grad_(False)
        ss_lin.requires_grad_(False)

        if unfreeze_type in ("double", "all"):
            ds_img_lin.requires_grad_(True)
            ds_txt_lin.requires_grad_(True)
        if unfreeze_type in ("single", "all"):
            ss_lin.requires_grad_(True)

        def _is_on(lin):
            return lin.weight.requires_grad

        def _param_count(lin):
            n = lin.weight.numel()
            if lin.bias is not None:
                n += lin.bias.numel()
            return n

        total_trainable = 0
        if _is_on(ds_img_lin):
            total_trainable += _param_count(ds_img_lin)
        if _is_on(ds_txt_lin):
            total_trainable += _param_count(ds_txt_lin)
        if _is_on(ss_lin):
            total_trainable += _param_count(ss_lin)   

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler",torch_dtype=self.weight_dtype)
        self.fixed_timestep = self.config.train.get("fixed_timestep", None)

    def build_dataloader(self) -> None:
        resolution = self.config.data.resolution
        datasets_cfg = self.config.data.get("datasets", {})

        train_datasets, val_datasets = [], []

        for ds_idx, (ds_name, ds_cfg) in enumerate(datasets_cfg.items()):
            is_val = ds_name.startswith("ValDataset")
            dataset = PairedDataset2(
                lq_path=ds_cfg.lq_path, 
                hq_path=ds_cfg.hq_path,
                resolution=resolution, 
                prompt=ds_cfg.get("prompt", ""),
                dataset_idx=ds_idx, 
                enlarge_ratio=ds_cfg.get("enlarge_ratio", 1.0),
            )

            (val_datasets if is_val else train_datasets).append(dataset)

        self.train_datasets = train_datasets
        self.val_datasets = val_datasets

        train_cfg = self.config.data.dataloader.train
        train_combined_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
        self.train_dataloader = DataLoader(
            train_combined_dataset,
            shuffle=True,
            collate_fn=self._collate_fn,
            batch_size=train_cfg.get("batch_size", 4),
            num_workers=train_cfg.get("num_workers", 2),
            pin_memory=train_cfg.get("pin_memory", True),
            persistent_workers=train_cfg.get("persistent_workers", True) if train_cfg.get("num_workers", 2) > 0 else False,
            drop_last=train_cfg.get("drop_last", True),
            prefetch_factor=train_cfg.get("prefetch_factor", 2) if train_cfg.get("num_workers", 2) > 0 else None,
        )

        val_cfg = self.config.data.dataloader.val
        self.val_dataloader = [
            DataLoader(
                ds,
                shuffle=False,
                collate_fn=self._collate_fn,
                batch_size=val_cfg.get("batch_size", 4),
                num_workers=val_cfg.get("num_workers", 2),
                pin_memory=val_cfg.get("pin_memory", True),
                persistent_workers=val_cfg.get("persistent_workers", True) if val_cfg.get("num_workers", 2) > 0 else False,
                drop_last=False,
                prefetch_factor=val_cfg.get("prefetch_factor", 2) if val_cfg.get("num_workers", 2) > 0 else None,
            ) for ds in val_datasets
        ]

    @staticmethod
    def _collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "lq_pixel_values": torch.stack([e["lq_pixel_values"] for e in examples]),
            "hq_pixel_values": torch.stack([e["hq_pixel_values"] for e in examples]),
            "dataset_indices": torch.tensor([e["dataset_idx"] for e in examples], dtype=torch.long),
        }

    def extract_deg_feat(self, lq_images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.deg_classifier(lq_images)
            probs = F.softmax(logits, dim=-1)[:, :, 0]
        deg_feat = probs.detach().to(dtype=self.weight_dtype) @ self._class_embedding_U_full
        return deg_feat
    def resume_checkpoint(self) -> None:
        # provide exp and ckpt path to resume
        resume_exp_dir = self.config.train.get("resume_exp_dir", None)
        resume_ckpt = self.config.train.get("resume_from", None)
        
        if not resume_ckpt or not resume_exp_dir:
            self.global_step = 0
            self.global_iter = 0
            return
        
        ckpt_path = os.path.join(resume_exp_dir, "checkpoints", resume_ckpt)
        if not os.path.isdir(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        metadata_path = os.path.join(ckpt_path, "metadata.pt")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        metadata = torch.load(metadata_path, map_location="cpu")
        self.global_step = metadata.get("global_step")
        self.global_iter = metadata.get("global_iter")

        self.load_ckpt(ckpt_path)

        if self.is_main:
            print(f"Resumed from checkpoint: {ckpt_path}")
            print(f"  Global step: {self.global_step}")
            print(f"  Global iter: {self.global_iter}")
    def setup_optimization(self) -> None:
        decay_params, no_decay_params = [], []
        for name, param in self.transformer.named_parameters():
            if not param.requires_grad:
                continue
            (decay_params if "class_embedding_U" in name else no_decay_params).append(param)

        adaln_names = {name for name, p in self.transformer.named_parameters() if ".linear." in name and p.requires_grad}
        mlp_names = {name for name, p in self.transformer.named_parameters() if p.requires_grad and "class_embedding_U" not in name and name not in adaln_names}

        adaln_count = len(adaln_names)
        mlp_count = len(mlp_names)
        self.accelerator.print(
            f"[Params] class_embedding_U: {len(decay_params)}, AdaLN Linear: {adaln_count}, MLP: {mlp_count}"
        )

        opt_cfg = self.config.optim
        self.optimizer = torch.optim.AdamW(
            [{"params": decay_params, "weight_decay": opt_cfg.get("weight_decay", 0.01)},
             {"params": no_decay_params, "weight_decay": 0.0}],
            lr=opt_cfg.get("lr", 1e-4),
            betas=(opt_cfg.get("beta1", 0.9), opt_cfg.get("beta2", 0.999)),
            eps=opt_cfg.get("epsilon", 1e-8),
            fused=True,
        )

        scheduler_cfg = self.config.scheduler
        num_train_iters = self.config.train.num_train_iters
        
        self.scheduler = get_scheduler(
            scheduler_cfg.type,
            optimizer=self.optimizer,
            num_warmup_steps=scheduler_cfg.get("lr_warmup_steps", 100),
            num_training_steps=num_train_iters,
            num_cycles=scheduler_cfg.get("lr_num_cycles", 1),
        )

        self.transformer, self.optimizer, self.train_dataloader, self.scheduler = (
            self.accelerator.prepare(
                self.transformer, self.optimizer, self.train_dataloader, self.scheduler
            )
        )

        self._class_embedding_U_full = self._class_embedding_U_cpu.to(
            self.accelerator.device, dtype=self.weight_dtype
        )

        # Move VAE BN stats to accelerator device (used in training_step)
        self.hq_bn_mean = self.hq_bn_mean.to(self.accelerator.device, dtype=torch.float32)
        self.hq_bn_std = self.hq_bn_std.to(self.accelerator.device, dtype=torch.float32)

        if not self.config.network.get("offload", False):
            self.vae = self.vae.to(self.accelerator.device)

        self.deg_classifier.to(self.accelerator.device, dtype=self.weight_dtype).eval()

        embed_dir = self.config.data.get("embed_dir", "./cached_embeddings")
        for ds in self.train_datasets + self.val_datasets:
            path = os.path.join(embed_dir, f"dataset_{ds.dataset_idx}_embeds.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(f"找不到 embedding 文件: {path}")
            data = torch.load(path, map_location="cpu", weights_only=True)
            self.prompt_embed_cache[ds.dataset_idx] = (
                data["prompt_embeds"],   
                data["text_ids"],        
            )

        free_memory()

    def _get_prompt_embeds_from_cache(self, dataset_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.accelerator.device
        prompt_embeds = torch.cat([
            self.prompt_embed_cache[int(i)][0].to(device, dtype=self.weight_dtype)
            for i in dataset_indices.tolist()
        ], dim=0)
        text_ids = self.prompt_embed_cache[int(dataset_indices[0].item())][1].to(device, dtype=self.weight_dtype)
        return prompt_embeds, text_ids

    def _predict_x0(
        self,
        lq_pixel_values: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        dataset_indices: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict x0 from LQ pixels (inference helper, not used in current training pipeline)."""
        lq_latents = encode_images(lq_pixel_values, self.vae, self.weight_dtype)
        batchsize = lq_latents.shape[0]

        # FLUX.2: patchify + BN normalization
        lq_latents_patched = Flux2KleinPipeline._patchify_latents(lq_latents)
        lq_latents_normed = (lq_latents_patched - self.hq_bn_mean) / self.hq_bn_std

        sigmas = self._get_sigmas(timesteps, lq_latents.ndim, lq_latents.dtype)
        noisy_input_normed = (1.0 - sigmas) * lq_latents_normed + sigmas * noise

        lq_latents_unpatched = Flux2KleinPipeline._unpatchify_latents(noisy_input_normed)
        packed_input = Flux2KleinPipeline._pack_latents(noisy_input_normed)
        latent_image_ids = Flux2KleinPipeline._prepare_latent_ids(lq_latents_unpatched)

        prompt_embeds, text_ids = self._get_prompt_embeds_from_cache(dataset_indices)
        deg_feat = self.extract_deg_feat(lq_pixel_values)

        model_pred_raw = self._unwrap_transformer()(
            hidden_states=packed_input,
            timestep=timesteps.float() / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            deg_emb=deg_feat,
            return_dict=False,
        )[0]

        model_pred_unpacked = Flux2KleinPipeline._unpack_latents_with_ids(model_pred_raw, latent_image_ids)
        x0_normed = noisy_input_normed - sigmas * model_pred_unpacked
        x0_denormed = x0_normed * self.hq_bn_std + self.hq_bn_mean
        return model_pred_unpacked, Flux2KleinPipeline._unpatchify_latents(x0_denormed)

    def training_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        with self.accelerator.accumulate(self.transformer):
            if self.config.network.get("offload", False):
                self.vae.to(self.accelerator.device)

            # Encode HQ to VAE latent space
            hq_latents = encode_images(batch["hq_pixel_values"], self.vae, self.weight_dtype)
            # Also encode LQ (needed for deg_feat extraction)
            encode_images(batch["lq_pixel_values"], self.vae, self.weight_dtype)

            if self.config.network.get("offload", False):
                self.vae.cpu()
                free_memory()

            batchsize = hq_latents.shape[0]

            # FLUX.2: patchify + BN normalization before DiT
            hq_latents_patched = Flux2KleinPipeline._patchify_latents(hq_latents)
            hq_latents_normed = (hq_latents_patched - self.hq_bn_mean) / self.hq_bn_std

            # Generate noise and sample timestep
            noise = torch.randn_like(hq_latents_patched)  # match patched shape (B, 512, 32, 32)

            if self.fixed_timestep is not None:
                timesteps = torch.full((batchsize,), self.fixed_timestep, device=hq_latents.device, dtype=hq_latents.dtype)
                sigmas = self._get_sigmas(timesteps, hq_latents.ndim, hq_latents.dtype)
            else:
                density = compute_density_for_timestep_sampling(
                    weighting_scheme=self.config.loss.get("weighting_scheme", "none"),
                    batch_size=batchsize,
                    logit_mean=self.config.loss.get("logit_mean", 0.0),
                    logit_std=self.config.loss.get("logit_std", 1.0),
                    mode_scale=self.config.loss.get("mode_scale", 1.29),
                )
                indices = (density * self.noise_scheduler.config.num_train_timesteps).long()
                timesteps = self.noise_scheduler.timesteps[indices].to(device=hq_latents.device, dtype=hq_latents.dtype)
                sigmas = self._get_sigmas(timesteps, hq_latents.ndim, hq_latents.dtype)

            # Noisy input (normalized space)
            noisy_input_normed = (1.0 - sigmas) * hq_latents_normed + sigmas * noise

            # Pack for DiT
            hq_latents_unpatched = Flux2KleinPipeline._unpatchify_latents(noisy_input_normed)
            packed_input = Flux2KleinPipeline._pack_latents(noisy_input_normed)
            latent_image_ids = Flux2KleinPipeline._prepare_latent_ids(hq_latents_unpatched)

            prompt_embeds, text_ids = self._get_prompt_embeds_from_cache(batch["dataset_indices"])
            if self.config.train.get("proportion_empty_prompts", 0) > 0 and torch.rand(1).item() < self.config.train.proportion_empty_prompts:
                prompt_embeds.zero_()

            guidance = None
            if self._unwrap_transformer().config.guidance_embeds:
                guidance = torch.full((batchsize,), self.config.train.get("guidance_scale", 3.0), device=noisy_input_normed.device, dtype=self.weight_dtype)

            deg_feat = self.extract_deg_feat(batch["lq_pixel_values"].to(self.accelerator.device))

            model_pred = self._unwrap_transformer()(
                hidden_states=packed_input,
                timestep=timesteps / 1000,
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                deg_emb=deg_feat,
                return_dict=False,
            )[0]

            weighting = compute_loss_weighting_for_sd3(
                weighting_scheme=self.config.loss.get("weighting_scheme", "none"),
                sigmas=sigmas,
            )
            model_pred_unpacked = Flux2KleinPipeline._unpack_latents_with_ids(model_pred, latent_image_ids)

            # Target in normalized, patched, unpacked space
            target = noise - hq_latents_normed
            loss = torch.mean(
                (weighting.float() * (model_pred_unpacked.float() - target.float()) ** 2).reshape(batchsize, -1)
            ).mean()

            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.transformer.parameters(), self.config.train.get("max_grad_norm", 1.0))

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            return loss

    def _get_sigmas(self, timesteps: torch.Tensor, n_dim: int, dtype: torch.dtype) -> torch.Tensor:
        sigmas = self.noise_scheduler.sigmas.to(device=self.accelerator.device, dtype=dtype)
        schedule = self.noise_scheduler.timesteps.to(device=self.accelerator.device, dtype=timesteps.dtype)
        sigma = sigmas[torch.searchsorted(schedule, timesteps)].flatten()
        while sigma.ndim < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma


    @torch.no_grad()
    def validation(self) -> Optional[Dict[str, Any]]:
        if self.config.network.get("offload", False):
            self.vae.to(self.accelerator.device)

        guidance_scale = self.config.val.get("guidance_scale", 3.0)
        fixed_val_timestep = self.config.val.get("fixed_val_timestep", 300)
        num_vis_samples = self.config.val.get("num_vis_samples", 4)
        max_val_samples = self.config.val.get("max_val_samples", None)

        vis_candidates: List[Dict[str, Any]] = []
        seen_datasets = set()

        val_dataset_indices = set(ds.dataset_idx for ds in self.val_datasets)
        val_metrics: Dict[int, Dict[str, float]] = {}
        for ds_idx in val_dataset_indices:
            val_metrics[ds_idx] = {"ssim": 0.0, "psnr": 0.0, "count": 0}

        if self.is_main:
            vis_dir = os.path.join(self.output_dir, f"val_vis_iter_{self.global_iter}")
            os.makedirs(vis_dir, exist_ok=True)

        total_val_batches = sum(len(dl) for dl in self.val_dataloader)
        if max_val_samples:
            total_val_batches = min(total_val_batches, max_val_samples)

        val_pbar = tqdm(total=int(total_val_batches), desc=f"Validation @ Iter {self.global_iter}",
                        disable=not self.is_main, leave=False)

        batch_idx = 0
        max_val_reached = False
        for dataloader in self.val_dataloader:
            for batch in dataloader:
                if max_val_samples and batch_idx >= max_val_samples:
                    max_val_reached = True
                    break
                    
                hq_pixel_values = batch["hq_pixel_values"].to(self.accelerator.device)
                lq_pixel_values = batch["lq_pixel_values"].to(self.accelerator.device)
                dataset_indices = batch["dataset_indices"]
                batchsize = lq_pixel_values.shape[0]

                for i in range(batchsize):
                    ds_idx = int(dataset_indices[i].item())
                    if ds_idx not in seen_datasets and len(vis_candidates) < num_vis_samples * len(self.val_datasets):
                        seen_datasets.add(ds_idx)
                        vis_candidates.append({"batch_idx": batch_idx, "batch_idx_in_batch": i, "ds_idx": ds_idx})

                lq_latents = encode_images(lq_pixel_values, self.vae, self.weight_dtype)

                # FLUX.2: patchify + BN normalization before DiT
                lq_latents_patched = Flux2KleinPipeline._patchify_latents(lq_latents)
                lq_latents_normed = (lq_latents_patched - self.hq_bn_mean) / self.hq_bn_std

                noise = torch.randn_like(lq_latents_patched)  # match patched shape

                timesteps = torch.full((batchsize,), fixed_val_timestep, device=lq_latents.device, dtype=self.weight_dtype)
                sigmas = self._get_sigmas(timesteps, lq_latents.ndim, lq_latents.dtype)

                noisy_input_normed = (1.0 - sigmas) * lq_latents_normed + sigmas * noise

                lq_latents_unpatched = Flux2KleinPipeline._unpatchify_latents(noisy_input_normed)
                packed_input = Flux2KleinPipeline._pack_latents(noisy_input_normed)
                latent_image_ids = Flux2KleinPipeline._prepare_latent_ids(lq_latents_unpatched)

                prompt_embeds, text_ids = self._get_prompt_embeds_from_cache(batch["dataset_indices"])

                guidance = None
                if self._unwrap_transformer().config.guidance_embeds:
                    guidance = torch.full((batchsize,), guidance_scale, device=lq_latents.device, dtype=self.weight_dtype)

                deg_feat = self.extract_deg_feat(lq_pixel_values)

                model_pred = self._unwrap_transformer()(
                    hidden_states=packed_input,
                    timestep=timesteps.float() / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    deg_emb=deg_feat,
                    return_dict=False,
                )[0]

                model_pred_unpacked = Flux2KleinPipeline._unpack_latents_with_ids(model_pred, latent_image_ids)

                # Recover denoised x0 in normalized space, then denormalize + unpatchify
                x0_normed = noisy_input_normed - sigmas * model_pred_unpacked
                x0_denormed = x0_normed * self.hq_bn_std + self.hq_bn_mean
                latents_to_decode = Flux2KleinPipeline._unpatchify_latents(x0_denormed)

                if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor is not None:
                    latents_to_decode = (latents_to_decode / self.vae.config.scaling_factor) + self.vae.config.shift_factor

                generated = self.vae.decode(latents_to_decode.to(dtype=torch.float32)).sample
                
                pred_tensor = (generated / 2 + 0.5).clamp(0, 1).to(self.accelerator.device)
                gt_tensor = (hq_pixel_values + 1.0) / 2.0

                for i in range(batchsize):
                    ds_idx = int(dataset_indices[i].item())
                    if ds_idx not in val_dataset_indices:
                        continue
                    ssim_val = ssim_fn(pred_tensor[i:i+1], gt_tensor[i:i+1], data_range=1.0)
                    psnr_val = psnr_fn(pred_tensor[i:i+1], gt_tensor[i:i+1], data_range=1.0)
                    val_metrics[ds_idx]["ssim"] += ssim_val.item()
                    val_metrics[ds_idx]["psnr"] += psnr_val.item()
                    val_metrics[ds_idx]["count"] += 1

                is_vis = any(c["batch_idx"] == batch_idx for c in vis_candidates)
                if self.is_main and is_vis:
                    for cand in [c for c in vis_candidates if c["batch_idx"] == batch_idx]:
                        i = cand["batch_idx_in_batch"]
                        if i < batchsize:
                            ds_idx = cand["ds_idx"]
                            ds_name = list(self.config.data.datasets.keys())[ds_idx]
                            gen_img_np = pred_tensor[i].cpu().float().permute(1, 2, 0).numpy()
                            lq_img_np = lq_pixel_values[i].cpu().float().permute(1, 2, 0).numpy()
                            gt_img_np = gt_tensor[i].cpu().float().permute(1, 2, 0).numpy()
                            generated_pil = Image.fromarray((gen_img_np * 255).astype(np.uint8)).convert("RGB")
                            lq_pil = Image.fromarray(((lq_img_np + 1) * 127.5).astype(np.uint8)).convert("RGB")
                            gt_pil = Image.fromarray((gt_img_np * 255).astype(np.uint8)).convert("RGB")
                            vis_idx = cand["batch_idx"]
                            generated_pil.save(os.path.join(vis_dir, f"{ds_name}_pred_{vis_idx}.png"))
                            lq_pil.save(os.path.join(vis_dir, f"{ds_name}_LQ_{vis_idx}.png"))
                            gt_pil.save(os.path.join(vis_dir, f"{ds_name}_HQ_{vis_idx}.png"))
                            generated_pil.close()
                            lq_pil.close()
                            gt_pil.close()
                            del generated_pil, lq_pil, gt_pil

                batch_idx += 1
                val_pbar.update(1)
                if max_val_reached:
                    break

        val_pbar.close()

        if self.config.network.get("offload", False):
            self.vae.cpu()
            free_memory()

        dataset_results = {}
        for ds_idx, metrics in val_metrics.items():
            if metrics["count"] > 0:
                ds_name = list(self.config.data.datasets.keys())[ds_idx]
                dataset_results[ds_name] = {
                    "ssim": metrics["ssim"] / metrics["count"],
                    "psnr": metrics["psnr"] / metrics["count"],
                }

        return {"dataset_results": dataset_results, "vis_dir": vis_dir if self.is_main else None}


    def save_ckpt(self, is_best: bool = False, non_blocking: bool = True) -> None:
        unwrapped = self._unwrap_transformer()
        
        if is_best:
            save_path = os.path.join(self.ckpt_dir, "best_model")
        else:
            save_path = os.path.join(self.ckpt_dir, f"iter_{self.global_iter}")
        
        os.makedirs(save_path, exist_ok=True)
        
        trainable_params = {}
        for name, param in unwrapped.named_parameters():
            if param.requires_grad:
                trainable_params[name] = param.detach().clone()
        
        extra_state = {}
        for attr_name in ["class_embedding_U"]:
            if hasattr(unwrapped, attr_name):
                attr = getattr(unwrapped, attr_name)
                if isinstance(attr, torch.Tensor):
                    extra_state[attr_name] = attr.data.detach().clone()
        
        metadata = {
            "global_step": self.global_step,
            "global_iter": self.global_iter,
            "trainable_param_count": len(trainable_params),
        }
        
        def _do_save():
            try:
                if extra_state:
                    torch.save(extra_state, os.path.join(save_path, "extra_state.pt"))
                torch.save(trainable_params, os.path.join(save_path, "trainable_params.pt"))
                torch.save(metadata, os.path.join(save_path, "metadata.pt"))
                if self.scheduler is not None:
                    torch.save(self.scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))
                if self.is_main:
                    tag = "best" if is_best else f"iter {self.global_iter}"
                    print(f"Saved checkpoint [{tag}] to {save_path}")
                    print(f"   - trainable_params.pt ({len(trainable_params)} params)")
            except Exception as e:
                if self.is_main:
                    print(f"Failed to save checkpoint: {e}")
        
        if non_blocking:
            import threading
            threading.Thread(target=_do_save, name=f"ckpt-{self.global_iter}").start()
        else:
            _do_save()
    
    def load_ckpt(self, ckpt_path: str) -> None:
        unwrapped = self._unwrap_transformer()
        trainable_state = torch.load(os.path.join(ckpt_path, "trainable_params.pt"), map_location="cpu")
        missing, unexpected = unwrapped.load_state_dict(trainable_state, strict=False)
        if self.is_main:
            print(f"Loaded {len(trainable_state)} trainable params from {ckpt_path}")
            if missing:
                print(f"  Missing keys (expected): {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")

        extra_state_path = os.path.join(ckpt_path, "extra_state.pt")
        if os.path.exists(extra_state_path):
            extra_state = torch.load(extra_state_path, map_location="cpu")
            for attr_name, tensor in extra_state.items():
                if hasattr(unwrapped, attr_name):
                    getattr(unwrapped, attr_name).data.copy_(tensor)

        scheduler_state_path = os.path.join(ckpt_path, "scheduler.pt")
        if os.path.exists(scheduler_state_path) and self.scheduler is not None:
            self.scheduler.load_state_dict(torch.load(scheduler_state_path, map_location="cpu"))
            if self.is_main:
                print("  Restored scheduler state")

    def train(self) -> None:
        self.setup_dist()
        self.init_logger()
        self.build_models()
        self.build_dataloader()
        self.setup_optimization()
        self.resume_checkpoint()

        num_train_iters = self.config.train.num_train_iters
        resume_iter = self.global_iter

        if self.is_main:
            pbar = tqdm(total=num_train_iters, initial=resume_iter, desc="Training", unit="iter")
        
        # zero-shot validation as baseline
        if self.val_dataloader and not self.config.val.get("skip_init_val", False):
            self.transformer.eval()
            val_results = self.validation()
            
            if self.is_main and val_results is not None:
                dataset_results = val_results.get("dataset_results", {})
                for ds_name, metrics in dataset_results.items():
                    self.best_metrics[f"{ds_name}_ssim"] = metrics["ssim"]
                    self.best_metrics[f"{ds_name}_psnr"] = metrics["psnr"]
                
                for ds_name, metrics in dataset_results.items():
                    print(f"{ds_name} → SSIM: {metrics['ssim']:.4f} | PSNR: {metrics['psnr']:.2f}")
                print("=" * 60)

        # Training loop!
        self.transformer.train()

        free_memory()
        torch.cuda.reset_peak_memory_stats()

        val_freq_iters = self.config.val.get("val_freq_iters", 10000)
        save_freq_iters = self.config.val.get("save_freq_iters", 10000)
        last_val_iter = 0

        while self.global_iter < num_train_iters:
            
            for batch in self.train_dataloader:
                self.global_iter += 1
                
                loss = self.training_step(batch)
                
                if self.accelerator.sync_gradients:
                    self.global_step += 1  
                    if self.is_main:
                        pbar.set_postfix({
                            "step": self.global_step,
                            "iter": self.global_iter,      
                            "loss": f"{loss.item():.4f}", 
                            "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"
                        })
                        pbar.update(1)
                        
                    if self.global_iter > 0 and self.global_iter % self.config.logger.get("log_interval", 10) == 0:
                            self.accelerator.log(
                                {"train/loss": loss.item(), "train/lr": self.optimizer.param_groups[0]["lr"], "train/iter": self.global_iter},
                                step=self.global_iter
                            )

                    if save_freq_iters and self.global_iter % save_freq_iters == 0 and self.is_main:
                        self.save_ckpt()

                if self.global_iter >= num_train_iters:
                    break
                
                if self.val_dataloader and self.global_iter - last_val_iter >= val_freq_iters:
                    last_val_iter = self.global_iter
                    
                    free_memory()

                    self.transformer.eval()
                    val_results = self.validation()
                    self.transformer.train()
                    
                    if self.is_main and val_results is not None:
                        dataset_results = val_results.get("dataset_results", {})
                        is_best = False
                        
                        for ds_name, metrics in dataset_results.items():
                            ssim_key = f"{ds_name}_ssim"
                            psnr_key = f"{ds_name}_psnr"
                            
                            curr_ssim = metrics["ssim"]
                            curr_psnr = metrics["psnr"]
                            
                            if ssim_key not in self.best_metrics or curr_ssim > self.best_metrics.get(ssim_key, -1.0):
                                self.best_metrics[ssim_key] = curr_ssim
                                is_best = True
                            if psnr_key not in self.best_metrics or curr_psnr > self.best_metrics.get(psnr_key, -1.0):
                                self.best_metrics[psnr_key] = curr_psnr
                                is_best = True
                            
                            print(f"{ds_name} → SSIM: {curr_ssim:.4f} | PSNR: {curr_psnr:.2f}")
                            
                        if is_best:
                            self.save_ckpt(is_best=True)
                            print("New best model saved!")
                            
                        print(f"\n{'='*60}")
                        print(f"Validation @ Iter {self.global_iter}")
                        print(f"{'='*60}")
                        print("Best Metrics:")
                        if self.best_metrics:
                            for key, val in self.best_metrics.items():
                                print(f"  {key}: {val:.4f}" if "ssim" in key else f"  {key}: {val:.2f}")
                        else:
                            print("  (No best metrics yet)")
                        print("=" * 60)

        if self.is_main:
            pbar.close()
            self.save_ckpt()
        self.accelerator.end_training()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    trainer = Trainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
