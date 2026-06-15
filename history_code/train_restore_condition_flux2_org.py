import argparse
import gc
import logging
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# set if memory is not enough
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import swanlab
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from diffusers import (
    AutoencoderKLFlux2,
    FlowMatchEulerDiscreteScheduler,
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
    pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
    return pixel_latents.to(weight_dtype)

def _prepare_latent_image_ids(latents: torch.Tensor) -> torch.Tensor:
    # image ids: the index of the image in the batch
    batch_size, _, height, width = latents.shape
    device = latents.device
    
    t = torch.arange(1, device=device)
    h = torch.arange(height // 2, device=device)
    w = torch.arange(width // 2, device=device)
    layer = torch.arange(1, device=device)

    latent_ids = torch.cartesian_prod(t, h, w, layer)
    return latent_ids.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

def _pack_latents(latents: torch.Tensor, batch_size: int, height: int, width: int) -> torch.Tensor:
    # Notice : height and width should be divisible by 2 before packing!
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

def _flux2_transformer_forward():
    # rewrite the forward method of Flux2Transformer2DModel
    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        deg_emb: Optional[torch.Tensor] = None,
    ):
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if hasattr(self, "_peft_backend") and self._peft_backend:
            scale_lora_layers(self, lora_scale)

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
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states,
                    double_stream_mod_img, double_stream_mod_txt,
                    concat_rotary_emb, joint_attention_kwargs,
                )
            else:
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
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, None,
                    single_stream_mod, concat_rotary_emb, joint_attention_kwargs,
                )
            else:
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

        if hasattr(self, "_peft_backend") and self._peft_backend:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    Flux2Transformer2DModel.forward = patched_forward
_flux2_transformer_forward()

class ConditionTrainer:
    def __init__(self, args: argparse.Namespace) -> None:

        self.args = args
        self.config = OmegaConf.load(args.config)
        self.config_path = args.config 
        self.exp_name = f"{self.config.get('name', 'flux2_condition_finetune')}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        set_seed(self.config.seed)

        self.num_classes: int = self.config.network.get("num_classes", 4)
        
        self.global_step: int = 0
        self.global_iter: int = 0  

        self.is_main: bool = False
        self.accelerator=None
        self.unwrap_fn= None

        self.vae=None
        self.transformer=None
        self.noise_scheduler=None
        self.optimizer=None
        self.scheduler=None
        self.train_dataloader=None
        self.val_dataloader=None
        self.weight_dtype=None

        self.train_datasets=[]
        self.val_datasets=[]
        self.prompt_embed_cache={}
        self._fixed_vis_samples=None

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
        self.log_dir=None
        self.logger=logging.getLogger(__name__)

    def setup_dist(self) -> None:
        # setup the distributed data parallel
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=self.config.accelerator.get("find_unused_parameters", False)
        )

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.train.get("gradient_accumulation_steps", 8),
            mixed_precision=self.config.accelerator.get("mixed_precision", "bf16"),
            log_with="swanlab",
            project_dir=self.config.accelerator.get("project_dir", "./experiments"),
            kwargs_handlers=[ddp_kwargs],
        )
        self.is_main = self.accelerator.is_main_process
        self.unwrap_fn = self.accelerator.unwrap_model

    def init_logger(self) -> None:
        # initialize the output directory, checkpoint directory and log directory
        self.output_dir = os.path.join("./experiments", self.exp_name)
        self.ckpt_dir = os.path.join(self.output_dir, "checkpoints")
        self.log_dir = os.path.join(self.output_dir, "logs")

        if self.is_main:
            os.makedirs(self.ckpt_dir, exist_ok=True)
            OmegaConf.save(self.config, os.path.join(self.output_dir, "config.yaml"))
            src_script = os.path.abspath(__file__)
            dst_script = os.path.join(self.output_dir, os.path.basename(src_script))
            shutil.copy2(src_script, dst_script)
            with open(os.path.join(self.output_dir, "train_command.sh"), "w") as f:
                f.write("#!/bin/bash\n")
                f.write(f"# Generated at {datetime.now().isoformat()}\n")
                f.write(f"# Config: {self.config_path}\n")
                f.write(f"python {src_script} --config {self.config_path}\n")

        swanlab_config = OmegaConf.to_container(self.config)
        swanlab_config["log_dir"] = self.log_dir
        self.accelerator.init_trackers(
            project_name=self.config.logging.swanlab_project,
            config=swanlab_config,
            init_kwargs={"swanlab": {"experiment_name": self.exp_name, "log_dir": self.log_dir}},
        )

    def resume_checkpoint(self) -> None:
        
        resume_dir = self.config.train.get("resume_from", None)
        
        if not resume_dir:
            self.global_step = 0
            self.global_iter = 0
            return

        resume_exp_dir = self.config.train.get("resume_exp_dir", None)
        if resume_exp_dir:
            project_dir = self.config.accelerator.get("project_dir", "./experiments")
            base_ckpt_path = os.path.join(project_dir, resume_exp_dir)
        else:
            base_ckpt_path = self.output_dir

        ckpt_path = os.path.join(base_ckpt_path, "checkpoints", resume_dir)
        if not os.path.isdir(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        metadata_path = os.path.join(ckpt_path, "metadata.pt")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        metadata = torch.load(metadata_path, map_location="cpu")

        self.global_step = metadata.get("global_step") or 0
        self.global_iter = metadata.get("global_iter") or 0

        legacy_epoch = metadata.get("epoch")
        self.load_ckpt(ckpt_path)

        if self.is_main:
            print(f"Resumed from checkpoint: {ckpt_path}")
            print(f"  Global step: {self.global_step}")
            print(f"  Global iter: {self.global_iter}")
            if legacy_epoch is not None:
                print(f"  (Legacy epoch field: {legacy_epoch}, ignored — now iter-based)")

    def build_models(self) -> None:
        # build the models
        model_path = self.config.network.pretrained_model_name_or_path
        precisions = {"fp16": torch.float16, "bf16": torch.bfloat16}
        self.weight_dtype = precisions.get(
            self.config.accelerator.get("mixed_precision", "bf16"), torch.float32
        )

        self.vae = AutoencoderKLFlux2.from_pretrained(
            model_path, subfolder="vae",
            revision=self.config.network.get("revision"),
            variant=self.config.network.get("variant"),
            torch_dtype=torch.float32,
        )
        self.vae.requires_grad_(False)
        #self.vae.enable_tiling()

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
        self._class_embedding_U_cpu = self.transformer.class_embedding_U.data.clone().cpu()

        unfreeze_type = self.config.network.get("unfreeze_adaln_type", "all")

        ds_img_lin = self.transformer.double_stream_modulation_img.linear
        ds_txt_lin = self.transformer.double_stream_modulation_txt.linear
        ss_lin     = self.transformer.single_stream_modulation.linear
        norm_lin   = self.transformer.norm_out.linear

        ds_img_lin.requires_grad_(False)
        ds_txt_lin.requires_grad_(False)
        ss_lin.requires_grad_(False)
        norm_lin.requires_grad_(False)

        if unfreeze_type in ("double", "all"):
            ds_img_lin.requires_grad_(True)
            ds_txt_lin.requires_grad_(True)
        if unfreeze_type in ("single", "all"):
            ss_lin.requires_grad_(True)
        if unfreeze_type == "all":
            norm_lin.requires_grad_(True)

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
        if _is_on(norm_lin):
            total_trainable += _param_count(norm_lin)
        total_params_mb = total_trainable * 2 / 1e6
        self.logger.info(
            f"[Modulation] unfreeze_type={unfreeze_type} | "
            f"double({'ON' if _is_on(ds_img_lin) else 'off'}/{'ON' if _is_on(ds_txt_lin) else 'off'}) "
            f"single({'ON' if _is_on(ss_lin) else 'off'}) "
            f"norm_out({'ON' if _is_on(norm_lin) else 'off'}) | "
            f"trainable={total_params_mb:.0f} MB ({total_trainable/1e6:.1f}M BF16)"
        )

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
        self.fixed_timestep = self.config.train.get("fixed_timestep", None)

        if self.config.network.get("gradient_checkpointing", False):
            self.transformer.enable_gradient_checkpointing()
        self.vae.to(dtype=torch.float32)
        self.transformer.to(dtype=self.weight_dtype)

        if self.config.accelerator.get("allow_tf32", False):
            torch.backends.cuda.matmul.allow_tf32 = True

    def extract_deg_feat(self, lq_images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.deg_classifier(lq_images)
            probs = F.softmax(logits, dim=-1)[:, :, 0]
        deg_feat = probs.detach().to(dtype=self.weight_dtype) @ self._class_embedding_U_full
        return deg_feat

    def build_dataloader(self) -> None:
        # bulid dataloader for training and validation
        resolution = self.config.data.resolution
        datasets_cfg = self.config.data.get("datasets", {})

        train_datasets, val_datasets = [], []
        for ds_idx, (ds_name, ds_cfg) in enumerate(datasets_cfg.items()):
            is_val = ds_name.startswith("ValDataset")
            dataset = PairedDataset2(
                lq_path=ds_cfg.lq_path, hq_path=ds_cfg.hq_path,
                resolution=resolution, prompt=ds_cfg.get("prompt", ""),
                dataset_idx=ds_idx, enlarge_ratio=ds_cfg.get("enlarge_ratio", 1.0),
            )
            (val_datasets if is_val else train_datasets).append(dataset)

        self.train_datasets = train_datasets
        self.val_datasets = val_datasets

        train_cfg = self.config.data.dataloader.train
        # concat the train datasets if there are multiple train datasets
        train_combined_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]

        self.train_dataloader = DataLoader(
            train_combined_dataset,
            shuffle=True, 
            collate_fn=self._collate_fn,
            batch_size=train_cfg.batch_size,
            num_workers=train_cfg.get("num_workers", 2),
            pin_memory=train_cfg.get("pin_memory", True),
            persistent_workers=train_cfg.get("persistent_workers", True),
            drop_last=train_cfg.get("drop_last", True),
        )

        if val_datasets:
            val_cfg = self.config.data.dataloader.val
            self.val_dataloader = [
                DataLoader(
                    ds,
                    shuffle=False, 
                    collate_fn=self._collate_fn,
                    batch_size=val_cfg.get("batch_size", 1),
                    num_workers=val_cfg.get("num_workers", 2),
                    pin_memory=val_cfg.get("pin_memory", True),
                    persistent_workers=val_cfg.get("persistent_workers", True),
                    drop_last=False,
                ) for ds in val_datasets
            ]

    @staticmethod
    def _collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "pixel_values": torch.stack([e["pixel_values"] for e in examples]),
            "conditioning_pixel_values": torch.stack([e["conditioning_pixel_values"] for e in examples]),
            "dataset_indices": torch.tensor([ex["dataset_idx"] for ex in examples], dtype=torch.long),
        }

    def setup_optimization(self) -> None:
        
        opt_cfg = self.config.train.optim
        decay_params, no_decay_params = [], []

        for name, param in self.transformer.named_parameters():
            if not param.requires_grad:
                continue
            (decay_params if "class_embedding_U" in name else no_decay_params).append(param)

        adaln_names = {name for name, p in self.transformer.named_parameters() if ".linear." in name and p.requires_grad}
        mlp_names = {name for name, p in self.transformer.named_parameters() if p.requires_grad and "class_embedding_U" not in name and name not in adaln_names}

        adaln_count = len(adaln_names)
        mlp_count = len(mlp_names)
        self.logger.info(
            f"[Params] class_embedding_U: {len(decay_params)}, AdaLN Linear: {adaln_count}, MLP: {mlp_count}"
        )

        self.optimizer = torch.optim.AdamW(
            [{"params": decay_params, "weight_decay": opt_cfg.get("weight_decay", 0.01)},
             {"params": no_decay_params, "weight_decay": 0.0}],
            lr=opt_cfg.get("lr", 1e-4),
            betas=(opt_cfg.get("beta1", 0.9), opt_cfg.get("beta2", 0.999)),
            eps=opt_cfg.get("epsilon", 1e-8),
            fused=True,
        )

        scheduler_cfg = self.config.train.scheduler
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

        if not self.config.network.get("offload", False):
            self.vae = self.vae.to(self.accelerator.device)

        self.deg_classifier.to(self.accelerator.device, dtype=self.weight_dtype)
        self.deg_classifier.eval()

        embed_dir = self.config.data.get("embed_dir", "./cached_embeddings")
        for ds in self.train_datasets + self.val_datasets:
            if ds.dataset_idx in self.prompt_embed_cache:
                continue
            path = os.path.join(embed_dir, f"dataset_{ds.dataset_idx}_embeds.pt")
            if not os.path.exists(path):
                raise FileNotFoundError"找不到 embedding 文件: {path}")
            data = torch.load(path, map_location="cpu", weights_only=True)
            self.prompt_embed_cache[ds.dataset_idx] = (data["prompt_embeds"], data["text_ids"])

        gc.collect()
        torch.cuda.empty_cache()

    def _get_prompt_embeds_from_cache(self, dataset_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.accelerator.device
        prompt_embeds = torch.cat([
            self.prompt_embed_cache[int(i)][0].to(device, dtype=self.weight_dtype)
            for i in dataset_indices.tolist()
        ], dim=0)
        text_ids = self.prompt_embed_cache[int(dataset_indices[0].item())][1].to(device, dtype=self.weight_dtype)
        return prompt_embeds, text_ids

    def training_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        with self.accelerator.accumulate(self.transformer):
            if self.config.network.get("offload", False):
                self.vae.to(self.accelerator.device)

            hq_latents = encode_images(batch["pixel_values"], self.vae, self.weight_dtype)
            lq_latents = encode_images(batch["conditioning_pixel_values"], self.vae, self.weight_dtype)

            if self.config.network.get("offload", False):
                self.vae.cpu()
                torch.cuda.empty_cache()

            batchsize = hq_latents.shape[0]
            noise = torch.randn_like(lq_latents)

            if self.fixed_timestep is not None:
                timesteps = torch.full((batchsize,), self.fixed_timestep, device=lq_latents.device, dtype=lq_latents.dtype)
                sigmas = self._get_sigmas(timesteps, lq_latents.ndim, lq_latents.dtype)
            else:
                density = compute_density_for_timestep_sampling(
                    weighting_scheme=self.config.loss.get("weighting_scheme", "none"),
                    batch_size=batchsize,
                    logit_mean=self.config.loss.get("logit_mean", 0.0),
                    logit_std=self.config.loss.get("logit_std", 1.0),
                    mode_scale=self.config.loss.get("mode_scale", 1.29),
                )
                indices = (density * self.noise_scheduler.config.num_train_timesteps).long()
                timesteps = self.noise_scheduler.timesteps[indices].to(device=lq_latents.device, dtype=lq_latents.dtype)
                sigmas = self._get_sigmas(timesteps, lq_latents.ndim, lq_latents.dtype)

            noisy_input = (1.0 - sigmas) * hq_latents + sigmas * noise

            h, w = noisy_input.shape[2], noisy_input.shape[3]
            packed_input = _pack_latents(noisy_input, batchsize, h // 2, w // 2)
            latent_image_ids = _prepare_latent_image_ids(noisy_input)

            prompt_embeds, text_ids = self._get_prompt_embeds_from_cache(batch["dataset_indices"])
            if self.config.train.get("proportion_empty_prompts", 0) > 0 and torch.rand(1).item() < self.config.train.proportion_empty_prompts:
                prompt_embeds.zero_()

            guidance = None
            if self.unwrap_fn(self.transformer).config.guidance_embeds:
                guidance = torch.full((batchsize,), self.config.train.get("guidance_scale", 3.0), device=noisy_input.device, dtype=self.weight_dtype)

            deg_feat = self.extract_deg_feat(batch["conditioning_pixel_values"].to(self.accelerator.device))

            model_pred = self.transformer(
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
            model_pred = _unpack_latents_with_ids(model_pred, latent_image_ids, hq_latents.shape[1])

            target = noise - hq_latents
            loss = torch.mean(
                (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(batchsize, -1)
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
        if self.val_dataloader is None:
            return None

        if self.config.network.get("offload", False):
            self.vae.to(self.accelerator.device)

        guidance_scale = self.config.val.get("guidance_scale", 3.0)
        fixed_val_timestep = self.config.val.get("fixed_val_timestep", 300)
        num_vis_samples = self.config.val.get("num_vis_samples", 4)
        max_val_samples = self.config.val.get("max_val_samples",None)
        image_logs = []
        vis_candidates: List[Dict[str, Any]] = []
        seen_datasets = set()
        use_cached_vis = self._fixed_vis_samples is not None and len(self._fixed_vis_samples) > 0
        vis_batch_idx_set = set(
            c["batch_idx"] for c in (self._fixed_vis_samples if use_cached_vis else vis_candidates)
        )

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

        val_pbar = tqdm(
            total=total_val_batches,
            desc=f"Validation @ Iter {self.global_iter}",
            disable=not self.is_main,
            leave=False
        )

        batch_idx = 0
        max_val_reached = False
        for dataloader in self.val_dataloader:
            for batch in dataloader:
                if max_val_samples and batch_idx >= max_val_samples:
                    max_val_reached = True
                    break
                    
                pixel_values = batch["pixel_values"].to(self.accelerator.device)
                cond_pixel_values = batch["conditioning_pixel_values"].to(self.accelerator.device)
                dataset_indices = batch["dataset_indices"]
                batchsize = cond_pixel_values.shape[0]

                for i in range(batchsize):
                    ds_idx = int(dataset_indices[i].item())
                    if ds_idx not in seen_datasets and len(vis_candidates) < num_vis_samples * len(self.val_datasets):
                        seen_datasets.add(ds_idx)
                        vis_candidates.append({"batch_idx": batch_idx, "batch_idx_in_batch": i, "ds_idx": ds_idx})

                lq_latents = encode_images(cond_pixel_values, self.vae, self.weight_dtype)

                noise = torch.randn_like(lq_latents)
                
                timesteps = torch.full((batchsize,), fixed_val_timestep, device=lq_latents.device, dtype=self.weight_dtype)
                sigmas = self._get_sigmas(timesteps, lq_latents.ndim, lq_latents.dtype)
                
                noisy_input = (1.0 - sigmas) * lq_latents + sigmas * noise
                h, w = noisy_input.shape[2], noisy_input.shape[3]
                
                packed_input = _pack_latents(noisy_input, batchsize, h // 2, w // 2)
                latent_image_ids = _prepare_latent_image_ids(noisy_input)

                prompt_embeds, text_ids = self._get_prompt_embeds_from_cache(batch["dataset_indices"])

                guidance = None
                if self.unwrap_fn(self.transformer).config.guidance_embeds:
                    guidance = torch.full((batchsize,), guidance_scale, device=self.accelerator.device, dtype=self.weight_dtype)

                deg_feat = self.extract_deg_feat(cond_pixel_values)

                model_pred = self.unwrap_fn(self.transformer)(
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
                
                x_pred_unpacked = noisy_input - sigmas * model_pred_unpacked
                latents_to_decode = x_pred_unpacked
                
                if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor is not None:
                    latents_to_decode = (latents_to_decode / self.vae.config.scaling_factor) + self.vae.config.shift_factor

                generated = self.vae.decode(latents_to_decode.to(dtype=torch.float32)).sample
                
                pred_tensor = (generated / 2 + 0.5).clamp(0, 1).to(self.accelerator.device)
                gt_tensor = (pixel_values + 1.0) / 2.0

                for i in range(batchsize):
                    ds_idx = int(dataset_indices[i].item())
                    if ds_idx not in val_dataset_indices:
                        continue
                    ssim_val = ssim_fn(pred_tensor[i:i+1], gt_tensor[i:i+1], data_range=1.0)
                    psnr_val = psnr_fn(pred_tensor[i:i+1], gt_tensor[i:i+1], data_range=1.0)
                    val_metrics[ds_idx]["ssim"] += ssim_val.item()
                    val_metrics[ds_idx]["psnr"] += psnr_val.item()
                    val_metrics[ds_idx]["count"] += 1

                is_vis = vis_batch_idx_set.__contains__(batch_idx)
                if self.is_main and is_vis:
                    gen_img_np = pred_tensor[0].cpu().float().permute(1, 2, 0).numpy()
                    generated_pil = Image.fromarray((gen_img_np * 255).astype(np.uint8)).convert("RGB")

                    gt_img_np = gt_tensor[0].cpu().float().permute(1, 2, 0).numpy()
                    gt_img_pil = Image.fromarray((gt_img_np * 255).astype(np.uint8)).convert("RGB")

                    cond_img_np = ((cond_pixel_values[0].cpu() + 1.0) / 2.0).clamp(0, 1)
                    cond_img_pil = Image.fromarray((cond_img_np.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).convert("RGB")

                    ds_name = list(self.config.data.datasets.keys())[int(dataset_indices[0].item())]
                    cond_img_pil.save(os.path.join(vis_dir, f"{ds_name}_LQ.png"))
                    generated_pil.save(os.path.join(vis_dir, f"{ds_name}_pred.png"))
                    gt_img_pil.save(os.path.join(vis_dir, f"{ds_name}_HQ.png"))

                    image_logs.append({
                        "validation_image": cond_img_pil,
                        "images": [generated_pil],
                        "validation_prompt": self.val_datasets[0].prompt,
                        "dataset_idx": int(dataset_indices[0].item()),
                        "dataset_name": ds_name,
                    })

                del noise, latent_image_ids, sigmas, text_ids
                del pixel_values, cond_pixel_values, lq_latents, noisy_input
                del packed_input, prompt_embeds, deg_feat, model_pred
                del model_pred_unpacked, x_pred_unpacked, latents_to_decode, generated
                del pred_tensor, gt_tensor
                gc.collect()
                torch.cuda.empty_cache()

                batch_idx += 1
                val_pbar.update(1)

            if max_val_reached:
                break

        if self._fixed_vis_samples is None and vis_candidates:
            self._fixed_vis_samples = vis_candidates

        dataset_results = {}
        for ds_idx, metrics in val_metrics.items():
            if metrics["count"] > 0:
                ds_name = list(self.config.data.datasets.keys())[ds_idx]
                dataset_results[ds_name] = {
                    "ssim": metrics["ssim"] / metrics["count"],
                    "psnr": metrics["psnr"] / metrics["count"],
                }

        if self.is_main and dataset_results:
            for ds_name, metrics in dataset_results.items():
                self.accelerator.log({f"val/{ds_name}/SSIM": metrics["ssim"]}, step=self.global_step)
                self.accelerator.log({f"val/{ds_name}/PSNR": metrics["psnr"]}, step=self.global_step)

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
        gc.collect()
        torch.cuda.empty_cache()
        return {"image_logs": image_logs, "dataset_results": dataset_results}

    def save_ckpt(self, is_best: bool = False, non_blocking: bool = True) -> None:
        unwrapped = self.unwrap_fn(self.transformer)
        
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
        for attr_name in ["class_embedding_U", "class_embedding_S"]:
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
                    print(f"✅ Saved checkpoint [{tag}] to {save_path}")
                    print(f"   - trainable_params.pt ({len(trainable_params)} params)")
            except Exception as e:
                if self.is_main:
                    print(f"❌ Failed to save checkpoint: {e}")
        
        if non_blocking:
            import threading
            threading.Thread(target=_do_save, name=f"ckpt-{self.global_iter}").start()
        else:
            _do_save()
    
    def load_ckpt(self, ckpt_path: str) -> None:
        unwrapped = self.unwrap_fn(self.transformer)
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
            if self.is_main:
                print(f"  Restored extra state: {list(extra_state.keys())}")

        scheduler_state_path = os.path.join(ckpt_path, "scheduler.pt")
        if os.path.exists(scheduler_state_path) and self.scheduler is not None:
            self.scheduler.load_state_dict(torch.load(scheduler_state_path, map_location="cpu"))
            if self.is_main:
                print("  Restored scheduler state")
            if self.is_main:
                print("Restored scheduler state")

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

        if self.val_dataloader and not self.config.val.get("skip_init_val", False):
            if self.is_main:
                print(f"\n{'-'*60}\nRunning Initial Validation\n{'-'*60}")
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

        self.transformer.train()

        gc.collect()                  
        torch.cuda.empty_cache()     
        torch.cuda.reset_peak_memory_stats()

        val_freq_iters = self.config.val.get("val_freq_iters", 10000)
        last_val_iter = 0
        save_freq_iters = self.config.val.get("val_freq_iters", 10000)

        while self.global_iter < num_train_iters:
            self.transformer.train()
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
                    if self.global_iter > 0 and self.global_iter % self.config.logging.get("log_interval_iters", 10) == 0:
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
                    
                    gc.collect()
                    torch.cuda.empty_cache()

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
    ConditionTrainer(args).train()

if __name__ == "__main__":
    main()
