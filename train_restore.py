import argparse
import copy
import logging
import os
from datetime import datetime
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.utils.data import ChainDataset, DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from tqdm import tqdm
import swanlab
from PIL import Image
from omegaconf import OmegaConf

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxImg2ImgPipeline,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)

# 移除了 transformers 相关的导入

from src.data.dataset import InterleavedShuffleDataset, PairedDataset
from src.networks.degnet import DegNet_DINO


def encode_images(pixels: torch.Tensor, vae: torch.nn.Module, weight_dtype):
    pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
    pixel_latents = (
        pixel_latents - vae.config.shift_factor
    ) * vae.config.scaling_factor
    return pixel_latents.to(weight_dtype)


class Trainer:

    def __init__(self, args):
        self.args = args
        self.config = OmegaConf.load(args.config)

        self.exp_name = self.config.get("name", "flux_schnell_restore")
        self.exp_name = f"{self.exp_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        set_seed(self.config.seed)

        self.num_classes = self.config.network.get("num_classes", 6)
        self.current_epoch = 0
        self.global_step = 0

        self.logger = logging.getLogger(__name__)

        self.accelerator = None
        self.vae = None
        self.transformer = None
        self.noise_scheduler = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataloader = None
        self.weight_dtype = None

        self.train_datasets = []
        self.val_datasets = []
        self.prompt_embed_cache = {}

        self.deg_classifier = DegNet_DINO(
            dino_type=self.config.network.get("dino_type", None)
        )
        self.deg_classifier.load_state_dict(torch.load(self.config.network.degradation_classifier_path))

        self.output_dir = None
        self.ckpt_dir = None
        self.log_dir = None

    def setup_dist(self):
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=self.config.accelerator.get(
                "find_unused_parameters", False
            )
        )

        log_with = self.config.accelerator.get("report_to", "tensorboard")
        if self.config.logging.get("use_swanlab", False):
            log_with = "swanlab"

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.train.get(
                "gradient_accumulation_steps", 1
            ),
            mixed_precision=self.config.accelerator.get("mixed_precision", "no"),
            log_with=log_with,
            project_dir=self.config.accelerator.get("project_dir", "./experiments"),
            kwargs_handlers=[ddp_kwargs],
        )

        self.is_main = self.accelerator.is_main_process
        self.unwrap_fn = self.accelerator.unwrap_model

    def init_logger(self):
        self.output_dir = os.path.join(
            self.config.accelerator.get("project_dir", "./experiments"), self.exp_name
        )
        self.ckpt_dir = os.path.join(self.output_dir, "checkpoints")

        if self.is_main:
            os.makedirs(self.ckpt_dir, exist_ok=True)

            config_save_path = os.path.join(self.output_dir, "config.yaml")
            OmegaConf.save(self.config, config_save_path)
            print(f"[Config] Saved to {config_save_path}")

        if self.config.logging.get("use_swanlab", False):
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
        else:
            self.accelerator.init_trackers(
                project_name=self.config.logging.get("swanlab_project", "experiment")
            )

    def build_models(self):
        model_path = self.config.network.pretrained_model_name_or_path

        precisions = {"fp16": torch.float16, "bf16": torch.bfloat16}
        self.weight_dtype = precisions.get(
            self.config.accelerator.get("mixed_precision", "no"), torch.float32
        )

        self.vae = AutoencoderKL.from_pretrained(
            model_path,
            subfolder="vae",
            revision=self.config.network.get("revision"),
            variant=self.config.network.get("variant"),
            torch_dtype=torch.float32,
        )
        self.vae.requires_grad_(False)

        self.transformer = FluxTransformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            revision=self.config.network.get("revision"),
            variant=self.config.network.get("variant"),
            torch_dtype=self.weight_dtype,
        )
        self.transformer.requires_grad_(True)

        self.transformer.register_parameter(
            "class_embedding_U",
            nn.Parameter(torch.randn(self.num_classes, 768), requires_grad=True)
        )
        nn.init.orthogonal_(self.transformer.class_embedding_U)

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path,
            subfolder="scheduler",
        )

        self.fixed_timestep = self.config.train.get("fixed_timestep", None)
        if self.fixed_timestep is not None:
            print(f"[Trainer] Using fixed timestep: {self.fixed_timestep}")

        self.vae.to(dtype=torch.float32)
        self.transformer.to(dtype=self.weight_dtype)

        if self.config.network.get("gradient_checkpointing", False):
            self.transformer.enable_gradient_checkpointing()
        if self.config.accelerator.get("allow_tf32", False):
            torch.backends.cuda.matmul.allow_tf32 = True

    def extract_deg_feat(self, lq_images):
        self.deg_classifier.to(self.accelerator.device)
        self.deg_classifier.eval()

        with torch.no_grad():
            logits = self.deg_classifier(lq_images)
            probs = F.softmax(logits[:, :, 0], dim=-1)

        deg_feat = probs @ self.transformer.class_embedding_U

        return deg_feat

    def build_dataloader(self):
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
            )
            if is_val:
                val_datasets.append(dataset)
            else:
                train_datasets.append(dataset)

        self.train_datasets = train_datasets
        self.val_datasets = val_datasets

        train_loader_cfg = self.config.data.dataloader.train
        if len(train_datasets) > 1:
            self.train_dataloader = DataLoader(
                InterleavedShuffleDataset(
                    train_datasets, buffer_size=500, seed=self.config.seed
                ),
                shuffle=False,
                collate_fn=self.collate_fn,
                batch_size=train_loader_cfg.batch_size,
                num_workers=train_loader_cfg.get("num_workers", 4),
                pin_memory=train_loader_cfg.get("pin_memory", True),
                persistent_workers=train_loader_cfg.get("persistent_workers", True),
                drop_last=train_loader_cfg.get("drop_last", True),
            )
        else:
            self.train_dataloader = DataLoader(
                train_datasets[0],
                collate_fn=self.collate_fn,
                batch_size=train_loader_cfg.batch_size,
                num_workers=train_loader_cfg.get("num_workers", 4),
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
                batch_size=val_loader_cfg.batch_size,
                num_workers=val_loader_cfg.get("num_workers", 4),
                pin_memory=val_loader_cfg.get("pin_memory", True),
                persistent_workers=val_loader_cfg.get("persistent_workers", True),
                drop_last=val_loader_cfg.get("drop_last", False),
            )
        else:
            self.val_dataloader = None

    @staticmethod
    def collate_fn(examples):
        pixel_values = torch.stack([e["pixel_values"] for e in examples])
        cond_pixel_values = torch.stack(
            [e["conditioning_pixel_values"] for e in examples]
        )
        captions = [e["captions"] for e in examples]
        dataset_indices = torch.tensor(
            [ex["dataset_idx"] for ex in examples], dtype=torch.long
        )
        return {
            "pixel_values": pixel_values,
            "conditioning_pixel_values": cond_pixel_values,
            "captions": captions,
            "dataset_indices": dataset_indices,
        }

    def setup_optimization(self):
        opt_cfg = self.config.train.optim

        trainable_params = [p for p in self.transformer.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=opt_cfg.get("lr", 1e-4),
            betas=(opt_cfg.get("beta1", 0.9), opt_cfg.get("beta2", 0.999)),
            weight_decay=opt_cfg.get("weight_decay", 0.01),
            eps=opt_cfg.get("epsilon", 1e-8),
        )

        scheduler_cfg = self.config.train.scheduler
        num_training_steps = (
            len(self.train_dataloader) // self.accelerator.gradient_accumulation_steps
        ) * self.config.train.num_train_epochs

        self.scheduler = get_scheduler(
            scheduler_cfg.type,
            optimizer=self.optimizer,
            num_warmup_steps=scheduler_cfg.get("lr_warmup_steps", 500),
            num_training_steps=num_training_steps,
            num_cycles=scheduler_cfg.get("lr_num_cycles", 1),
        )

        self.transformer, self.optimizer, self.train_dataloader, self.scheduler = (
            self.accelerator.prepare(
                self.transformer, self.optimizer, self.train_dataloader, self.scheduler
            )
        )

        self.vae = self.vae.to(self.accelerator.device)

        # 直接读取预处理好的 prompt embeddings
        self._precompute_prompt_embeddings()

        gc.collect()
        torch.cuda.empty_cache()

    def _precompute_prompt_embeddings(self):
        # 从配置中获取目录，如果没有则使用默认目录
        embed_dir = self.config.data.get("embed_dir", "./cached_embeddings")
        
        datasets = self.train_datasets + self.val_datasets
        unique_dataset_idx = set([ds.dataset_idx for ds in datasets])

        for ds_idx in unique_dataset_idx:
            if ds_idx in self.prompt_embed_cache:
                continue

            load_path = os.path.join(embed_dir, f"dataset_{ds_idx}_embeds.pt")
            if not os.path.exists(load_path):
                raise FileNotFoundError(f"找不到预计算的特征文件: {load_path}. 请先运行提取脚本。")
            
            # 读取特征并存放在 CPU 上，随用随取
            loaded_data = torch.load(load_path, map_location="cpu", weights_only=True)
            
            self.prompt_embed_cache[ds_idx] = (
                loaded_data["prompt_embeds"],
                loaded_data["pooled_prompt_embeds"],
                loaded_data["text_ids"]
            )
            
            if self.is_main:
                print(f"  [Cache] dataset_idx={ds_idx} 成功加载硬盘缓存.")

        if self.is_main:
            print(f"[PromptCache] 共加载 {len(self.prompt_embed_cache)} 个 prompt embeddings")

    def _get_prompt_embeds_from_cache(self, dataset_indices):
        device = self.accelerator.device

        prompt_embeds_list = []
        pooled_list = []
        text_ids_list = []

        for idx in dataset_indices.tolist():
            p_embed, pooled, txt_ids = self.prompt_embed_cache[int(idx)]
            prompt_embeds_list.append(p_embed.to(device))
            pooled_list.append(pooled.to(device))
            text_ids_list.append(txt_ids.to(device))

        prompt_embeds = torch.stack(prompt_embeds_list)
        pooled_prompt_embeds = torch.stack(pooled_list)
        text_ids = torch.stack(text_ids_list)

        return prompt_embeds, pooled_prompt_embeds, text_ids

    def training_step(self, batch):
        with self.accelerator.accumulate(self.transformer):
            hq_latents = encode_images(batch["pixel_values"], self.vae, self.weight_dtype)
            lq_latents = encode_images(batch["conditioning_pixel_values"], self.vae, self.weight_dtype)

            if self.config.network.get("offload", False):
                self.vae.cpu()

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

                noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)
                indices = (density * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(
                    device=lq_latents.device
                )

                sigmas = self._get_sigmas(
                    timesteps, n_dim=lq_latents.ndim, dtype=lq_latents.dtype
                )
            noisy_input = (1.0 - sigmas) * lq_latents + sigmas * noise

            packed_input = FluxImg2ImgPipeline._pack_latents(
                noisy_input,
                batch_size=batchsize,
                num_channels_latents=noisy_input.shape[1],
                height=noisy_input.shape[2],
                width=noisy_input.shape[3],
            )

            latent_image_ids = FluxImg2ImgPipeline._prepare_latent_image_ids(
                batchsize=batchsize,
                height=noisy_input.shape[2] // 2,
                width=noisy_input.shape[3] // 2,
                device=self.accelerator.device,
                dtype=self.weight_dtype,
            )

            if self.unwrap_fn(self.transformer).config.guidance_embeds:
                guidance = torch.full(
                    (batchsize,),
                    self.config.train.get("guidance_scale", 30.0),
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
                height=noisy_input.shape[2] * 2,
                width=noisy_input.shape[3] * 2,
                vae_scale_factor=2,
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

    def _get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
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
    def validation(self):
        if self.val_dataloader is None:
            self.logger.info("No validation dataloader, skipping validation...")
            return None

        self.logger.info("Running single-step validation...")

        guidance_scale = self.config.val.get("guidance_scale", 30.0)
        fixed_val_timestep = self.config.val.get("fixed_val_timestep", 1.0)
        print(f"[Validation] Using single-step inference with timestep: {fixed_val_timestep}")

        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.accelerator.device)
        psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(self.accelerator.device)

        all_ssim = []
        all_psnr = []
        image_logs = []

        to_tensor = transforms.ToTensor()

        for batch_idx, batch in enumerate(self.val_dataloader):
            if (
                self.config.val.get("max_val_samples")
                and batch_idx >= self.config.val.max_val_samples
            ):
                break

            pixel_values = batch["pixel_values"].to(self.accelerator.device)
            cond_pixel_values = batch["conditioning_pixel_values"].to(
                self.accelerator.device
            )
            captions = batch["captions"]

            cond_img_pil = transforms.ToPILImage()(cond_pixel_values[0].cpu()).convert("RGB")

            cond_latents = encode_images(
                cond_pixel_values, self.vae, self.weight_dtype
            )

            timestep_tensor = torch.tensor(
                [fixed_val_timestep], device=self.accelerator.device, dtype=self.weight_dtype
            )

            prompt_embeds, pooled_prompt_embeds, text_ids = self._get_prompt_embeds_from_cache(
                batch["dataset_indices"]
            )

            latent_image_ids = FluxImg2ImgPipeline._prepare_latent_image_ids(
                batchsize=1,
                height=cond_latents.shape[2] // 2,
                width=cond_latents.shape[3] // 2,
                device=self.accelerator.device,
                dtype=self.weight_dtype,
            )

            packed_input = FluxImg2ImgPipeline._pack_latents(
                cond_latents,
                batch_size=1,
                num_channels_latents=cond_latents.shape[1],
                height=cond_latents.shape[2],
                width=cond_latents.shape[3],
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
                timestep=timestep_tensor / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]

            model_pred = FluxImg2ImgPipeline._unpack_latents(
                model_pred,
                height=cond_latents.shape[2] * 2,
                width=cond_latents.shape[3] * 2,
                vae_scale_factor=2,
            )

            model_pred = model_pred / self.vae.config.scaling_factor + self.vae.config.shift_factor
            generated = self.vae.decode(model_pred.to(dtype=torch.float32)).sample
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

            if batch_idx < 4:
                image_logs.append(
                    {
                        "validation_image": cond_img_pil,
                        "images": [generated],
                        "validation_prompt": captions[0] if captions else "",
                    }
                )

        avg_ssim = sum(all_ssim) / len(all_ssim) if all_ssim else None
        avg_psnr = sum(all_psnr) / len(all_psnr) if all_psnr else None

        self.logger.info(f"Validation SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.2f}")

        if self.is_main:
            val_metrics = {}
            if avg_ssim is not None:
                val_metrics["validation/SSIM"] = avg_ssim
            if avg_psnr is not None:
                val_metrics["validation/PSNR"] = avg_psnr

            if val_metrics:
                self.accelerator.log(val_metrics, step=self.global_step)

        if self.config.logging.get("use_swanlab", False) and self.is_main:
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

    def save_ckpt(self, is_best=False):
        unwrapped = self.unwrap_fn(self.transformer)

        if self.config.output.get("upcast_before_saving", False):
            unwrapped.to(torch.float32)
        filename = "best_model" if is_best else f"step_{self.global_step}"
        save_path = os.path.join(self.ckpt_dir, filename)
        unwrapped.save_pretrained(save_path)
        print(f"[Checkpoint] Saved to {save_path}")

    def train(self):
        self.setup_dist()
        self.init_logger()
        self.build_models()
        self.build_dataloader()
        self.setup_optimization()

        self.transformer.train()

        num_epochs = self.config.train.num_train_epochs

        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            pbar = tqdm(
                self.train_dataloader,
                disable=not self.is_main,
                desc=f"Epoch {self.current_epoch}",
            )

            for batch in pbar:
                loss = self.training_step(batch)

                if self.is_main:
                    pbar.set_postfix(
                        {
                            "loss": f"{loss.item():.4f}",
                            "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                        }
                    )

            val_cfg = self.config.val
            if (
                self.val_dataloader is not None
                and (epoch + 1) % val_cfg.get("val_freq", 1) == 0
            ):
                image_logs = self.validation()

                if self.is_main and image_logs:
                    print("\n" + "=" * 60)
                    print(f"Epoch {self.current_epoch} Validation")
                    print("=" * 60)

            save_freq = self.config.train.get("save_freq", 1)
            if self.is_main and (epoch + 1) % save_freq == 0:
                self.save_ckpt()

        if self.is_main:
            self.save_ckpt(is_best=True)

        self.accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to YAML config"
    )

    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.train()