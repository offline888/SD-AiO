import logging
import math
import os
import shutil
from pathlib import Path

import torch
import torch.utils.data
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_loss_weighting_for_sd3, free_memory
from diffusers.utils import check_min_version
from diffusers.utils.logging import set_verbosity_error, set_verbosity_info
from diffusers.utils.torch_utils import is_compiled_module
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

from src.data.dataset import PairedDataset
from src.models import FLUX2ModulationV2
from src.models.deg_extractor import DegFeatExtractor
from src.flux2.pipelines.latent_utils import (
    pack_latents,
    patchify_latents,
    prepare_latent_ids,
    unpack_latents_with_ids,
)
from src.flux2.pipelines.text_encoder import compute_text_embeddings
from src.flux2.transformer_flux2 import Flux2Transformer2DModel
from src.utils.log import log_once

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, args):
        self.args = args
        self._setup_accelerator()
        self._setup_dtype()
        self._setup_models()
        self._setup_deg_extractor()
        self._replace_modulation()
        self._freeze_unfreeze()
        self._move_to_device()
        self._setup_gradient_checkpointing()
        self._setup_optimizer()
        self._load_datasets()
        self._precompute_text_embeddings()
        self._prepare_validation()
        self._setup_lr_scheduler()
        self._prepare()
        self._register_hooks()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_accelerator(self):
        logging_dir = Path(self.args.output_dir, self.args.logging_dir)
        project_config = ProjectConfiguration(
            project_dir=self.args.output_dir, logging_dir=logging_dir
        )
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=self.args.mixed_precision,
            log_with=self.args.report_to,
            project_config=project_config,
            kwargs_handlers=[ddp_kwargs],
        )
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        log_once(self.accelerator.state, self.accelerator)
        if self.accelerator.is_local_main_process:
            set_verbosity_info()
        else:
            set_verbosity_error()
        if self.accelerator.is_main_process:
            os.makedirs(self.args.output_dir, exist_ok=True)

    def _setup_dtype(self):
        self.weight_dtype = torch.bfloat16
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

    def _setup_models(self):
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="tokenizer",
        )

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="scheduler",
        )
        self._original_sigmas = self.noise_scheduler.sigmas.clone()

        self.vae = self._load_vae()
        self.latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(self.accelerator.device)
        self.latents_bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
        ).to(self.accelerator.device)

        self.transformer = Flux2Transformer2DModel.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype=self.weight_dtype,
        )

        self.text_encoder = Qwen3ForCausalLM.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="text_encoder",
        )

        # Pipeline for text encoding only (no VAE, no transformer, no scheduler)
        from src.flux2.pipelines.flux2_klein import Flux2KleinIRPipeline
        self.text_encoding_pipeline = Flux2KleinIRPipeline.from_pretrained(
            self.args.pretrained_model_name_or_path,
            vae=None,
            transformer=None,
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            scheduler=None,
        )

    def _load_vae(self):
        from diffusers import AutoencoderKLFlux2
        return AutoencoderKLFlux2.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="vae",
        )

    def _setup_deg_extractor(self):
        self.deg_extractor = DegFeatExtractor(
            inner_dim=self.transformer.inner_dim,
            num_deg_types=self.args.num_deg_types,
            weight_dtype=self.weight_dtype,
            args=self.args,
            deg_embedding=None,
            device=self.accelerator.device,
        )
        self.transformer.register_parameter("deg_embedding", self.deg_extractor.deg_embedding)

    # ------------------------------------------------------------------
    # Modulation replacement
    # ------------------------------------------------------------------

    def _replace_modulation(self):
        use_conv = (self.args.mod_lq_type == "convnext")
        use_vae = (self.args.mod_lq_type == "vae")

        replacements = {
            "double_stream_modulation_img": 2,
            "single_stream_modulation": 1,
        }

        for mod_name, mod_param_sets in replacements.items():
            if not hasattr(self.transformer, mod_name):
                continue
            from diffusers.models.transformers.transformer_flux2 import Flux2Modulation
            if isinstance(getattr(self.transformer, mod_name), Flux2Modulation):
                orig = getattr(self.transformer, mod_name)
                new_mod = FLUX2ModulationV2(
                    dim=orig.linear.in_features,
                    mod_param_sets=mod_param_sets,
                    bias=orig.linear.bias is not None,
                    use_block_emb=True,
                    use_conv=use_conv,
                    use_vae=use_vae,
                    vae_path=self.args.pretrained_model_name_or_path,
                )
                new_mod.linear.load_state_dict(orig.linear.state_dict())
                setattr(self.transformer, mod_name, new_mod)
                log_once(
                    f"Replaced {mod_name} with FLUX2ModulationV2 ({self.args.mod_lq_type})",
                    self.accelerator,
                )

    # ------------------------------------------------------------------
    # Freeze / Unfreeze
    # ------------------------------------------------------------------

    def _freeze_unfreeze(self):
        use_conv = (self.args.mod_lq_type == "convnext")
        use_vae = (self.args.mod_lq_type == "vae")

        self.transformer.requires_grad_(False)

        modulation_names = ["double_stream_modulation_img", "single_stream_modulation"]

        for mod_name in modulation_names:
            if not hasattr(self.transformer, mod_name):
                continue
            mod = getattr(self.transformer, mod_name)

            for sub in ["block_proj", "block_embedder"]:
                if hasattr(mod, sub):
                    getattr(mod, sub).requires_grad_(True)
                    log_once(f"Unlocked {mod_name}.{sub}", self.accelerator)

            if use_conv:
                for sub in [
                    "conv_stem_s1", "conv_down1_s2", "conv_down2_s3",
                    "conv_time_mod1", "conv_time_mod2", "conv_time_mod3",
                    "feat_proj",
                ]:
                    if hasattr(mod, sub):
                        getattr(mod, sub).requires_grad_(True)
                        log_once(f"Unlocked {mod_name}.{sub}", self.accelerator)

            if use_vae:
                for sub in ["vae_time_mods", "vae_mid_time_mod", "vae_proj"]:
                    if hasattr(mod, sub):
                        getattr(mod, sub).requires_grad_(True)
                        log_once(f"Unlocked {mod_name}.{sub}", self.accelerator)
                for n, p in mod.vae.named_parameters():
                    p.requires_grad_(False)
                log_once(f"Frozen {mod_name}.vae (all params)", self.accelerator)

            # linear is now trainable (no longer frozen)
            if hasattr(mod, "linear"):
                mod.linear.requires_grad_(True)
                log_once(f"Unlocked {mod_name}.linear (was frozen in original)", self.accelerator)

        if hasattr(self.transformer, "deg_embedding"):
            self.transformer.deg_embedding.requires_grad_(True)
            log_once("Unlocked transformer.deg_embedding", self.accelerator)

        params = [p for p in self.transformer.parameters() if p.requires_grad]
        total = sum(x.numel() for x in params)
        log_once(f"Total number of parameters to optimize: {total}", self.accelerator)

    # ------------------------------------------------------------------
    # Device & Precision
    # ------------------------------------------------------------------

    def _move_to_device(self):
        self.vae.to(device=self.accelerator.device, dtype=self.weight_dtype)
        self.transformer.to(device=self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder.to(device=self.accelerator.device, dtype=self.weight_dtype)

    def _setup_gradient_checkpointing(self):
        if self.args.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def _setup_optimizer(self):
        if self.args.optimizer.lower() not in ("adamw", "prodigy"):
            log_once(
                f"Unsupported optimizer: {self.args.optimizer}. Defaulting to AdamW.",
                self.accelerator,
            )
            self.args.optimizer = "adamw"

        params = [
            {"params": [p for p in self.transformer.parameters() if p.requires_grad],
             "lr": self.args.learning_rate},
        ]

        if self.args.optimizer.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                params,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                weight_decay=self.args.adam_weight_decay,
                eps=self.args.adam_epsilon,
            )
        else:
            try:
                import prodigyopt
            except ImportError:
                raise ImportError("To use Prodigy, install: `pip install prodigyopt`")
            self.optimizer = prodigyopt.Prodigy(
                params,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                weight_decay=self.args.adam_weight_decay,
                eps=self.args.adam_epsilon,
            )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _load_datasets(self):
        data_config = OmegaConf.load(self.args.datasets_config)
        train_datasets_cfg = [
            data_config[k] for k in data_config.keys() if k.startswith("Train")
        ]

        self.train_datasets = []
        for key in data_config.keys():
            if not key.startswith("Train"):
                continue
            ds_cfg = data_config[key]
            dataset = PairedDataset(
                lq_path=str(ds_cfg.lq_path),
                hq_path=str(ds_cfg.hq_path),
                resolution=self.args.resolution,
                prompt=str(ds_cfg.prompt),
                dataset_idx=len(self.train_datasets),
                deg_type=str(ds_cfg.deg_type),
                enlarge_ratio=float(ds_cfg.enlarge_ratio),
            )
            self.train_datasets.append(dataset)
            log_once(
                f"Loaded train dataset '{key}': lq={ds_cfg.lq_path}, hq={ds_cfg.hq_path}, "
                f"prompt='{ds_cfg.prompt}'",
                self.accelerator,
            )

        if not self.train_datasets:
            raise ValueError(
                f"No training datasets found in {self.args.datasets_config}. "
                "Expected keys starting with 'Train'."
            )

        from src.data.collate import collate_fn
        train_dataset = ConcatDataset(self.train_datasets)
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
            drop_last=True,
        )

    def _precompute_text_embeddings(self):
        data_config = OmegaConf.load(self.args.datasets_config)
        train_datasets_cfg = [
            data_config[k] for k in data_config.keys() if k.startswith("Train")
        ]
        self.prompt_embeds_cache = {}
        self.text_ids_cache = {}
        for idx, cfg in enumerate(train_datasets_cfg):
            embeds, tids = compute_text_embeddings(cfg.prompt, self.text_encoding_pipeline)
            self.prompt_embeds_cache[idx] = embeds.to(self.accelerator.device)
            self.text_ids_cache[idx] = tids.to(self.accelerator.device)

        del self.text_encoder, self.tokenizer, self.text_encoding_pipeline
        free_memory()

    def _prepare_validation(self):
        self.val_monitor_steps = (
            self.args.val_monitor_steps
            if self.args.val_monitor_steps is not None
            else self.args.save_checkpointing_steps
        )
        self.monitor_items = []
        self.monitor_output_dir = None

        num_val_samples = getattr(self.args, "num_val_samples_per_dataset", 1)
        if num_val_samples < 1:
            num_val_samples = 1

        if self.val_monitor_steps > 0 and self.accelerator.is_main_process:
            data_config = OmegaConf.load(self.args.datasets_config)
            train_datasets_cfg = [
                data_config[k] for k in data_config.keys() if k.startswith("Train")
            ]
            for ds_idx, (ds, cfg) in enumerate(zip(self.train_datasets, train_datasets_cfg)):
                num_available = len(ds.pairs)
                num_to_sample = min(num_val_samples, num_available)

                for sample_idx in range(num_to_sample):
                    hq_path, lq_path = ds.pairs[sample_idx]
                    hq_pil = ds._load_image(hq_path)
                    lq_pil = ds._load_image(lq_path)
                    lq_t = ds.transforms(lq_pil)
                    hq_t = ds.transforms(hq_pil)
                    self.monitor_items.append({
                        "label": f"{cfg.deg_type}_{sample_idx}",
                        "lq_pixel_values": lq_t,
                        "hq_pixel_values": hq_t.clone(),
                        "prompt_embeds": self.prompt_embeds_cache[ds_idx].squeeze(0),
                        "text_ids": self.text_ids_cache[ds_idx].squeeze(0),
                    })
                    log_once(
                        f"[ValMonitor] ds={ds_idx} ({cfg.deg_type}) "
                        f"sample {sample_idx + 1}/{num_to_sample}: lq={lq_path}",
                        self.accelerator,
                    )
            self.monitor_output_dir = os.path.join(self.args.output_dir, "val_monitor")
            log_once(
                f"[ValMonitor] Will snapshot every {self.val_monitor_steps} steps "
                f"to {self.monitor_output_dir} "
                f"({num_to_sample} image(s) per dataset, total {len(self.monitor_items)})",
                self.accelerator,
            )

    # ------------------------------------------------------------------
    # LR Scheduler
    # ------------------------------------------------------------------

    def _setup_lr_scheduler(self):
        num_warmup_steps = self.args.lr_warmup_steps * self.accelerator.num_processes
        if self.args.max_train_steps is None:
            len_dataloader_sharded = math.ceil(
                len(self.train_dataloader) / self.accelerator.num_processes
            )
            num_update_steps_per_epoch = math.ceil(
                len_dataloader_sharded / self.args.gradient_accumulation_steps
            )
            self.num_training_steps = (
                self.args.num_train_epochs
                * self.accelerator.num_processes
                * num_update_steps_per_epoch
            )
        else:
            self.num_training_steps = self.args.max_train_steps * self.accelerator.num_processes

        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.num_training_steps,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power,
        )

    # ------------------------------------------------------------------
    # Prepare (accelerator.wrap)
    # ------------------------------------------------------------------

    def _prepare(self):
        self.transformer, self.optimizer, self.train_dataloader, self.lr_scheduler = (
            self.accelerator.prepare(
                self.transformer, self.optimizer, self.train_dataloader, self.lr_scheduler
            )
        )
        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.args.gradient_accumulation_steps
        )
        if self.args.max_train_steps is None:
            self.args.max_train_steps = (
                self.args.num_train_epochs * self.num_update_steps_per_epoch
            )
        self.args.num_train_epochs = math.ceil(
            self.args.max_train_steps / self.num_update_steps_per_epoch
        )

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                "flux2-image-restoration",
                config=vars(self.args),
            )

    # ------------------------------------------------------------------
    # Checkpoint hooks
    # ------------------------------------------------------------------

    def _unwrap_model(self, model):
        model = self.accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def _register_hooks(self):
        def save_model_hook(models, weights, output_dir):
            transformer_model = None
            for model in models:
                m = self._unwrap_model(model)
                if isinstance(m, Flux2Transformer2DModel):
                    transformer_model = model
                    break
            if transformer_model is None:
                raise ValueError("No Flux2Transformer2DModel found in models")
            if weights:
                weights.pop()
            if self.accelerator.is_main_process:
                state_dict = self.accelerator.get_state_dict(transformer_model)
                trainable = {
                    k: v.to("cpu") for k, v in state_dict.items() if v.requires_grad
                }
                if not trainable:
                    return
                torch.save(trainable, os.path.join(output_dir, "modulation_weights.pt"))
                log_once(
                    f"Saved {len(trainable)} tensors "
                    f"({sum(v.numel() for v in trainable.values()):,} params) "
                    f"to {output_dir}/modulation_weights.pt",
                    self.accelerator,
                )

        def load_model_hook(models, input_dir):
            transformer_ = None
            while models:
                model = models.pop()
                m = self._unwrap_model(model)
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
                    f"Restored {len(state_dict)} tensors "
                    f"({sum(v.numel() for v in state_dict.values()):,} params) "
                    f"from {ckpt_path}",
                    self.accelerator,
                )
                if missing:
                    log_once(f"  missing: {missing}", self.accelerator)
                if unexpected:
                    log_once(f"  unexpected: {unexpected}", self.accelerator)

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def _get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self._original_sigmas.to(
            device=self.accelerator.device, dtype=dtype
        )
        timesteps = timesteps.to(self.accelerator.device)
        step_indices = timesteps.long()
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def training_step(self, batch, global_step):
        dataset_indices = batch["dataset_indices"].to(self.accelerator.device)
        bsz = dataset_indices.shape[0]
        unique_ds = dataset_indices.unique()

        if unique_ds.numel() == 1:
            ds_idx = unique_ds.item()
            pe = self.prompt_embeds_cache[ds_idx].to(self.accelerator.device)
            ti = self.text_ids_cache[ds_idx].to(self.accelerator.device)
            prompt_embeds = pe.repeat(bsz, 1, 1)
            text_ids = ti.repeat(bsz, 1, 1)
        else:
            all_pes, all_tis = [], []
            for idx in range(bsz):
                ds_idx = dataset_indices[idx].item()
                pe = self.prompt_embeds_cache[ds_idx].to(self.accelerator.device)
                ti = self.text_ids_cache[ds_idx].to(self.accelerator.device)
                all_pes.append(pe)
                all_tis.append(ti)
            prompt_embeds = torch.cat(all_pes, dim=0)
            text_ids = torch.cat(all_tis, dim=0)

        lq_pixel_values = batch["lq_pixel_values"].to(
            device=self.accelerator.device, dtype=self.weight_dtype
        )
        hq_pixel_values = batch["hq_pixel_values"].to(
            device=self.accelerator.device, dtype=self.weight_dtype
        )

        with torch.no_grad():
            model_input = self.vae.encode(lq_pixel_values).latent_dist.mode()
            hq_latent = self.vae.encode(hq_pixel_values).latent_dist.mode()

        model_input = patchify_latents(model_input)
        model_input = (model_input - self.latents_bn_mean) / self.latents_bn_std
        lq_input = model_input.clone()
        hq_target = patchify_latents(hq_latent)
        hq_target = (hq_target - self.latents_bn_mean) / self.latents_bn_std
        model_input_ids = prepare_latent_ids(model_input).to(model_input.device)

        noise = torch.randn_like(model_input)

        fixed_idx = min(
            int(self.args.fixed_timestep), len(self.noise_scheduler.timesteps) - 1
        )
        timesteps = torch.full(
            (bsz,), fixed_idx, dtype=torch.long, device=model_input.device
        )

        sigmas = self._get_sigmas(
            timesteps, n_dim=model_input.ndim, dtype=model_input.dtype
        )
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
        packed_noisy_model_input = pack_latents(noisy_model_input)

        orig_input_shape = packed_noisy_model_input.shape
        orig_input_ids_shape = model_input_ids.shape

        guidance = torch.full(
            (bsz,),
            self.args.guidance_scale,
            device=self.accelerator.device,
            dtype=self.weight_dtype,
        )

        deg_token = self.deg_extractor(lq_pixel_values).unsqueeze(1)
        if self.accelerator.sync_gradients and global_step % 100 == 0:
            log_once(
                f"[Step {global_step}] deg_token  min={deg_token.min().item():.4f}  "
                f"max={deg_token.max().item():.4f}  mean={deg_token.mean().item():.4f}",
                self.accelerator,
            )
        deg_txt_id = text_ids[:, :1, :].clone()
        deg_txt_ids = torch.cat([deg_txt_id, text_ids], dim=1)

        model_pred = self.transformer(
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
        model_input_ids_trimmed = model_input_ids[:, : orig_input_ids_shape[1], :]
        model_pred = unpack_latents_with_ids(
            model_pred,
            model_input_ids_trimmed,
        )

        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme="none", sigmas=sigmas
        )
        # fm target:$\varepsilon$$\varepsilon + \frac{(1-t)x^{\mathrm{L}} - x^{\mathrm{H}}}{t}$
        target = noise + ((1 - sigmas) * lq_input - hq_target) / sigmas
        loss = torch.mean(
            (
                weighting.float() * (model_pred.float() - target.float()) ** 2
            ).reshape(model_pred.shape[0], -1),
            1,
        ).mean()

        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                self.transformer.parameters(), self.args.max_grad_norm
            )
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _run_validation(self, global_step):
        unwrapped_transformer = self._unwrap_model(self.transformer)
        from src.flux2.pipelines.flux2_klein import Flux2KleinIRPipeline
        val_pipeline = Flux2KleinIRPipeline(
            vae=self.vae,
            transformer=unwrapped_transformer,
            scheduler=self.noise_scheduler,
            text_encoder=None,
            tokenizer=None,
        )
        val_pipeline.deg_extractor = self.deg_extractor
        val_pipeline.to(device=self.accelerator.device, dtype=self.weight_dtype)

        from src.training.validation import validate
        validate(
            pipeline=val_pipeline,
            val_items=self.monitor_items,
            guidance_scale=self.args.guidance_scale,
            fixed_timestep=int(self.args.fixed_timestep),
            device=self.accelerator.device,
            output_dir=self.monitor_output_dir,
            global_step=global_step,
        )
        log_once(f"[ValMonitor] saved snapshot at step={global_step}", self.accelerator)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        if isinstance(self.args.fixed_timestep, list):
            self.args.fixed_timestep = self.args.fixed_timestep[0]
        if not isinstance(self.args.fixed_timestep, int):
            self.args.fixed_timestep = int(self.args.fixed_timestep)

        if self.args.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

        global_step = 0
        first_epoch = 0

        if self.args.resume_from_checkpoint:
            if self.args.resume_from_checkpoint != "latest":
                path = os.path.basename(self.args.resume_from_checkpoint)
            else:
                dirs = sorted(
                    [d for d in os.listdir(self.args.output_dir)
                     if d.startswith("checkpoint")],
                    key=lambda x: int(x.split("-")[1]),
                )
                path = dirs[-1] if dirs else None

            if path is None:
                log_once(
                    f"Checkpoint '{self.args.resume_from_checkpoint}' not found. Starting fresh.",
                    self.accelerator,
                )
                self.args.resume_from_checkpoint = None
                initial_global_step = 0
            else:
                log_once(f"Resuming from checkpoint {path}", self.accelerator)
                self.accelerator.load_state(os.path.join(self.args.output_dir, path))
                global_step = int(path.split("-")[1])
                initial_global_step = global_step
                first_epoch = global_step // self.num_update_steps_per_epoch
        else:
            initial_global_step = 0

        progress_bar = tqdm(
            range(0, self.args.max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            disable=not self.accelerator.is_local_main_process,
        )

        for epoch in range(first_epoch, self.args.num_train_epochs):
            self.transformer.train()

            for step, batch in enumerate(self.train_dataloader):
                models_to_accumulate = [self.transformer]
                with self.accelerator.accumulate(models_to_accumulate):
                    loss = self.training_step(batch, global_step)

                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if global_step % self.args.save_checkpointing_steps == 0:
                        if self.args.checkpoints_total_limit is not None:
                            checkpoints = sorted(
                                [d for d in os.listdir(self.args.output_dir)
                                 if d.startswith("checkpoint")],
                                key=lambda x: int(x.split("-")[1]),
                            )
                            if len(checkpoints) >= self.args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - self.args.checkpoints_total_limit + 1
                                )
                                removing = checkpoints[:num_to_remove]
                                log_once(
                                    f"Removing {len(removing)} old checkpoints "
                                    f"(limit={self.args.checkpoints_total_limit})",
                                    self.accelerator,
                                )
                                for rc in removing:
                                    shutil.rmtree(
                                        os.path.join(self.args.output_dir, rc)
                                    )
                        save_path = os.path.join(
                            self.args.output_dir, f"checkpoint-{global_step}"
                        )
                        self.accelerator.save_state(save_path)
                        log_once(f"Saved state to {save_path}", self.accelerator)

                    if self.val_monitor_steps > 0 and global_step % self.val_monitor_steps == 0 and self.accelerator.is_main_process:
                        self._run_validation(global_step)

                    logs = {
                        "loss": loss.detach().item(),
                        "lr": self.lr_scheduler.get_last_lr()[0],
                    }
                    progress_bar.set_postfix(**logs)
                    self.accelerator.log(logs, step=global_step)

                if global_step >= self.args.max_train_steps:
                    break

            self.accelerator.wait_for_everyone()

        # Final save
        if self.accelerator.is_main_process:
            unwrapped = self.accelerator.unwrap_model(self.transformer)
            state_dict = self.accelerator.get_state_dict(unwrapped)
            trainable_dict = {
                k: v.to("cpu") for k, v in state_dict.items() if v.requires_grad
            }
            if trainable_dict:
                path = os.path.join(self.args.output_dir, "modulation_weights.pt")
                torch.save(trainable_dict, path)
                log_once(
                    f"Saved {len(trainable_dict)} tensors "
                    f"({sum(v.numel() for v in trainable_dict.values()):,} params) "
                    f"to {path}",
                    self.accelerator,
                )

            for entry in os.scandir(self.args.output_dir):
                if entry.is_dir() and entry.name.startswith("checkpoint"):
                    for name in ["class_embedding_U.pt", "trainable_weights.pt"]:
                        p = os.path.join(entry.path, name)
                        if os.path.exists(p):
                            os.remove(p)
                            log_once(f"Cleaned legacy: {p}", self.accelerator)

            self.accelerator.end_training()
