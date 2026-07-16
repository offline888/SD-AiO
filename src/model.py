import glob
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from peft import LoraConfig
from transformers import AutoTokenizer, CLIPTextModel
from typing import Optional, Literal

class VAEHook:
    def __init__(self, net, tile_size, is_decoder, fast_decoder=True, fast_encoder=True,
                 color_fix=False, to_gpu=True):
        self.net = net
        self.tile_size = tile_size
        self.is_decoder = is_decoder
        self.fast_mode = (fast_encoder and not is_decoder) or (fast_decoder and is_decoder)
        self.color_fix = color_fix and not is_decoder
        self.to_gpu = to_gpu
        self.pad = 11 if is_decoder else 32

    def __call__(self, x):
        orig_device = next(self.net.parameters()).device
        try:
            if self.to_gpu:
                self.net.to(x.device)
            if max(x.shape[2], x.shape[3]) <= self.pad * 2 + self.tile_size:
                return self.net.original_forward(x)
            return self.tile_forward(x)
        finally:
            self.net.to(orig_device)

    def tile_forward(self, z):
        device = z.device
        N, H, W = z.shape[0], z.shape[2], z.shape[3]
        tile_size = min(self.tile_size, min(H, W))
        stride = tile_size - self.pad

        grid_rows = max(1, (H - self.pad + stride - 1) // stride)
        grid_cols = max(1, (W - self.pad + stride - 1) // stride)

        scale = 8 if self.is_decoder else 1
        out_pad = 32 if self.is_decoder else 0
        out_h, out_w = H * scale, W * scale
        out_ch = z.shape[1] if not self.is_decoder else 3
        result = torch.zeros((N, out_ch, out_h, out_w), device=device, dtype=z.dtype)
        tile_count = torch.zeros((N, out_ch, out_h, out_w), device=device, dtype=torch.float32)

        def offset(i, n, total):
            return total - tile_size if i == n - 1 else max(i * stride - self.pad * i, 0)

        for r in range(grid_rows):
            for c in range(grid_cols):
                y0, x0 = offset(r, grid_rows, H), offset(c, grid_cols, W)
                y1, x1 = y0 + tile_size, x0 + tile_size

                y0_out, x0_out = max(y0 * scale - out_pad, 0), max(x0 * scale - out_pad, 0)
                y1_out, x1_out = min(y1 * scale + out_pad, out_h), min(x1 * scale + out_pad, out_w)

                pad_l, pad_t = max(0, -x0), max(0, -y0)
                pad_r, pad_b = max(0, x1 - W), max(0, y1 - H)
                tile = z[:, :, y0:y1, x0:x1]
                if pad_t or pad_l or pad_r or pad_b:
                    tile = F.pad(tile, (pad_l, pad_r, pad_t, pad_b), mode="replicate")

                with torch.cuda.amp.autocast(enabled=(z.dtype == torch.float16), dtype=torch.float16):
                    out_tile = self.net.original_forward(tile)

                result[:, :, y0_out:y1_out, x0_out:x1_out] += out_tile
                tile_count[:, :, y0_out:y1_out, x0_out:x1_out] += 1
                del tile, out_tile
                torch.cuda.empty_cache()

        return result / tile_count.clamp_min(1e-6)


class SDSingleStepRestoration(nn.Module):
    LORA_TARGETS = {
        "only_attn": ["to_k", "to_q", "to_v", "to_out.0"],
        "only_mlp": ["conv", "conv1", "conv2", "conv_shortcut", "conv_out",
                     "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"],
        "full": ["to_k", "to_q", "to_v", "to_out.0",
                 "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
                 "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"],
    }
    VAE_TARGET = r"^(encoder|decoder)\..*(conv1|conv2|conv_in|conv_shortcut|conv|conv_out|to_k|to_q|to_v|to_out\.0)$"

    def __init__(self, sd_path, lora_rank_unet=None, lora_rank_vae=None,
                 lora_rank_vae_encoder=None, lora_rank_vae_decoder=None,
                 lora_unet_strategy: Optional[Literal["only_attn", "only_mlp", "full"]] = "full",
                 num_inference_steps=1, enable_xformers=False,
                 enable_vae_tile=False, vae_tile_size=512, merge_lora=False):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")
        self.noise_scheduler = DDPMScheduler.from_pretrained(sd_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(num_inference_steps)

        if lora_rank_vae is not None:
            lora_rank_vae_encoder = lora_rank_vae_encoder or lora_rank_vae
            lora_rank_vae_decoder = lora_rank_vae_decoder or lora_rank_vae

        self.vae_config = self.vae.config
        self.train_unet_conv_in = False
        self.lora_rank_unet = lora_rank_unet
        self.lora_rank_vae_encoder = lora_rank_vae_encoder
        self.lora_rank_vae_decoder = lora_rank_vae_decoder
        self.num_inference_steps = num_inference_steps

        if lora_rank_unet:
            self.add_lora_to_unet(lora_rank_unet, strategy=lora_unet_strategy)
        if lora_rank_vae_encoder:
            self.add_lora_to_vae(lora_rank_vae_encoder, "encoder", "vae_encoder_restoration_lora")
        if lora_rank_vae_decoder:
            self.add_lora_to_vae(lora_rank_vae_decoder, "decoder", "vae_decoder_restoration_lora")
        if merge_lora:
            self.merge_lora()

        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)

        if enable_xformers:
            try:
                self.unet.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(f"[SDSingleStepRestoration] xformers enable failed: {e}")

        if enable_vae_tile:
            self.init_tiled_vae(vae_tile_size)

    def merge_lora(self):
        for owner in (self.unet, self.vae):
            if owner.peft_config:
                for name in owner.peft_config:
                    set_weights_and_activate_adapters(owner, [name], [1.0])
                owner.merge_and_unload()

    def init_tiled_vae(self, tile_size=512):
        for net, is_decoder in [(self.vae.encoder, False), (self.vae.decoder, True)]:
            if not hasattr(net, "original_forward"):
                setattr(net, "original_forward", net.forward)
            net.forward = VAEHook(net, tile_size, is_decoder)

    def add_lora_to_vae(self, rank, subfolder, adapter_name):
        config = LoraConfig(r=rank, init_lora_weights="gaussian",
                            target_modules=self.VAE_TARGET.replace("^(encoder|decoder)", f"^{subfolder}"))
        self.vae.add_adapter(config, adapter_name=adapter_name)
        print(f"[Lora] Added to vae {subfolder}, rank={rank}")

    def add_lora_to_unet(self, rank, strategy=None):
        if strategy not in self.LORA_TARGETS:
            raise ValueError(f"Invalid lora strategy: {strategy}")
        config = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=self.LORA_TARGETS[strategy])
        self.unet.add_adapter(config, adapter_name="unet_restoration_lora")
        print(f"[Lora] Added to unet, rank={rank}, strategy={strategy}")

    def free_text_encoder(self):
        self.text_encoder.cpu()
        torch.cuda.empty_cache()

    # def is_trainable_param(self, name):
    #    return "lora" in name or "conv_in" in name
    def is_trainable_param(self, name):
        if "lora" in name:
            return True
        if self.train_unet_conv_in and "conv_in" in name:
            return True
        return False

    def set_trainable_params(self):
        self.unet.train(); self.vae.train()
        for n, p in self.unet.named_parameters():
            p.requires_grad = self.is_trainable_param(n)
        for n, p in self.vae.named_parameters():
            p.requires_grad = "lora" in n

    def set_train(self): self.set_trainable_params()

    def set_eval(self):
        self.unet.eval(); self.vae.eval(); self.merge_lora()

    def trainable_parameters(self):
        params = [p for n, p in self.unet.named_parameters() if self.is_trainable_param(n)]
        params += [p for n, p in self.vae.named_parameters() if "lora" in n]
        return params

    def encode_image(self, image):
        return self.vae.encode(image).latent_dist.sample() * self.vae_config.scaling_factor

    def decode_latent(self, latent):
        return self.vae.decode(latent / self.vae_config.scaling_factor).sample.clamp(-1, 1)

    def eps_to_coeff(self, timesteps):
        a = self.noise_scheduler.alphas_cumprod.to(device=timesteps.device, dtype=torch.float32)[timesteps]
        a = a.view(-1, 1, 1, 1)
        return ((1 - a) ** 0.5) / (a ** 0.5)

    def forward(self, lq_image, text_embedding, timestep=None,
                timestep_range=(50, 150), cond_module=None,
                deg_extractor=None, pretrained_encoder=None):
        unet_p = next(self.unet.parameters())
        vae_p = next(self.vae.parameters())
        enc = pretrained_encoder or getattr(self, 'pretrained_encoder', None)
        deg_ext = deg_extractor or getattr(self, 'deg_extractor', None)

        if enc is not None and deg_ext is not None:
            f_deg = deg_ext.get_deg_feat(lq_image)
            z_raw = enc(lq_image.to(device=vae_p.device, dtype=vae_p.dtype),
                        f_deg.to(device=vae_p.device, dtype=vae_p.dtype))
            z_mean = self.vae.quant_conv(z_raw)[:, :4]
            encoded_latent = z_mean * self.vae_config.scaling_factor
            encoded_latent = encoded_latent.to(device=unet_p.device, dtype=unet_p.dtype)
        else:
            f_deg = None
            encoded_latent = self.encode_image(lq_image.to(device=vae_p.device, dtype=vae_p.dtype))
            encoded_latent = encoded_latent.to(device=unet_p.device, dtype=unet_p.dtype)

        noise = torch.randn_like(encoded_latent)
        if timestep is None or timestep <= 0:
            timestep = torch.randint(timestep_range[0], timestep_range[1] + 1, (1,)).item()
        t_tensor = torch.full((encoded_latent.shape[0],), timestep,
                              device=unet_p.device, dtype=torch.long)

        noisy_latent = self.noise_scheduler.add_noise(encoded_latent, noise, t_tensor)

        if text_embedding is None:
            raise ValueError("text_embedding must not be None")
        text_embedding = text_embedding.to(device=unet_p.device, dtype=unet_p.dtype)

        if cond_module is not None:
            _, text_embedding = cond_module.get_modulation(
                lq_image, text_embedding=text_embedding, timestep=timestep, f_deg=f_deg)

        noise_pred = self.unet(noisy_latent, t_tensor, encoder_hidden_states=text_embedding).sample
        coeff = self.eps_to_coeff(t_tensor).to(device=unet_p.device, dtype=unet_p.dtype)
        denoised = (encoded_latent + coeff * (noise - noise_pred)).to(device=vae_p.device, dtype=vae_p.dtype)
        return self.decode_latent(denoised)

    def save_state(self, path, optimizer=None, lr_scheduler=None, scaler=None,
                   epoch=0, global_step=0, weights_only=False):
        unet_sd = {k: v.cpu() for k, v in self.unet.named_parameters()
                   if (v.requires_grad if weights_only else self.is_trainable_param(k))}
        vae_sd = {k: v.cpu() for k, v in self.vae.named_parameters()
                  if "lora" in k}

        checkpoint = {
            "state_dict_unet": unet_sd,
            "state_dict_vae": vae_sd,
            "rank_unet": getattr(self, "lora_rank_unet", None),
            "rank_vae_encoder": getattr(self, "lora_rank_vae_encoder", None),
            "rank_vae_decoder": getattr(self, "lora_rank_vae_decoder", None),
            "epoch": epoch, "global_step": global_step,
        }
        if optimizer is not None: checkpoint["optimizer"] = optimizer.state_dict()
        if lr_scheduler is not None: checkpoint["lr_scheduler"] = lr_scheduler.state_dict()
        if scaler is not None: checkpoint["scaler"] = scaler.state_dict()
        torch.save(checkpoint, path)
        print(f"[Save] {path} (step {global_step})")

    def save_checkpoint(self, path, optimizer=None, lr_scheduler=None, scaler=None,
                        epoch=0, global_step=0):
        self.save_state(path, optimizer, lr_scheduler, scaler, epoch, global_step, weights_only=False)

    def save_weights(self, path):
        self.save_state(path, weights_only=True)

    def load_state(self, path, optimizer=None, lr_scheduler=None, scaler=None, weights_only=True):
        map_loc = "cpu"
        ckpt = torch.load(path, map_location=map_loc, weights_only=weights_only)
        if ckpt.get("state_dict_unet"):
            self.unet.load_state_dict(ckpt["state_dict_unet"], strict=False)
        if ckpt.get("state_dict_vae"):
            self.vae.load_state_dict(ckpt["state_dict_vae"], strict=False)
        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if lr_scheduler is not None and "lr_scheduler" in ckpt:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        if scaler is not None and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        return ckpt.get("epoch", 0), ckpt.get("global_step", 0)

    def load_checkpoint(self, path, optimizer=None, lr_scheduler=None, scaler=None):
        epoch, step = self.load_state(path, optimizer, lr_scheduler, scaler, weights_only=False)
        print(f"[Load] {path} (epoch {epoch}, step {step})")
        return epoch, step

    def load_weights(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        for owner, key in ((self.unet, "state_dict_unet"), (self.vae, "state_dict_vae")):
            if ckpt.get(key):
                _, unexpected = owner.load_state_dict(ckpt[key], strict=False)
                if unexpected:
                    print(f"[Warning] {owner.__class__.__name__}: skipped {len(unexpected)} keys; first 3: {unexpected[:3]}")
        print(f"[Load] {path}")

    @staticmethod
    def step_num(path):
        m = re.search(r'step_(\d+)', os.path.basename(path))
        return int(m.group(1)) if m else 0

    @classmethod
    def latest_ckpt(cls, run_dir, glob_pat):
        files = sorted(glob.glob(os.path.join(run_dir, glob_pat)))
        return max(files, key=cls.step_num) if files else None

    def save_training_state(self, run_dir, step, cond_module, optimizer, lr_scheduler):
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        self.save_weights(os.path.join(ckpt_dir, f"step_{step}.pkl"))
        trainable_keys = {n for n, p in cond_module.named_parameters() if p.requires_grad}
        cm_state = {k: v for k, v in cond_module.state_dict().items() if k in trainable_keys}
        torch.save(cm_state, os.path.join(ckpt_dir, f"cond_module_{step}.pth"))
        torch.save({"optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "optimizer_step": step},
                   os.path.join(ckpt_dir, f"training_state_{step}.pt"))

    def load_training_state(self, run_dir, cond_module, optimizer=None, lr_scheduler=None):
        model_ckpt = self.latest_ckpt(run_dir, "step_*.pkl")
        cm_ckpt = self.latest_ckpt(run_dir, "cond_module_*.pth")
        state_ckpt = self.latest_ckpt(run_dir, "training_state_*.pt")
        if not (model_ckpt and cm_ckpt and state_ckpt):
            return 0
        self.load_checkpoint(model_ckpt)
        cm_state = torch.load(cm_ckpt, map_location="cpu")
        if not cm_state:
            raise RuntimeError(f"Empty cond_module checkpoint: {cm_ckpt}")
        missing, unexpected = cond_module.load_state_dict(cm_state, strict=False)
        if missing or unexpected:
            print(f"[Load] cond_module: {len(missing)} missing, {len(unexpected)} unexpected")
            if missing: print(f"  missing[:3]: {missing[:3]}")
            if unexpected: print(f"  unexpected[:3]: {unexpected[:3]}")
        state = torch.load(state_ckpt, map_location="cpu")
        if optimizer is not None:
            optimizer.load_state_dict(state["optimizer"])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(state["lr_scheduler"])
        return state["optimizer_step"]