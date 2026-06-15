import torch
import torch.nn as nn
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from peft import LoraConfig
from transformers import AutoTokenizer, CLIPTextModel

class VAEHook:
    def __init__(
        self,
        net,
        tile_size,
        is_decoder,
        fast_decoder=True,
        fast_encoder=True,
        color_fix=False,
        to_gpu=True,
    ):
        self.net = net
        self.tile_size = tile_size
        self.is_decoder = is_decoder
        self.fast_mode = (fast_encoder and not is_decoder) or (
            fast_decoder and is_decoder
        )
        self.color_fix = color_fix and not is_decoder
        self.to_gpu = to_gpu
        self.pad = 11 if is_decoder else 32
        self._orig_forward = None

    def __call__(self, x):
        B, C, H, W = x.shape
        orig_dev = next(self.net.parameters()).device
        try:
            if self.to_gpu:
                self.net.to(x.device)
            if max(H, W) <= self.pad * 2 + self.tile_size:
                return self.net.original_forward(x)
            return self._tile_forward(x)
        finally:
            self.net.to(orig_dev)

    def _tile_forward(self, z):
        import torch.nn.functional as F

        device = z.device
        N, H, W = z.shape[0], z.shape[2], z.shape[3]

        tile_size = min(self.tile_size, min(H, W))
        overlap = self.pad
        stride = tile_size - overlap

        grid_rows = max(1, (H - overlap + stride - 1) // stride)
        grid_cols = max(1, (W - overlap + stride - 1) // stride)

        def offset(i, n, total):
            if i < n - 1:
                return max(i * stride - overlap * i, 0)
            return total - tile_size

        out_h = H * 8 if self.is_decoder else H // 8
        out_w = W * 8 if self.is_decoder else W // 8
        result = torch.zeros(
            (N, z.shape[1] if not self.is_decoder else 3, out_h, out_w),
            device=device,
            dtype=z.dtype,
        )

        tile_count = torch.zeros(
            (N, z.shape[1] if not self.is_decoder else 3, out_h, out_w),
            device=device,
            dtype=torch.float32,
        )

        tile_h = tile_size
        tile_w = tile_size

        for r in range(grid_rows):
            for c in range(grid_cols):
                y0 = offset(r, grid_rows, H)
                x0 = offset(c, grid_cols, W)
                y1, x1 = y0 + tile_h, x0 + tile_w

                y0_out = max(
                    y0 * (8 if self.is_decoder else 1) - (32 if self.is_decoder else 0),
                    0,
                )
                x0_out = max(
                    x0 * (8 if self.is_decoder else 1) - (32 if self.is_decoder else 0),
                    0,
                )
                y1_out = min(
                    y1 * (8 if self.is_decoder else 1) + (32 if self.is_decoder else 0),
                    out_h,
                )
                x1_out = min(
                    x1 * (8 if self.is_decoder else 1) + (32 if self.is_decoder else 0),
                    out_w,
                )

                in_box_h = y1 - y0
                in_box_w = x1 - x0

                pad_l = max(0, -x0)
                pad_t = max(0, -y0)
                pad_r = max(0, x1 - W)
                pad_b = max(0, y1 - H)

                tile = z[:, :, y0:y1, x0:x1]
                if pad_t or pad_l or pad_r or pad_b:
                    tile = torch.nn.functional.pad(
                        tile, (pad_l, pad_r, pad_t, pad_b), mode="replicate"
                    )

                with torch.cuda.amp.autocast(
                    enabled=(z.dtype == torch.float16), dtype=torch.float16
                ):
                    out_tile = self.net.original_forward(tile)

                oy0, ox0 = 0, 0
                oy1, ox1 = out_tile.shape[2], out_tile.shape[3]

                out_crop = out_tile[:, :, oy0:oy1, ox0:ox1]
                result[:, :, y0_out:y1_out, x0_out:x1_out] += out_crop
                tile_count[:, :, y0_out:y1_out, x0_out:x1_out] += 1

                del tile, out_tile
                torch.cuda.empty_cache()

        return result / tile_count.clamp_min(1e-6)


class SDSingleStepRestoration(nn.Module):
    def __init__(
        self,
        sd_path,
        lora_rank_unet: int = None,
        lora_rank_vae: int = None,
        num_inference_steps: int = 1,
        enable_xformers: bool = False,
        enable_vae_tile: bool = False,
        vae_tile_size: int = 512,
        merge_lora: bool = False,
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            sd_path, subfolder="text_encoder"
        )
        self.vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            sd_path, subfolder="scheduler"
        )
        self.noise_scheduler.set_timesteps(num_inference_steps)

        self.vae_config = self.vae.config
        self.lora_rank_unet = lora_rank_unet
        self.lora_rank_vae = lora_rank_vae
        self.num_inference_steps = num_inference_steps
        self.enable_vae_tile = enable_vae_tile

        if lora_rank_unet:
            self.add_lora_to_unet(lora_rank_unet)
        if lora_rank_vae:
            self.add_lora_to_vae(lora_rank_vae)
        if merge_lora:
            self._merge_lora()

        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.unet.conv_in.requires_grad_(True)

        if enable_xformers:
            try:
                self.unet.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

        if enable_vae_tile:
            self._init_tiled_vae(vae_tile_size)

    def _merge_lora(self):
        if self.lora_rank_unet:
            for name in self.unet.peft_config:
                set_weights_and_activate_adapters(self.unet, [name], [1.0])
            self.unet.merge_and_unload()
        if self.lora_rank_vae:
            for name in self.vae.peft_config:
                set_weights_and_activate_adapters(self.vae, [name], [1.0])
            self.vae.merge_and_unload()

    def _init_tiled_vae(self, tile_size=512):
        for net, is_dec in [(self.vae.encoder, False), (self.vae.decoder, True)]:
            if not hasattr(net, "original_forward"):
                setattr(net, "original_forward", net.forward)
            net.forward = VAEHook(net, tile_size, is_dec)

    def add_lora_to_vae(self, rank):
        target = r"^encoder\..*(conv1|conv2|conv_in|conv_shortcut|conv|conv_out|to_k|to_q|to_v|to_out\.0)$"
        self.vae.add_adapter(
            LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=target),
            adapter_name="vae_restoration",
        )

    def add_lora_to_unet(self, rank):
        modules = [
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
            "conv",
            "conv1",
            "conv2",
            "conv_shortcut",
            "conv_out",
            "proj_in",
            "proj_out",
            "ff.net.2",
            "ff.net.0.proj",
        ]
        self.unet.add_adapter(
            LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=modules)
        )

    def free_text_encoder(self):
        self.text_encoder.cpu()
        torch.cuda.empty_cache()

    def set_train(self):
        self.unet.train()
        self.vae.train()
        for n, p in self.unet.named_parameters():
            p.requires_grad = "lora" in n or "conv_in" in n
        for n, p in self.vae.named_parameters():
            p.requires_grad = "lora" in n

    def set_eval(self, merge_lora=False):
        # WARNING: merge_lora=True permanently removes LoRA adapters.
        # Calling set_train() after set_eval(merge_lora=True) will
        # silently lose all LoRA trainable parameters.
        self.unet.eval()
        self.vae.eval()
        if merge_lora:
            self._merge_lora()

    def trainable_parameters(self):
        params = []
        for n, p in self.unet.named_parameters():
            if "lora" in n or "conv_in" in n:
                params.append(p)
        for n, p in self.vae.named_parameters():
            if "lora" in n:
                params.append(p)
        return params

    def encode_image(self, image):
        latent = self.vae.encode(image).latent_dist.sample()
        return latent * self.vae_config.scaling_factor

    def decode_latent(self, latent):
        image = self.vae.decode(latent / self.vae_config.scaling_factor).sample
        return image.clamp(-1, 1)

    def _eps_to_coeff(self, timesteps):
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(dtype=torch.float32)
        a = alphas_cumprod[timesteps]
        for _ in range(3):
            a = a.unsqueeze(-1)
        beta = 1 - a
        return (beta**0.5) / (a**0.5)

    def forward(
        self,
        lq_image: torch.Tensor,
        text_embedding: torch.Tensor,
        timestep: int = None,
        cond_module=None,
    ):

        unet_param = next(self.unet.parameters())
        unet_device = unet_param.device
        unet_dtype = unet_param.dtype

        vae_param = next(self.vae.parameters())
        vae_device = vae_param.device
        vae_dtype = vae_param.dtype

        lq_image = lq_image.to(device=vae_device, dtype=vae_dtype)
        encoded = self.encode_image(lq_image)
        encoded = encoded.to(device=unet_device, dtype=unet_dtype)

        if timestep is not None and timestep > 0:
            noise = torch.randn_like(encoded)
            t_tensor = torch.tensor([timestep], device=unet_device, dtype=torch.long)
            noisy = self.noise_scheduler.add_noise(encoded, noise, t_tensor)
        else:
            noisy = encoded
            noise = torch.zeros_like(encoded)

        noisy = noisy.to(device=unet_device, dtype=unet_dtype)
        noise = noise.to(device=unet_device, dtype=unet_dtype)

        t_tensor = torch.tensor(
            [timestep if timestep is not None else 999], device=unet_device, dtype=torch.long
        )

        if text_embedding is None:
            raise ValueError("text_embedding must not be None")
        text_embedding = text_embedding.to(device=unet_device, dtype=unet_dtype)

        if cond_module is not None:
            cond_module.get_modulation(lq_image, timestep=timestep)

        noise_pred = self.unet(
            noisy, t_tensor, encoder_hidden_states=text_embedding
        ).sample

        coeff = self._eps_to_coeff(t_tensor).to(device=unet_device, dtype=unet_dtype)
        denoised = encoded + coeff * (noise - noise_pred)

        denoised = denoised.to(device=vae_device, dtype=vae_dtype)
        return self.decode_latent(denoised)

    def save_checkpoint(self, path):
        sd = {
            "state_dict_unet": {},
            "state_dict_vae": {},
            "rank_unet": self.lora_rank_unet,
            "rank_vae": self.lora_rank_vae,
        }
        if self.lora_rank_unet:
            for k, v in self.unet.state_dict().items():
                if "lora" in k:
                    sd["state_dict_unet"][k] = v
        for k, v in self.unet.state_dict().items():
            if "conv_in" in k:
                sd["state_dict_unet"][k] = v
        if self.lora_rank_vae:
            for k, v in self.vae.state_dict().items():
                if "lora" in k:
                    sd["state_dict_vae"][k] = v
        torch.save(sd, path)

    def load_checkpoint(self, path, strict=False):
        sd = torch.load(path, map_location="cpu", weights_only=False)
        unet_sd = self.unet.state_dict()
        ckpt_unet_keys = set(sd.get("state_dict_unet", {}))
        skipped = ckpt_unet_keys - set(unet_sd)
        for k in sd.get("state_dict_unet", {}):
            if k in unet_sd:
                unet_sd[k] = sd["state_dict_unet"][k]
        self.unet.load_state_dict(unet_sd, strict=strict)
        vae_sd = self.vae.state_dict()
        ckpt_vae_keys = set(sd.get("state_dict_vae", {}))
        skipped |= ckpt_vae_keys - set(vae_sd)
        for k in sd.get("state_dict_vae", {}):
            if k in vae_sd:
                vae_sd[k] = sd["state_dict_vae"][k]
        self.vae.load_state_dict(vae_sd, strict=strict)
        if skipped:
            print(f"[load_checkpoint] Warning: {len(skipped)} keys in checkpoint "
                  f"not found in model — silently skipped."
                  f"\n  (e.g. LoRA keys from ckpt but current model has no LoRA)")
            for name in sorted(skipped)[:5]:
                print(f"    - {name}")
            if len(skipped) > 5:
                print(f"    ... and {len(skipped) - 5} more")
