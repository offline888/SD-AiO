import ast
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

from diffusers import AutoencoderKLFlux2
from diffusers.models.embeddings import TimestepEmbedding, Timesteps


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
        x = self.expand(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.project(x)
        return x


class TimeModulator(nn.Module):
    def __init__(self, in_channels: int, time_emb_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, in_channels * 2))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        scale_shift = self.mlp(temb)
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
            # Convert all ConvNeXt submodules to weight_dtype for mixed-precision consistency
            self.conv_stem_s1 = convnext.features[:2]
            self.conv_down1_s2 = convnext.features[2:4]
            self.conv_down2_s3 = convnext.features[4:6]

            for mod in [self.conv_stem_s1, self.conv_down1_s2, self.conv_down2_s3]:
                mod.to(dtype=torch.bfloat16)

            self.conv_time_mod1 = TimeModulator(in_channels=96, time_emb_dim=dim)
            self.conv_time_mod2 = TimeModulator(in_channels=192, time_emb_dim=dim)
            self.conv_time_mod3 = TimeModulator(in_channels=384, time_emb_dim=dim)

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

            self.vae_time_mods = nn.ModuleList(
                [
                    TimeModulator(in_channels=ch, time_emb_dim=dim)
                    for ch in vae_out_channels
                ]
            )
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
        seq_len: int | None = None,
    ) -> torch.Tensor:
        B = temb.size(0)
        w_dtype = self.linear.weight.dtype

        if self.use_block_emb and block_idx is not None:
            if isinstance(block_idx, int):
                block_tensor = torch.tensor([block_idx], dtype=torch.long, device=temb.device)
            else:
                block_tensor = block_idx.long().to(device=temb.device)
            bemb = self.block_proj(block_tensor)
            bemb = self.block_embedder(bemb.to(dtype=w_dtype))
            temb = (temb.to(dtype=w_dtype) + bemb)

        mod_time = self.act_fn(temb)
        mod_time = self.linear(mod_time)

        lq_mod = None
        if lq_tensor is not None:
            if self.use_conv and not self.use_vae:
                lq_tensor = lq_tensor.to(dtype=torch.bfloat16)
                x = self.conv_stem_s1(lq_tensor)
                x = self.conv_time_mod1(x, temb)
                x = self.conv_down1_s2(x)
                x = self.conv_time_mod2(x, temb)
                x = self.conv_down2_s3(x)
                x = self.conv_time_mod3(x, temb)

                lq_mod = self.feat_proj(x)
                lq_mod = lq_mod.permute(0, 2, 3, 1).reshape(B, -1, 3 * self.mod_param_sets * self.dim)

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

                lq_feat, _ = torch.chunk(x, 2, dim=1)
                lq_mod = self.vae_proj(lq_feat)
                lq_mod = lq_mod.permute(0, 2, 3, 1).reshape(B, -1, 3 * self.mod_param_sets * self.dim)

        if seq_len is not None:
            mod = mod_time.unsqueeze(1).expand(B, seq_len, -1)
            if lq_mod is not None:
                lq_mod_up = F.interpolate(
                    lq_mod.transpose(1, 2),
                    size=seq_len,
                    mode='linear',
                    align_corners=False,
                ).transpose(1, 2)
                mod = mod + lq_mod_up
        else:
            mod = mod_time.unsqueeze(1)
            if lq_mod is not None:
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
