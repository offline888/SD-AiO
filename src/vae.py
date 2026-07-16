import copy
import torch
import torch.nn as nn


class AdaIN(nn.Module):
    """F_Deg → MLP → (γ, β) per-channel modulation."""

    def __init__(self, cond_dim: int, channel_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(cond_dim, channel_dim),
            nn.SiLU(),
            nn.Linear(channel_dim, channel_dim * 2),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x: torch.Tensor, f_deg: torch.Tensor):
        factors = self.proj(f_deg).unsqueeze(-1).unsqueeze(-1)  # [B, 2C, 1, 1]
        gamma, beta = factors.chunk(2, dim=1)
        return x * (1 + gamma) + beta


class PreRestoreEncoder(nn.Module):

    def __init__(self, encoder, block_out_channels, cond_dim=768, adaln_layers=None):
        super().__init__()
        self.encoder = copy.deepcopy(encoder)
        self.adaln = nn.ModuleDict()
        self._adaln_names = adaln_layers or []

        # Channel dim at each injection point
        channel = {}
        for i in range(4):
            channel[f"down{i}"] = block_out_channels[min(i + 1, 3)]
        channel["mid"] = block_out_channels[-1]

        for name in self._adaln_names:
            if name not in channel:
                raise ValueError(f"Unknown adaln layer: {name}, expected one of {list(channel.keys())}")
            self.adaln[name] = AdaIN(cond_dim, channel[name])

    def forward(self, x, f_deg):
        x = self.encoder.conv_in(x)
        for i in range(4):
            x = self.encoder.down_blocks[i](x)
            name = f"down{i}"
            if name in self.adaln:
                x = self.adaln[name](x, f_deg)
        x = self.encoder.mid_block(x)
        if "mid" in self.adaln:
            x = self.adaln["mid"](x, f_deg)
        x = self.encoder.conv_norm_out(x)
        x = self.encoder.conv_act(x)
        x = self.encoder.conv_out(x)
        return x
