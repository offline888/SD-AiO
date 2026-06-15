import torch
import torch.nn as nn
import torch.nn.functional as F

from degnet import DegFeatExtractor


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class BaseConditionModule(nn.Module):
    def setup(self, unet):
        pass

    def get_modulation(self, lq_image, timestep=None):
        raise NotImplementedError

    def forward(self, lq_image):
        raise NotImplementedError


class IdentityConditionModule(BaseConditionModule):
    def __init__(self, **kwargs):
        super().__init__()

    def get_modulation(self, lq_image, timestep=None):
        return None, None

    def forward(self, lq_image):
        return None, None


# ═══════════════════════════════════════════════════════════════
#  SFT Layer
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
#  Design 1b — CODSR LQFM (LQ-guided Feature Modulation)
#   LQ → pixel_unshuffle → shared encoder → per-layer SFT → γ,β
#   Time-aware: λ_t = √ᾱ_t / √(1-ᾱ_t) scales modulation strength
#   F' = F × (1 + λ_t · γ) + λ_t · β
# ═══════════════════════════════════════════════════════════════

class CODSRFiLMModule(BaseConditionModule):
    """
    CODSR-style full-layer modulation with pixel-unshuffle fidelity preservation.
    Shared encoder runs once per forward; per-layer SFT heads are lightweight.
    """

    def __init__(self, embed_dim=128, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        # Shared encoder: pixel_unshuffle'd LQ (192ch @ H/8×W/8) → compact feature
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(192, 256, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(256, embed_dim, 3, padding=1),
            nn.SiLU(),
        )
        self.sft_layers = nn.ModuleDict()
        self._convs = {}
        self._hooked = False

    def _make_sft_head(self, out_ch):
        """Per-layer lightweight SFT: shared feature → scale + shift."""
        return nn.ModuleDict({
            "scale": nn.Sequential(
                nn.Conv2d(self.embed_dim, out_ch, 1), nn.SiLU(),
                zero_module(nn.Conv2d(out_ch, out_ch, 1)),
            ),
            "shift": nn.Sequential(
                nn.Conv2d(self.embed_dim, out_ch, 1), nn.SiLU(),
                zero_module(nn.Conv2d(out_ch, out_ch, 1)),
            ),
        })

    def _hook_conv(self, conv, hook_name):
        self._convs[hook_name] = conv
        orig = conv.forward

        def hooked(x):
            out = orig(x)
            scale = getattr(conv, "_sft_scale", None)
            if scale is not None:
                shift = conv._sft_shift
                if scale.shape[2:] != out.shape[2:]:
                    scale = F.interpolate(scale, size=out.shape[2:], mode="bilinear", align_corners=False)
                    shift = F.interpolate(shift, size=out.shape[2:], mode="bilinear", align_corners=False)
                out = out * (1 + scale) + shift
                conv._sft_scale = None
                conv._sft_shift = None
            return out

        conv.forward = hooked

    def setup(self, unet):
        if self._hooked:
            return
        self._hooked = True

        device = unet.conv_in.weight.device

        def register(name, conv):
            self.sft_layers[name] = self._make_sft_head(conv.out_channels).to(device)

        register("conv_in", unet.conv_in);  self._hook_conv(unet.conv_in, "conv_in")
        for down_idx, block in enumerate(unet.down_blocks):
            for res_idx, resnet in enumerate(block.resnets):
                conv = getattr(resnet, "conv2", None)
                if conv is not None:
                    n = f"down_{down_idx}_res_{res_idx}"; register(n, conv); self._hook_conv(conv, n)
            for ds_idx, ds in enumerate(getattr(block, "downsamplers", []) or []):
                conv = getattr(ds, "conv", None)
                if conv is not None:
                    n = f"down_{down_idx}_ds_{ds_idx}"; register(n, conv); self._hook_conv(conv, n)
        for res_idx, resnet in enumerate(getattr(unet.mid_block, "resnets", [])):
            conv = getattr(resnet, "conv2", None)
            if conv is not None:
                n = f"mid_res_{res_idx}"; register(n, conv); self._hook_conv(conv, n)
        for up_idx, block in enumerate(unet.up_blocks):
            for res_idx, resnet in enumerate(block.resnets):
                conv = getattr(resnet, "conv2", None)
                if conv is not None:
                    n = f"up_{up_idx}_res_{res_idx}"; register(n, conv); self._hook_conv(conv, n)
            for us_idx, us in enumerate(getattr(block, "upsamplers", []) or []):
                conv = getattr(us, "conv", None)
                if conv is not None:
                    n = f"up_{up_idx}_us_{us_idx}"; register(n, conv); self._hook_conv(conv, n)

    def get_modulation(self, lq_image, timestep=None):
        # 1. Pixel-unshuffle: (B,3,H,W) → (B,192,H/8,W/8) — lossless spatial→channel
        device = next(self.shared_encoder.parameters()).device
        lq = lq_image.to(device)
        xl = F.pixel_unshuffle(lq, downscale_factor=8)  # [B, 192, H/8, W/8]

        # 2. Shared encoder (once per forward)
        feat = self.shared_encoder(xl)  # [B, embed_dim, H/8, W/8]

        # 3. Time-aware λ_t = √ᾱ / √(1-ᾱ)
        #    Pre-multiplied into scale/shift so hooks stay unchanged
        if timestep is not None and timestep > 0:
            # DDPM ᾱ_t: rough approximation from timestep
            # ᾱ_t ≈ 1 - t/1000 (linear approximation for SD scheduler)
            alpha_bar = max(1.0 - timestep / 1000.0, 0.001)
            lam = (alpha_bar ** 0.5) / max((1 - alpha_bar) ** 0.5, 0.02)
            lam = min(lam, 10.0)  # cap to avoid exploding at small t
        else:
            lam = 1.0

        # 4. Per-layer SFT heads
        for name, sft_head in self.sft_layers.items():
            conv = self._convs[name]
            # Interpolate shared feat to match conv's expected spatial resolution
            # (the hook will interpolate again if needed, but we give it at H/8×W/8)
            scale = sft_head["scale"](feat)  # [B, out_ch, H/8, W/8]
            shift = sft_head["shift"](feat)  # [B, out_ch, H/8, W/8]
            conv._sft_scale = (lam * scale).to(device=conv.weight.device)
            conv._sft_shift = (lam * shift).to(device=conv.weight.device)
        return None, None

    def forward(self, lq_image):
        return self.get_modulation(lq_image)


# ═══════════════════════════════════════════════════════════════
#  Cross-Attention Block
# ═══════════════════════════════════════════════════════════════

class CrossAttnModulationBlock(nn.Module):
    def __init__(self, feat_in_dim, feat_out_dim, deg_dim):
        super().__init__()
        self.feat_in_dim = feat_in_dim
        self.feat_out_dim = feat_out_dim

        self.q_proj = nn.Conv2d(feat_in_dim, feat_out_dim, kernel_size=1)
        self.k_proj = nn.Linear(deg_dim, feat_out_dim)
        self.v_proj = nn.Linear(deg_dim, feat_out_dim)
        self.scale_proj = zero_module(nn.Linear(feat_out_dim, feat_out_dim))
        self.shift_proj = zero_module(nn.Linear(feat_out_dim, feat_out_dim))
        self.norm = nn.LayerNorm(feat_out_dim)

        self._deg_k = None
        self._deg_v = None

    def set_deg(self, deg_feat):
        proj_weight = self.k_proj.weight
        deg_feat = deg_feat.to(device=proj_weight.device, dtype=proj_weight.dtype)
        self._deg_k = self.k_proj(deg_feat)
        self._deg_v = self.v_proj(deg_feat)

    def forward(self, feat):
        q_weight = self.q_proj.weight
        feat = feat.to(device=q_weight.device, dtype=q_weight.dtype)

        b, _, h, w = feat.shape
        q = self.q_proj(feat).permute(0, 2, 3, 1).reshape(b, h * w, self.feat_out_dim)
        k = self._deg_k.unsqueeze(1)
        v = self._deg_v.unsqueeze(1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.feat_out_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v).mean(dim=1)
        out = self.norm(out + q.mean(dim=1))

        scale = self.scale_proj(out).view(b, self.feat_out_dim, 1, 1)
        shift = self.shift_proj(out).view(b, self.feat_out_dim, 1, 1)
        return scale, shift


# ═══════════════════════════════════════════════════════════════
#  Design 1a — Degradation Cross-Attention (feature pyramid + CrossAttn)
# ═══════════════════════════════════════════════════════════════

class DegCrossAttnModule(BaseConditionModule):
    def __init__(self, inner_dim=768, kv_dim=128, args=None, **kwargs):
        super().__init__()
        self.inner_dim = inner_dim
        self.kv_dim = kv_dim
        self._deg_extractor = None
        self._args = args
        self._hooked = False
        self.hook_blocks = nn.ModuleDict()

        self.feature_pyramid = nn.ModuleDict({
            "4": nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=1, padding=1), nn.SiLU(),
                nn.Conv2d(32, 4, 3, stride=1, padding=1), nn.SiLU(),
            ),
            "320": nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=1, padding=1), nn.SiLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(128, 320, 3, stride=1, padding=1), nn.SiLU(),
            ),
            "640": nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=1, padding=1), nn.SiLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(128, 320, 3, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(320, 640, 3, stride=1, padding=1), nn.SiLU(),
            ),
            "1280": nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=1, padding=1), nn.SiLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(128, 320, 3, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(320, 640, 3, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(640, 1280, 3, stride=1, padding=1), nn.SiLU(),
            ),
        })

    def build_deg_extractor(self, device):
        if self._deg_extractor is not None:
            return
        self._deg_extractor = DegFeatExtractor(
            inner_dim=self.inner_dim,
            num_deg_types=self._args.num_deg_types,
            weight_dtype=torch.float32,
            args=self._args,
            deg_embedding=None,
            device=device,
        )
        # DegFeatExtractor.__init__ already froze everything via requires_grad_(False).
        # Optionally unfreeze decoder for fine-tuning:
        if not getattr(self._args, "freeze_decoder", True):
            self._deg_extractor.deg_classifier.decoder.requires_grad_(True)

    def _get_pyramid_feature(self, lq_image, channels):
        key = str(channels)
        if key not in self.feature_pyramid:
            raise ValueError(f"Unsupported UNet hook channels: {channels}")
        module = self.feature_pyramid[key]
        first_weight = next(module.parameters())
        feat = lq_image.to(device=first_weight.device, dtype=first_weight.dtype)
        return module(feat)

    def _register_hook_block(self, name, conv):
        self.hook_blocks[name] = CrossAttnModulationBlock(
            feat_in_dim=conv.in_channels,
            feat_out_dim=conv.out_channels,
            deg_dim=self.inner_dim,
        )

    def _hook_conv(self, conv, hook_name, module, source_channels=None):
        orig = conv.forward

        def hooked(x):
            out = orig(x)
            if getattr(module, "_deg_feat", None) is None:
                return out
            fc = conv.in_channels if source_channels is None else source_channels
            feat = module._get_pyramid_feature(module._current_lq_image, fc)
            feat = F.interpolate(feat, size=x.shape[2:], mode="bilinear", align_corners=False)
            scale, shift = module.hook_blocks[hook_name](feat)
            scale = scale.to(device=out.device, dtype=out.dtype)
            shift = shift.to(device=out.device, dtype=out.dtype)
            return out * (1 + scale) + shift

        conv.forward = hooked

    def setup(self, unet):
        if self._hooked:
            return
        self._hooked = True

        self._register_hook_block("conv_in", unet.conv_in)
        self._hook_conv(unet.conv_in, "conv_in", self, source_channels=4)

        for down_idx, block in enumerate(unet.down_blocks):
            for res_idx, resnet in enumerate(block.resnets):
                conv = getattr(resnet, "conv2", None)
                if conv is not None:
                    name = f"down_{down_idx}_res_{res_idx}"
                    self._register_hook_block(name, conv)
                    self._hook_conv(conv, name, self)
            for ds_idx, ds in enumerate(getattr(block, "downsamplers", []) or []):
                conv = getattr(ds, "conv", None)
                if conv is not None:
                    name = f"down_{down_idx}_ds_{ds_idx}"
                    self._register_hook_block(name, conv)
                    self._hook_conv(conv, name, self)

        for res_idx, resnet in enumerate(getattr(unet.mid_block, "resnets", [])):
            conv = getattr(resnet, "conv2", None)
            if conv is not None:
                name = f"mid_res_{res_idx}"
                self._register_hook_block(name, conv)
                self._hook_conv(conv, name, self)

        for up_idx, block in enumerate(unet.up_blocks):
            for res_idx, resnet in enumerate(block.resnets):
                conv = getattr(resnet, "conv2", None)
                if conv is not None:
                    name = f"up_{up_idx}_res_{res_idx}"
                    self._register_hook_block(name, conv)
                    self._hook_conv(conv, name, self)
            for us_idx, us in enumerate(getattr(block, "upsamplers", []) or []):
                conv = getattr(us, "conv", None)
                if conv is not None:
                    name = f"up_{up_idx}_us_{us_idx}"
                    self._register_hook_block(name, conv)
                    self._hook_conv(conv, name, self)

    def get_modulation(self, lq_image, timestep=None):
        self.build_deg_extractor(lq_image.device)
        probs = self._deg_extractor.get_probs(lq_image)
        deg_feat = self._deg_extractor.embed_probs(probs, lq_image.device)
        self._current_lq_image = lq_image
        self._deg_feat = deg_feat
        for block in self.hook_blocks.values():
            block.set_deg(deg_feat)
        return None, None

    def forward(self, lq_image):
        self.get_modulation(lq_image)
        return None, None


# ═══════════════════════════════════════════════════════════════
#  Registry & builder
# ═══════════════════════════════════════════════════════════════

MODULE_REGISTRY = {
    "none": IdentityConditionModule,
    "codsr_lqfm": CODSRFiLMModule,
    "deg_cross_attn": DegCrossAttnModule,
}


def build_condition_module(module_type, embed_dim=256, device=None, unet=None, training=True, **kwargs):
    cls = MODULE_REGISTRY.get(module_type)
    if cls is None:
        raise ValueError(f"Unknown: {module_type}. Options: {list(MODULE_REGISTRY)}")
    module = cls(embed_dim=embed_dim, **kwargs)
    if device:
        module = module.to(device)
    if training:
        module.train()
    else:
        module.eval()
    if unet is not None:
        module.setup(unet)
    return module
