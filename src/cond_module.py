import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal
from diffusers.models.resnet import ResnetBlock2D
import torchvision.models as models
from degnet import DegFeatExtractor


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class BaseConditionModule(nn.Module):
    def setup(self, network: nn.Module):
        pass
    def get_modulation(self, lq_image, timestep=None):
        return None, None
    def forward(self, lq_image):
        return self.get_modulation(lq_image)

class IdentityConditionModule(BaseConditionModule):
    def __init__(self, **kwargs):
        super().__init__()


# Adapted from StableSR (https://github.com/IceClear/StableSR)
class SPADE(nn.Module):
    def __init__(self, 
                 output_channels: int,
                 cond_input_channels: int,
                 kernel_size: int = 3 , 
                 norm_type: Optional[Literal["instance", "batch", "layer", "group"]] = "group"):
        
        super().__init__() 

        if norm_type == "group":
            self.norm = nn.GroupNorm(num_groups=32, num_channels=output_channels)
        elif norm_type == "batch":
            self.norm = nn.BatchNorm2d(output_channels)
        elif norm_type == "layer":
            # layer norm is a special case of group norm with 1 group
            self.norm = nn.GroupNorm(num_groups=1, num_channels=output_channels) 
        elif norm_type == "instance":
            self.norm = nn.InstanceNorm2d(output_channels)
        else:
            raise ValueError(f"Invalid norm type: {norm_type}")

        pad = kernel_size // 2
        self.mlp_shared = nn.Sequential(nn.Conv2d(cond_input_channels, 128, kernel_size, padding=pad), nn.ReLU())
        self.mlp_gamma = nn.Conv2d(128, output_channels, kernel_size, padding=pad)
        self.mlp_beta = nn.Conv2d(128, output_channels, kernel_size, padding=pad)
    def forward(self, x: torch.Tensor, cond_feat: torch.Tensor):
        # x shape: (B, C, H, W)
        assert cond_feat.shape[2:] == x.shape[2:], "Condition feature must have the same spatial resolution as the input"
        
        normalized_x = self.norm(x)

        h = self.mlp_shared(cond_feat)
        gamma = self.mlp_gamma(h)
        beta = self.mlp_beta(h)

        out = normalized_x * (1 + gamma) + beta
        return out

class SPADEWrapper(nn.Module):
    def __init__(self, target_module: nn.Conv2d, condition_channels: int):
        super().__init__()
        self.target_module = target_module

        self.spade = SPADE(
            output_channels=target_module.out_channels,
            cond_input_channels=condition_channels,
            kernel_size=3,
            norm_type="group"
        )

        self.current_cond_feat = None

    @property
    def weight(self): return self.target_module.weight
    @property
    def bias(self): return self.target_module.bias
    @property
    def kernel_size(self): return self.target_module.kernel_size
    @property
    def stride(self): return self.target_module.stride
    @property
    def padding(self): return self.target_module.padding
    @property
    def dilation(self): return self.target_module.dilation
    @property
    def groups(self): return self.target_module.groups
    @property
    def out_channels(self): return self.target_module.out_channels
    @property
    def in_channels(self): return self.target_module.in_channels

    def forward(self, x):
        hidden_states = self.target_module(x)

        if self.current_cond_feat is None:
            raise ValueError("Condition feature is None!!")

        cond_feat = self.current_cond_feat.to(device=hidden_states.device, dtype=hidden_states.dtype)
        assert cond_feat.shape[2:] == hidden_states.shape[2:], "Condition feature must have the same spatial resolution as the input"

        hidden_states = self.spade(hidden_states, cond_feat)
        self.current_cond_feat = None

        return hidden_states

    @classmethod
    def inject(cls, unet, condition_channels):
        wrappers = []
        for _, module in unet.named_modules():
            if isinstance(module, ResnetBlock2D):
                wrapper = cls(module.conv2, condition_channels)
                module.conv2 = wrapper
                wrappers.append(wrapper)
        return wrappers

class MultiScaleExtractor(nn.Module):
    C320, C640, C1280 = 320, 640, 1280

    def __init__(self, backbone_type: Literal["simple-conv", "resnet18", "convnext_tiny"] = "resnet18"):
        super().__init__()
        self.backbone_type = backbone_type.lower()
        
        self.ch_320 = 320
        self.ch_640 = 640
        self.ch_1280 = 1280
        
        if self.backbone_type == "simple-conv":

            self.conv_in = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(128, self.ch_320, kernel_size=4, stride=2, padding=1), nn.SiLU()
            )
            self.down_C640 = nn.Sequential(nn.Conv2d(self.ch_320, self.ch_640, 4, stride=2, padding=1), nn.SiLU())
            self.down_C1280_D = nn.Sequential(nn.Conv2d(self.ch_640, self.ch_1280, 4, stride=2, padding=1), nn.SiLU())
            self.down_C1280_M = nn.Sequential(nn.Conv2d(self.ch_1280, self.ch_1280, 4, stride=2, padding=1), nn.SiLU())

        elif self.backbone_type == "resnet18":
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
            self.layer_320_raw, self.layer_640_raw, self.layer_1280_raw = resnet.layer2, resnet.layer3, resnet.layer4
            self.proj_320 = nn.Conv2d(128, self.C320, 1)
            self.proj_640 = nn.Conv2d(256, self.C640, 1)
            self.proj_1280 = nn.Conv2d(512, self.C1280, 1)
            self.down_C1280_M = nn.Sequential(nn.Conv2d(self.C1280, self.C1280, 3, 2, 1), nn.SiLU())
        elif self.backbone_type == "convnext_tiny":
            features = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1).features
            self.stem = nn.Sequential(features[0], features[1])
            self.layer_320_raw = nn.Sequential(features[2], features[3])
            self.layer_640_raw = nn.Sequential(features[4], features[5])
            self.layer_1280_raw = nn.Sequential(features[6], features[7])
            self.proj_320 = nn.Conv2d(192, self.C320, 1)
            self.proj_640 = nn.Conv2d(384, self.C640, 1)
            self.proj_1280 = nn.Conv2d(768, self.C1280, 1)
            self.down_C1280_M = nn.Sequential(nn.Conv2d(self.C1280, self.C1280, 3, 2, 1), nn.GELU())
        else:
            raise ValueError(f"Unsupported backbone: {backbone_type}")

    def forward(self, x):
        if self.backbone_type == "simple-conv":
            f_320 = self.conv_in(x)
            f_640 = self.down_C640(f_320)
            f_1280_D = self.down_C1280_D(f_640)
            f_1280_M = self.down_C1280_M(f_1280_D)
        else:
            h = self.stem(x)
            h_320 = self.layer_320_raw(h)
            h_640 = self.layer_640_raw(h_320)
            h_1280 = self.layer_1280_raw(h_640)
            f_320 = self.proj_320(h_320)
            f_640 = self.proj_640(h_640)
            f_1280_D = self.proj_1280(h_1280)
            f_1280_M = self.down_C1280_M(f_1280_D)
        return {"C320": f_320, "C640": f_640, "C1280_Down": f_1280_D, "C1280_Mid": f_1280_M}


class SimpleModule(BaseConditionModule):
    DOWN_KEYS = {0: "C320", 1: "C640", 2: "C1280_Down", 3: "C1280_Mid"}
    UP_KEYS = {0: "C1280_Mid", 1: "C1280_Down", 2: "C640", 3: "C320"}

    def __init__(self, backbone_type="resnet18", **kwargs):
        super().__init__()
        self.extractor = MultiScaleExtractor(backbone_type=backbone_type)
        self._scale_groups = {k: [] for k in self.DOWN_KEYS.values()}
        self.spade_wrappers = nn.ModuleList()
        self._hooked = False

    def clear_cache(self):
        for wrappers in self._scale_groups.values():
            for w in wrappers:
                w.current_cond_feat = None

    def setup(self, unet):
        if self._hooked:
            return
        self._hooked = True

        def hook_conv2(resnet, channel_key):
            conv = getattr(resnet, "conv2", None)
            if conv is None:
                return
            out_ch = getattr(conv, "out_channels", None) or conv.base_layer.out_channels
            wrapper = SPADEWrapper(conv, out_ch)
            self._scale_groups[channel_key].append(wrapper)
            self.spade_wrappers.append(wrapper)
            resnet.conv2 = wrapper

        for i, block in enumerate(unet.down_blocks):
            for resnet in block.resnets:
                hook_conv2(resnet, self.DOWN_KEYS[i])
        for resnet in getattr(unet.mid_block, "resnets", []):
            hook_conv2(resnet, "C1280_Mid")
        for i, block in enumerate(unet.up_blocks):
            for resnet in block.resnets:
                hook_conv2(resnet, self.UP_KEYS[i])

        print(f"[SimpleModule] backbone={self.extractor.backbone_type.upper()} "
              f"hooks={sum(len(w) for w in self._scale_groups.values())}")

    def get_modulation(self, lq_image, timestep=None):
        device, dtype = next(self.parameters()).device, next(self.parameters()).dtype
        feats = self.extractor(lq_image.to(device=device, dtype=dtype))
        for key, wrappers in self._scale_groups.items():
            for w in wrappers:
                w.current_cond_feat = feats[key]
        return None, None

    def forward(self, lq_image):
        return self.get_modulation(lq_image)


class DegTextFusion(nn.Module):
    """F_Deg → text token → prepend to CLIP text embedding → UNet cross-attention."""
    def __init__(self, inner_dim=768, text_dim=1024):
        super().__init__()
        self.deg_token_proj = nn.Sequential(
            nn.Linear(inner_dim, text_dim), nn.GELU(),
            nn.Linear(text_dim, text_dim))

    def forward(self, F_deg, text_embedding):
        if text_embedding is None:
            return None
        deg_token = self.deg_token_proj(F_deg).unsqueeze(1)
        return torch.cat([deg_token, text_embedding], dim=1)


class DegAwareModule(BaseConditionModule):
    def __init__(self, backbone_type="resnet18", embed_dim=768, inner_dim=768, args=None, **kwargs):
        super().__init__()
        self.inner_dim = inner_dim
        self._args = args
        self.spatial_extractor = SimpleModule(backbone_type=backbone_type)
        self.text_fusion = DegTextFusion(inner_dim=inner_dim, text_dim=1024)
        self.deg_extractor = None

    def build_deg_extractor(self, device):
        if self.deg_extractor is not None:
            return
        if self._args is None:
            raise ValueError("DegAwareModule requires args for DegFeatExtractor")
        self.deg_extractor = DegFeatExtractor(
            inner_dim=self.inner_dim, num_deg_types=self._args.num_deg_types,
            weight_dtype=torch.float32, args=self._args, deg_embedding=None, device=device)
        self.deg_extractor.deg_classifier._freeze_encoder()
        if getattr(self._args, "freeze_decoder", True):
            self.deg_extractor.deg_classifier.decoder.requires_grad_(False)

    def setup(self, unet):
        self.spatial_extractor.setup(unet)

    def get_modulation(self, lq_image, text_embedding=None, timestep=None, f_deg=None):
        device, dtype = next(self.parameters()).device, next(self.parameters()).dtype

        # Spatial: backbone → SPADE (no degradation modulation)
        self.spatial_extractor.get_modulation(lq_image)

        # Text: F_Deg → deg_token → prepend to text_embedding
        if f_deg is None:
            self.build_deg_extractor(lq_image.device)
            f_deg = self.deg_extractor.get_deg_feat(lq_image).to(device=device, dtype=dtype)
        else:
            f_deg = f_deg.to(device=device, dtype=dtype)
        text_embedding = self.text_fusion(f_deg, text_embedding)

        return None, text_embedding

    def forward(self, lq_image, text_embedding=None):
        return self.get_modulation(lq_image, text_embedding)

MODULE_REGISTRY = {
    "none": IdentityConditionModule,
    "simple": SimpleModule,
    "deg-aware": DegAwareModule,
    # backward-compat alias
    "deg_aware_sft": DegAwareModule,
}

def build_condition_module(module_type, embed_dim=256, device=None, unet=None,
                           training=False, backbone_type="resnet18", **kwargs):
    """Default training=False; caller should call .train() explicitly during training."""
    cls = MODULE_REGISTRY.get(module_type)
    if cls is None:
        raise ValueError(f"Unknown: {module_type}. Options: {list(MODULE_REGISTRY)}")
    module = cls(embed_dim=embed_dim, backbone_type=backbone_type, **kwargs)
    if device:
        module = module.to(device)
    module.train() if training else module.eval()
    if unet is not None:
        module.setup(unet)
    return module