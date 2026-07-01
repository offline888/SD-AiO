import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import argparse

class Classifier1D(nn.Module):
    def __init__(self, feature_dim: int = 768, num_classes: int = 6):
        super().__init__()
        self.num_classes = num_classes
        self.norm = nn.LayerNorm(feature_dim)
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, num_classes * 2),
        )

    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        out = self.fc(x)
        return out.view(x.shape[0], self.num_classes, 2)


class DegNet_DINO(nn.Module):
    def __init__(
        self,
        feature_dim: int = 512,
        num_types: int = 4,
        dino_type: str = None,
        freeze_encoder: bool = False,
        patch_size: int = 14,
        encoder_layer_index: int = -1,
    ):
        super().__init__()

        if dino_type is not None:
            config = transformers.Dinov2Config.from_pretrained(dino_type)
            self.encoder = transformers.Dinov2Model.from_pretrained(dino_type)
        else:
            config = transformers.Dinov2Config()
            self.encoder = transformers.Dinov2Model(config)

        self.dino_dim = config.hidden_size
        self.num_types = num_types
        self.patch_size = patch_size
        self.encoder_layer_index = encoder_layer_index
        self.decoder = Classifier1D(feature_dim=self.dino_dim, num_classes=self.num_types)
        self._init_weights()

        if freeze_encoder:
            self._freeze_encoder()

    def _init_weights(self):
        def init_linear(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.decoder.apply(init_linear)

    def _freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward_cls(self, x: torch.Tensor):
        if self.encoder_layer_index != -1:
            outputs = self.encoder(pixel_values=x, output_hidden_states=True)
            hidden_states = outputs.hidden_states[self.encoder_layer_index]
        else:
            outputs = self.encoder(pixel_values=x)
            hidden_states = outputs.last_hidden_state
        cls_token = hidden_states[:, 0, :]
        logits = self.decoder(cls_token)
        return cls_token, logits

    def forward(self, x: torch.Tensor):
        return self.forward_cls(x)[1]


class DegFeatExtractor(nn.Module):
    def __init__(
        self,
        inner_dim: int,
        num_deg_types: int,
        weight_dtype: torch.dtype,
        args: argparse.Namespace,
        deg_embedding: nn.Parameter | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()

        if deg_embedding is not None:
            self.deg_embedding = deg_embedding
        else:
            self.deg_embedding = nn.Parameter(torch.randn(num_deg_types, inner_dim))
            nn.init.orthogonal_(self.deg_embedding)

        self.deg_alpha = nn.Parameter(torch.tensor(10.0))
        self.weight_dtype = weight_dtype

        self.deg_classifier = DegNet_DINO(
            dino_type=args.dino_type,
            num_types=num_deg_types,
        )
        ckpt_path = getattr(args, "degradation_classifier_path", None)
        if ckpt_path:
            state_dict = torch.load(ckpt_path, map_location="cpu")
            missing, unexpected = self.deg_classifier.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[DegFeatExtractor] Warning: {len(missing)} missing keys; first 3: {missing[:3]}")
            if unexpected:
                print(f"[DegFeatExtractor] Warning: {len(unexpected)} unexpected keys; first 3: {unexpected[:3]}")
        elif device is not None:
            print("[DegFeatExtractor] Warning: no classifier checkpoint — using random weights")

        self.deg_classifier.requires_grad_(False).eval()
        self.deg_classifier.to(device=device or torch.device("cpu"))

    def get_deg_feat(self, lq_images: torch.Tensor) -> torch.Tensor:
        enc_dtype = next(self.deg_classifier.encoder.parameters()).dtype
        lq = lq_images.to(dtype=enc_dtype)
        with torch.no_grad():
            cls_token, logits = self.deg_classifier.forward_cls(lq)
            cls_token = cls_token.to(dtype=self.weight_dtype)
            probs = torch.softmax(logits, dim=-1)[:, :, 0].to(dtype=self.weight_dtype)
        embedding = self.deg_embedding.to(device=lq_images.device, dtype=self.weight_dtype)
        return cls_token + self.deg_alpha * (probs @ embedding)

    def forward(self, lq_images: torch.Tensor) -> torch.Tensor:
        return self.get_deg_feat(lq_images)
