import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import argparse

logger = logging.getLogger(__name__)

class Classifier1D(nn.Module):
    def __init__(
        self,
        feature_dim: int = 768,
        num_classes: int = 6,
    ):
        super().__init__()
        self.num_classes = num_classes
        
        self.norm = nn.LayerNorm(feature_dim)
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, num_classes * 2)
        )

    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        out = self.fc(x)
        B, _ = x.shape
        out = out.view(B, self.num_classes, 2)  # [B, num_classes, 2]
        return out

class ResidualClassifier1D(nn.Module):
    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 384,   
        num_classes: int = 6,
        dropout_prob: float = 0.2
    ):
        super().__init__()
        self.num_classes = num_classes
        
        self.norm = nn.LayerNorm(feature_dim)
        
        self.residual_block = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, feature_dim), # 升维回到 768
            nn.Dropout(dropout_prob)
        )
        
        self.classifier = nn.Linear(feature_dim, num_classes * 2)

    def forward(self, x: torch.Tensor):
        # x shape: [B, 768]
        x_norm = self.norm(x)
        
        h = x_norm + self.residual_block(x_norm) 
        
        out = self.classifier(h)
        B = x.shape[0]
        out = out.view(B, self.num_classes, 2)  # [B, num_class, 2]
        return out
    
class GLUClassifier1D(nn.Module):
    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 384,  
        num_classes: int = 6,
        dropout_prob: float = 0.2
    ):
        super().__init__()
        self.num_classes = num_classes
        
        self.norm = nn.LayerNorm(feature_dim)
        
        self.gate_proj = nn.Linear(feature_dim, hidden_dim)
        self.up_proj = nn.Linear(feature_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_dim, num_classes * 2)

    def forward(self, x: torch.Tensor):
        # x shape: [B, 768]
        x = self.norm(x)
        
        hidden = F.silu(self.gate_proj(x)) * self.up_proj(x)
        
        hidden = self.dropout(hidden)
        out = self.classifier(hidden)
        
        B = x.shape[0]
        out = out.view(B, self.num_classes, 2) 
        return out

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
        super(DegNet_DINO, self).__init__()

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

        #self.decoder=GLUClassifier1D(
        #    feature_dim=self.dino_dim,
        #    hidden_dim=self.dino_dim // 2,
        #    num_classes=self.num_types,
        #    dropout_prob=0.1
        #)
        self.decoder = Classifier1D(
            feature_dim=self.dino_dim,
            num_classes=self.num_types,
        )

        self._init_weights()

        if freeze_encoder:
            self._freeze_encoder()

    def _init_weights(self):
    
        def init_linear(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        self.decoder.apply(init_linear)

    def _freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def load_encoder(self, model_path):
        self.encoder = transformers.Dinov2Model.from_pretrained(model_path)
        self._freeze_encoder()

    def forward(self, x: torch.Tensor):
        if self.encoder_layer_index != -1:
            outputs = self.encoder(pixel_values=x, output_hidden_states=True)
            hidden_states = outputs.hidden_states[self.encoder_layer_index]
        else:
            outputs = self.encoder(pixel_values=x)
            hidden_states = outputs.last_hidden_state
        
        cls_token = hidden_states[:, 0, :]
        
        logits = self.decoder(cls_token)  # [B, num_classes, 2]

        return logits

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
        self._log_counter = 0
        if deg_embedding is not None:
            self.deg_embedding = deg_embedding
        else:
            self.deg_embedding = nn.Parameter(torch.randn(num_deg_types, inner_dim))
            nn.init.orthogonal_(self.deg_embedding)

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
                print(f"[DegFeatExtractor] Warning: {len(missing)} missing keys "
                      f"when loading classifier checkpoint.\n  First 3: {missing[:3]}")
            if unexpected:
                print(f"[DegFeatExtractor] Warning: {len(unexpected)} unexpected keys "
                      f"in classifier checkpoint.\n  First 3: {unexpected[:3]}")
        elif device is not None:
            print("[DegFeatExtractor] Warning: no classifier checkpoint provided — "
                  "using random weights (classifier will produce garbage)")
        self.deg_classifier.requires_grad_(False).eval()
        self.deg_classifier.to(device=device or torch.device("cpu"))

    def forward(self, lq_images: torch.Tensor) -> torch.Tensor:
        logits = self.deg_classifier(lq_images)
        deg_probs = torch.softmax(logits, dim=-1)[:, :, 0].to(dtype=self.weight_dtype)
        embedding = self.deg_embedding.to(device=lq_images.device, dtype=self.weight_dtype)
        return deg_probs @ embedding

    def get_probs(self, lq_images: torch.Tensor) -> torch.Tensor:
        """Classifier-only forward (no_grad safe). Returns probs V ∈ R^{B×C}."""
        with torch.no_grad():
            logits = self.deg_classifier(lq_images)
            return torch.softmax(logits, dim=-1)[:, :, 0].to(dtype=self.weight_dtype)

    def embed_probs(self, probs: torch.Tensor, device: torch.device) -> torch.Tensor:
        """probs @ embedding — trainable embedding gets gradient."""
        embedding = self.deg_embedding.to(device=device, dtype=self.weight_dtype)
        return probs.to(dtype=self.weight_dtype) @ embedding
