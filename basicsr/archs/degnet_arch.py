import os
import torch
import torch.nn as nn
import transformers
from basicsr.utils.registry import ARCH_REGISTRY


class PromptIR_DC(nn.Module):
    def __init__(
        self,
        latent_dim=1024,
        num_layers=3,
        num_classes=3,
    ):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim//2),
            nn.LayerNorm(latent_dim//2),
            nn.GELU()
        )

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim//2, latent_dim//2),
                nn.LayerNorm(latent_dim//2),
                nn.GELU(),
                nn.Linear(latent_dim//2, latent_dim//2)
            ) for _ in range(num_layers)
        ])

        self.act = nn.GELU()

        self.classifier = nn.Sequential (
            nn.Linear(latent_dim//2, latent_dim//4),
            nn.GELU(),
            nn.Linear(latent_dim//4, num_classes),
        )

    def forward(self, latent_vec):
        x = self.input_proj(latent_vec)
        
        for layer in self.layers:
            x = x + layer(x)
            x = self.act(x)
            
        severity = self.classifier(x)

        return severity


@ARCH_REGISTRY.register()
class DegNet_CLIP(nn.Module):
    """
    Input: low quality image.
    Output:
        - deg_feat: Reconstructed degradation feature based on dictionary
        - probs: Sigmoid probabilities
        - severity: Predicted severity vector (V)

    Concept:
        1. Prediction: V (severity) = NN(feat)
        2. Reconstruction: X_deg = V * U^T (Linear Combination of Basis)
    """
    def __init__(
        self,
        feature_dim: int = 512,
        num_types: int = 4,
        encoder_type: str = "/home/yhmi/data/model/clip-vit-base-patch32",
        freeze_encoder: bool = False,
    ):
        super(DegNet_CLIP, self).__init__()

        # Load CLIP encoder
        try:
            if os.path.exists(encoder_type):
                self.encoder = transformers.CLIPVisionModel.from_pretrained(encoder_type)
                config = self.encoder.config
            else:
                config = transformers.CLIPVisionConfig.from_pretrained(encoder_type)
                self.encoder = transformers.CLIPVisionModel(config)
        except Exception as e:
            print(f"Warning: Failed to load CLIP model from {encoder_type}, using default config. Error: {e}")
            config = transformers.CLIPVisionConfig()
            self.encoder = transformers.CLIPVisionModel(config)

        self.encoder_dim = config.hidden_size
        self.num_types = num_types

        # Dictionary for degradation reconstruction
        self.deg_dict = nn.Parameter(torch.randn(feature_dim, num_types))
        nn.init.orthogonal_(self.deg_dict)

        # Deg head: encoder_dim -> feature_dim -> feature_dim//2 -> feature_dim//4 -> GELU
        self.deg_head = nn.Sequential(
            nn.Linear(self.encoder_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.GELU(),
        )

        # Classifier
        self.classifier = PromptIR_DC(
            latent_dim=feature_dim // 4,
            num_layers=3,
            num_classes=self.num_types,
        )

        self._init_weights()

        if freeze_encoder:
            self._freeze_encoder()

    def _init_weights(self):
        nn.init.orthogonal_(self.deg_dict)
        
        def init_linear(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        self.deg_head.apply(init_linear)
        self.classifier.apply(init_linear)

    def _freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def load_encoder(self, model_path):
        self.encoder = transformers.CLIPVisionModel.from_pretrained(model_path)
        self._freeze_encoder()

    def set_train(self, enable_encoder_train=False):
        """Set model to train mode, optionally freeze encoder."""
        if not enable_encoder_train:
            self._freeze_encoder()
        else:
            for param in self.encoder.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor):
        # Extract feature using CLIP encoder [CLS] token
        outputs = self.encoder(pixel_values=x)
        feat = outputs.last_hidden_state[:, 0, :]

        # Predict severity
        severity = self.deg_head(feat)
        severity = self.classifier(severity)

        # Reconstruct degradation feature
        deg_feat = severity @ self.deg_dict.T

        # Sigmoid probabilities for multi-label classification
        probs = torch.sigmoid(severity)

        return deg_feat, probs, severity


@ARCH_REGISTRY.register()
class DegNet_DINO(nn.Module):
    """
    Input: low quality image.
    Output:
        - deg_feat: Reconstructed degradation feature based on dictionary
        - probs: Sigmoid probabilities
        - severity: Predicted severity vector (V)

    Concept:
        1. Prediction: V (severity) = NN(feat)
        2. Reconstruction: X_deg = V * U^T (Linear Combination of Basis)
    """
    def __init__(
        self,
        feature_dim: int = 512,
        num_types: int = 4,
        encoder_type: str = "/home/yhmi/data/model/dinov2-base",
        freeze_encoder: bool = False,
    ):
        super(DegNet_DINO, self).__init__()

        # Load DINO encoder
        try:
            if os.path.exists(encoder_type):
                config = transformers.Dinov2Config.from_pretrained(encoder_type)
                self.encoder = transformers.Dinov2Model.from_pretrained(encoder_type)
            else:
                config = transformers.Dinov2Config.from_pretrained(encoder_type)
                self.encoder = transformers.Dinov2Model(config)
        except Exception as e:
            print(f"Warning: Failed to load DINO model from {encoder_type}, using default config. Error: {e}")
            config = transformers.Dinov2Config()
            self.encoder = transformers.Dinov2Model(config)

        self.encoder_dim = config.hidden_size
        self.num_types = num_types

        # Dictionary for degradation reconstruction
        self.deg_dict = nn.Parameter(torch.randn(feature_dim, num_types))
        nn.init.orthogonal_(self.deg_dict)

        # Deg head: encoder_dim -> feature_dim -> feature_dim//2 -> feature_dim//4 -> GELU
        self.deg_head = nn.Sequential(
            nn.Linear(self.encoder_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.GELU(),
        )

        # Classifier
        self.classifier = PromptIR_DC(
            latent_dim=feature_dim // 4,
            num_layers=3,
            num_classes=self.num_types,
        )

        self._init_weights()

        if freeze_encoder:
            self._freeze_encoder()

    def _init_weights(self):
        nn.init.orthogonal_(self.deg_dict)
        
        def init_linear(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        self.deg_head.apply(init_linear)
        self.classifier.apply(init_linear)

    def _freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def load_encoder(self, model_path):
        self.encoder = transformers.Dinov2Model.from_pretrained(model_path)
        self._freeze_encoder()

    def set_train(self, enable_encoder_train=False):
        """Set model to train mode, optionally freeze encoder."""
        if not enable_encoder_train:
            self._freeze_encoder()
        else:
            for param in self.encoder.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor):
        # Extract feature using DINO encoder [CLS] token
        outputs = self.encoder(pixel_values=x)
        feat = outputs.last_hidden_state[:, 0, :]

        # Predict severity
        severity = self.deg_head(feat)
        severity = self.classifier(severity)

        # Reconstruct degradation feature
        deg_feat = severity @ self.deg_dict.T

        # Sigmoid probabilities for multi-label classification
        probs = torch.sigmoid(severity)

        return deg_feat, probs, severity
