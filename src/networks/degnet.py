import torch
import torch.nn as nn
import transformers
from torchvision.models.resnet import ResNet, Bottleneck as BottleneckBlock

class PromptIR_DC(nn.Module):
    def __init__(
        self,
        latent_dim=1024,    # 输入向量的维度 D
        num_layers=3,       # 残差块的数量
        num_classes=3,      # 最终分类数
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

        self.act=nn.GELU()

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

class DegNet_CLIP(nn.Module):
    """
    Input: low quality image.\n
    Output:
        - deg_feat: Reconstructed degradation feature based on dictionary
        - severity: Predicted severity vector (V)
        - logits: Classification logits

    Concept:
        1. Prediction: V (severity) = NN(feat)
        2. Reconstruction: X_deg = V * U^T (Linear Combination of Basis)
    """

    def __init__(
        self,
        feature_dim: int = 512,
        num_types: int = 6,
        clip_type: str = "openai/clip-vit-base-patch32",
        freeze_encoder: bool = False,
        freeze_deg_dict: bool = False,
    ):
        super(DegNet_CLIP, self).__init__()

        #try:
        #    config = transformers.CLIPVisionConfig.from_pretrained(clip_type)
        #    self.encoder = transformers.CLIPVisionModel(config)
        #except:
        config = transformers.CLIPVisionConfig()
        self.encoder = transformers.CLIPVisionModel(config)

        self.clip_dim = config.hidden_size
        self.num_types = num_types

        self.deg_dict = nn.Parameter(torch.randn(feature_dim, num_types))
        # nn.init.xavier_uniform_(self.deg_dict)

        self.deg_head = nn.Sequential(
            nn.Linear(self.clip_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2,  feature_dim // 4),
            nn.GELU(),
        )

        #self.classifier = nn.Linear(feature_dim // 4, num_types)
        #self.classifier=models.resnet18(
        #                pretrained=False,
        #                num_classes=num_types,
        #                in_channels=feature_dim // 4)
        self.classifier = PromptIR_DC (
                        latent_dim = feature_dim // 4, 
                        num_layers = 3, 
                        num_classes = self.num_types,
                        )

        self._init_weights()

        if freeze_encoder:
            self._freeze_encoder()

        if freeze_deg_dict:
            self._freeze_deg_dict()

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

    def _freeze_deg_dict(self):
        self.deg_dict.requires_grad = False

    def load_encoder(self, model_path):
        self.encoder = transformers.CLIPVisionModel.from_pretrained(model_path)
        self._freeze_encoder()

    def forward(self, x: torch.Tensor):
        # [CLS] token
        outputs = self.encoder(pixel_values=x)
        feat = outputs.last_hidden_state[:, 0, :]

        # V:[batch size, num_types]
        severity = self.deg_head(feat)  
        severity = self.classifier(severity)

        deg_feat = severity @ self.deg_dict.T

        probs=torch.sigmoid(severity)

        return deg_feat, probs, severity

class DegNet_DINO(nn.Module):
    """
    Input: low quality image.
    Output:
        - deg_feat: Reconstructed degradation feature based on dictionary
        - severity: Predicted severity vector (V)

    Concept:
        1. Prediction: V (severity) = NN(feat)
        2. Reconstruction: X_deg = V * U^T (Linear Combination of Basis)
    """

    def __init__(
        self,
        feature_dim: int = 512,
        num_types: int = 6,
        dino_type: str = "facebook/dinov2-base",
        freeze_encoder: bool = False,
        freeze_deg_dict: bool = False,
    ): 
        super(DegNet_DINO, self).__init__()

        try:
            config = transformers.Dinov2Config.from_pretrained(dino_type)
            self.encoder = transformers.Dinov2Model(config)
        except:
            config = transformers.Dinov2Config()
            self.encoder = transformers.Dinov2Model(config)

        self.dino_dim = config.hidden_size
        self.num_types = num_types

        self.deg_dict = nn.Parameter(torch.randn(feature_dim, num_types))
        nn.init.orthogonal_(self.deg_dict)
        # nn.init.xavier_uniform_(self.deg_dict)

        self.deg_head = nn.Sequential(
            nn.Linear(self.dino_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2,  feature_dim // 4),
            nn.GELU(),
        )

        #self.classifier = nn.Linear(feature_dim // 4, num_types)
        #self.classifier = models.resnet18(
        #                pretrained=False,
        #                num_classes=num_types,
        #                in_channels=feature_dim // 4)
        self.classifier = PromptIR_DC (
                        latent_dim = feature_dim // 4, 
                        num_layers = 3, 
                        num_classes = self.num_types,
                        )

        self._init_weights()

        if freeze_encoder:
            self._freeze_encoder()

        if freeze_deg_dict:
            self._freeze_deg_dict()


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

    def _freeze_deg_dict(self):
        self.deg_dict.requires_grad = False

    def load_encoder(self, model_path):
        self.encoder = transformers.Dinov2Model.from_pretrained(model_path)
        self._freeze_encoder()

    def forward(self, x: torch.Tensor):
        # [CLS] token
        outputs = self.encoder(pixel_values=x)
        feat = outputs.last_hidden_state[:, 0, :]

        # V:[batch size, num_types]
        severity = self.deg_head(feat)  
        severity = self.classifier(severity)

        deg_feat = severity @ self.deg_dict.T

        probs = torch.sigmoid(severity)

        return deg_feat, probs, severity

