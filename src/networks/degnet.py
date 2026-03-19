import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

class Classifier1D(nn.Module):
    '''
    Design for CLS token feature
    '''
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn

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
    '''
    Classifier With SwiGLU 
    '''
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
class Classifier2D(nn.Module):
    '''
    Simple CNN classifier for patch tokens
    '''
    def __init__(
        self,
        feature_dim: int = 768,
        num_classes: int = 6,
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # 1x1 Conv projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, 1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.GELU()
        )
        
        dim = feature_dim // 2
        
        # Simple CNN blocks instead of ResNet Bottleneck
        self.conv_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        in_ch = dim
        for i in range(3):
            layers = []
            # Two conv layers per block
            layers.append(nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(in_ch))  # Use BatchNorm instead of LayerNorm for flexibility
            layers.append(nn.GELU())
            layers.append(nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(in_ch))
            layers.append(nn.GELU())
            
            self.conv_layers.append(nn.Sequential(*layers))
            
            self.downsample_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, in_ch, 1, bias=False),
                    nn.MaxPool2d(2, 2),
                    nn.GELU(),
                )
            )
        
        # Calculate output channels after downsampling
        final_ch = dim
        
        self.fc = nn.Sequential(
            nn.Linear(final_ch, final_ch // 2),
            nn.GELU(),
            nn.Linear(final_ch // 2, num_classes * 2)
        )

    def forward(self, patch_tokens):
        B, N, C = patch_tokens.shape
        H = W = int(math.sqrt(N))
        feat = patch_tokens.transpose(-1, -2).contiguous().view(B, C, H, W)
        
        feat = self.input_proj(feat)  # [B, dim//2, 14, 14]
        
        for conv, downsample in zip(self.conv_layers, self.downsample_layers):
            feat = conv(feat)
            feat = downsample(feat)
        
        feat = feat.mean(dim=[-1, -2])  # [B, dim//2]
        out = self.fc(feat)  # [B, num_classes * 2]
        out = out.view(B, self.num_classes, 2)  # [B, num_classes, 2]
        
        return out


class DegNet_CLIP(nn.Module):

    def __init__(
        self,
        feature_dim: int = 512,
        num_types: int = 6,
        clip_type: str = None,
        freeze_encoder: bool = False,
        patch_size: int = 14,
        encoder_layer_index: int = -1,
    ):
        super(DegNet_CLIP, self).__init__()

        if clip_type is not None:
            config = transformers.CLIPVisionConfig.from_pretrained(clip_type)
            self.encoder = transformers.CLIPVisionModel(config)
        else:
            config = transformers.CLIPVisionConfig()
            self.encoder = transformers.CLIPVisionModel(config)

        self.model_dim = config.hidden_size
        self.num_types = num_types
        self.patch_size = patch_size
        self.encoder_layer_index = encoder_layer_index

        self.decoder = Classifier2D(
            feature_dim=self.model_dim,
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
        self.encoder = transformers.CLIPVisionModel.from_pretrained(model_path)
        self._freeze_encoder()

    def forward(self, x: torch.Tensor):
        if self.encoder_layer_index >= 0:
            outputs = self.encoder(pixel_values=x, output_hidden_states=True)
            hidden_states = outputs.hidden_states[self.encoder_layer_index]
        else:
            outputs = self.encoder(pixel_values=x)
            hidden_states = outputs.last_hidden_state
        
        patch_tokens = hidden_states[:, 1:, :]
        logits = self.decoder(patch_tokens)  # [B, num_classes, 2]
        
        # probs = torch.softmax(logits, dim=-1)

        return logits

class DegNet_DINO(nn.Module):

    def __init__(
        self,
        feature_dim: int = 512,
        num_types: int = 6,
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

