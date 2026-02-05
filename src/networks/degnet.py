import torch
import torch.nn as nn
import transformers


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
        num_types: int = 4,
        clip_type: str = "openai/clip-vit-base-patch32",
        freeze_encoder: bool = True,
    ):
        super(DegNet_CLIP, self).__init__()

        try:
            config = transformers.CLIPVisionConfig.from_pretrained(clip_type)
            self.encoder = transformers.CLIPVisionModel(config)
        except:
            config = transformers.CLIPVisionConfig()
            self.encoder = transformers.CLIPVisionModel(config)

        self.clip_dim = config.hidden_size
        self.num_types = num_types

        self.deg_dict = nn.Parameter(torch.randn(feature_dim, num_types))
        nn.init.orthogonal_(self.deg_dict) 
        
        self.adapter = nn.Sequential(
            nn.Linear(self.clip_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
        )

        self.deg_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, num_types),
            nn.ReLU(),
        )

        #self.classifier = nn.Linear(feature_dim, num_types)

        self._init_weights()

        if freeze_encoder:
            self._freeze_encoder()

    def _init_weights(self):
        for m in [self.adapter, self.deg_head, self.classifier]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def load_encoder(self, model_path):
        self.encoder = transformers.CLIPVisionModel.from_pretrained(model_path)
        self._freeze_encoder()

    def forward(self, x: torch.Tensor):
        # [CLS] token
        outputs = self.encoder(pixel_values=x)
        feat = outputs.last_hidden_state[:, 0, :]
        feat = self.adapter(feat)  

        # V:[batch size, num_types]
        severity = self.deg_head(feat)  
        deg_feat = severity @ self.deg_dict.T

        return deg_feat, severity

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
        num_types: int = 4,
        dino_type: str = "facebook/dinov2-base",
        freeze_encoder: bool = True,
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

        self.adapter = nn.Sequential(
            nn.Linear(self.dino_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
        )

        self.deg_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, num_types),
            nn.ReLU(), # 保持和你 CLIP 一致的归一化
        )

        self._init_weights()

        if freeze_encoder:
            self._freeze_encoder()

    def _init_weights(self):
        for m in [self.adapter, self.deg_head]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def load_encoder(self, model_path):
        self.encoder = transformers.Dinov2Model.from_pretrained(model_path)
        self._freeze_encoder()

    def forward(self, x: torch.Tensor):
        outputs = self.encoder(x)
        last_hidden_state = outputs.last_hidden_state

        # [bs, seq_len, dim] -> [bs, dim]
        # 对 Patch tokens 求平均
        patch_tokens = last_hidden_state[:, 1:, :]
        feat = patch_tokens.mean(dim=1)  

        feat = self.adapter(feat)
        severity = self.deg_head(feat)
        
        deg_feat = severity @ self.deg_dict.T

        return deg_feat, severity
class PromptIR_Encoder(nn.Module):
    """
    PromptIR 中的 Degradation Estimator 结构。
    通常由几个卷积层和 Global Average Pooling 组成。
    """
    def __init__(self, input_dim=3, output_dim=64):
        super(PromptIR_Encoder, self).__init__()
        
        # 典型的轻量级 5层 CNN 结构
        # 每一层下采样，提取纹理特征
        self.features = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # 最后一层保持 spatial size 或者直接 GAP
            nn.Conv2d(128, output_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.flatten(1) # [bs, output_dim]
        return x
class DegNet_MaskDCPT(nn.Module):
    """
    Input: Image
    Output: deg_feat, severity
    """
    def __init__(
        self,
        maskdcpt_path: str = None, # 预训练权重路径
        feature_dim: int = 512,    # DegNet 的字典维度
        num_types: int = 4,        # 退化类型数量
        promptir_dim: int = 64,    # PromptIR 编码器输出维度 (通常是 64)
        freeze_encoder: bool = True,
    ):
        super(DegNet_MaskDCPT, self).__init__()

        # 1. 初始化 PromptIR Backbone
        self.encoder = PromptIR_Encoder(output_dim=promptir_dim)
        self.promptir_dim = promptir_dim

        # 2. 加载 MaskDCPT 预训练权重
        if maskdcpt_path:
            self._load_weights(maskdcpt_path)
        else:
            print("Warning: No MaskDCPT path provided. Initializing randomly.")

        # 3. DegNet 组件 (Analysis-Synthesis)
        # 注意：这里的输入维度是 promptir_dim (如 64)，而不是 768
        self.feature_dim = feature_dim
        
        # 字典 (U)
        self.deg_dict = nn.Parameter(torch.randn(feature_dim, num_types))
        nn.init.orthogonal_(self.deg_dict)

        # 适配器 (Adapter): 64 -> 512
        # 因为 PromptIR 维度很低，这里其实是一个升维过程 (Upsampling features)
        self.adapter = nn.Sequential(
            nn.Linear(self.promptir_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
        )

        # 预测头 (Severity Head V): 512 -> 4
        self.deg_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, num_types),
            nn.ReLU(), # 保证非负
        )

        # 初始化自定义层
        self._init_weights()

        # 冻结 Encoder
        if freeze_encoder:
            self._freeze_encoder()

    def _load_weights(self, path):
        print(f"Loading MaskDCPT weights from {path}...")
        checkpoint = torch.load(path, map_location='cpu')
        
        # 处理可能的 key 不匹配问题
        # MaskDCPT 权重可能是 {'state_dict': ...} 或者直接是 model
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'params' in checkpoint:
            state_dict = checkpoint['params']
        else:
            state_dict = checkpoint

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # 移除 'module.' (多卡训练遗留)
            name = k.replace('module.', '')
            # 移除 'encoder_q.' (MoCo 训练遗留)
            name = name.replace('encoder_q.', '')
            
            # 极其重要：MaskDCPT 的权重可能叫 'E.0.weight' 或 'features.0.weight'
            # 你需要根据下载的权重文件 key 来调整这里的映射逻辑
            # 这里假设 MaskDCPT 保存的 key 能直接对应 PromptIR_Encoder 的 'features'
            if name.startswith('features'):
                new_state_dict[name] = v
            # 如果 MaskDCPT 用的名字不叫 features，比如叫 'net.'，你得在这里 replace
            
        # 加载权重 (strict=False 以防万一有多余的 head)
        try:
            self.encoder.load_state_dict(new_state_dict, strict=False)
            print("Encoder weights loaded successfully.")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Please check the keys in the .pth file and the PromptIR_Encoder definition.")

    def _init_weights(self):
        for m in [self.adapter, self.deg_head]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # 1. 提取特征 [bs, 64]
        feat = self.encoder(x)
        
        # 2. 升维适配 [bs, 64] -> [bs, 512]
        feat = self.adapter(feat)

        # 3. 预测系数 V
        severity = self.deg_head(feat)

        # 4. 字典重构
        deg_feat = severity @ self.deg_dict.T
        
        return deg_feat, severity