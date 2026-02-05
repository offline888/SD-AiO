import torch
import torch.nn as nn
import transformers
from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class DegNet_CLIP(nn.Module):
    '''
    Input: low quality image.\n
    Output: Degradation severity vector-V.\n
    
    Concept: X_deg = U(shared) * V(data-dependent)
    '''
    def __init__(self,
                 feature_dim:int = 512, 
                 num_types:int = 4,     
                 clip_type:str="clip-vit-base-patch32"):
        super(DegNet_CLIP, self).__init__()
        config=transformers.CLIPVisionConfig()
        self.encoder=transformers.CLIPVisionModel(config)
        self.clip_dim=config.hidden_size
        self.num_types=num_types # num_ranks of A

        self.A=nn.Parameter(torch.randn(feature_dim, num_types),requires_grad=True)
        nn.init.orthogonal_(self.A) 

        self.adapter=nn.Sequential(
            nn.Linear(self.clip_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU()
        )
        self.type_head=nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, num_types), 
        )
        self.severity_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, feature_dim), # Output matches D dimension
            # nn.Tanh() 
        )
        self.classifier=nn.Linear(feature_dim, num_types) 
    def set_train(self,enable_encoder_train=False):
        for param in self.encoder.parameters():
            param.requires_grad=enable_encoder_train
        self.adapter.requires_grad_(True)
        self.type_head.requires_grad_(True)
        self.severity_head.requires_grad_(True)
        self.classifier.requires_grad_(True)
        self.A.requires_grad_(True)
    def load_encoder(self,model_path):
        self.encoder=transformers.CLIPVisionModel.from_pretrained(model_path)
        self.encoder.to('cuda')
    def forward(self, x:torch.Tensor):

        feat=self.encoder(pixel_values=x).pooler_output
        feat=self.adapter(feat)
        
        weight=self.type_head(feat)   # [bs, r]
        severity=self.severity_head(feat) # [bs, d]

        deg_feat=weight @ self.A.T + severity  # [bs,d]

        logits=self.classifier(deg_feat) 

        return deg_feat,weight,severity,logits

@ARCH_REGISTRY.register()
class DegNet_DINO(nn.Module):
    '''
    Input: low quality image.\n
    Output: 
        - D: Composite degradation feature [bs, feature_dim]
        - type_logits: For auxiliary classification loss
        - severity: (Optional) for regression loss
    
    Concept: D = A(shared) * b(data-dependent) + c(data-dependent)
    '''
    def __init__(self,
                 feature_dim:int = 512, 
                 num_types:int = 4,     
                 dino_type:str="dinov2-base"):
        super(DegNet_DINO, self).__init__()
        configuration=transformers.Dinov2Config()
        self.encoder=transformers.Dinov2Model(configuration)
        self.dino_dim=configuration.hidden_size
        self.num_types=num_types # num_ranks of A

        self.A=nn.Parameter(torch.randn(feature_dim, num_types),requires_grad=True)
        nn.init.orthogonal_(self.A) 

        self.adapter=nn.Sequential(
            nn.Linear(self.dino_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU()
        )
        self.type_head=nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, num_types), 
        )
        self.severity_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, feature_dim), # Output matches D dimension
            # nn.Tanh() 
        )
        self.classifier=nn.Linear(feature_dim, num_types) 
    def set_train(self,enable_encoder_train=False):
        for param in self.encoder.parameters():
            param.requires_grad=enable_encoder_train
        self.adapter.requires_grad_(True)
        self.type_head.requires_grad_(True)
        self.severity_head.requires_grad_(True)
        self.classifier.requires_grad_(True)
        self.A.requires_grad_(True)
    def load_encoder(self,model_path):
        self.encoder=transformers.Dinov2Model.from_pretrained(model_path)
        self.encoder.to('cuda')
    def forward(self, x:torch.Tensor):
        outputs=self.encoder(x)
        last_hidden_state=outputs.last_hidden_state
        # use oatch tokens rather than cls tokens
        patch_tokens = last_hidden_state[:, 1:, :]
        feat=patch_tokens.mean(dim=1) # [bs,dino_dim]
        feat=self.adapter(feat)
        
        weight=self.type_head(feat)   # [bs, r]
        severity=self.severity_head(feat) # [bs, d]

        deg_feat=weight @ self.A.T + severity  # [bs,d]

        logits=self.classifier(deg_feat) 

        return deg_feat,weight,severity,logits