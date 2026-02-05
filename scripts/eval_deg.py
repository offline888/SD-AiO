import sys
import torch
from torchinfo import summary

sys.path.append("/home/yhmi/All_in_one")
from basicsr.archs.degnet_arch import DegNet_CLIP 

model = DegNet_CLIP(feature_dim=512, num_types=4)
checkpoint = torch.load("/home/yhmi/All_in_one/experiments/DegNetCLIP_CDD11_seed0_AdamW_lr1e3_epoch10/models/net_dc_100000.pth")
if 'params' in checkpoint:
    state_dict = checkpoint['params']
else:
    state_dict = checkpoint

model.load_state_dict(state_dict, strict=True)

print(model)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
print(f"总参数量: {total_params:,} (约{total_params/1e6:.2f}M)")
print(f"可训练参数量: {trainable_params:,}")