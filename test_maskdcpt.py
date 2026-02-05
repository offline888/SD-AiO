import torch
from  basicsr.archs.promptir_arch import PromptIR

model=PromptIR()
state_dicts=torch.load("/home/yhmi/data/model/MaskDCPT/nafnet_maskdcpt_5d.pth")

#print(state_dicts['params'].keys())
print(model)