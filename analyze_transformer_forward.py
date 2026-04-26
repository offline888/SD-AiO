#!/usr/bin/env python3
import torch
import sys
sys.path.insert(0, '/home/yhmi/All_in_one')

from src.flux2.transformer_flux2 import Flux2Transformer2DModel

device = 'cuda:0'

# 加载 transformer
print("Loading transformer...")
transformer = Flux2Transformer2DModel.from_pretrained(
    '/home/yhmi/data/model/flux.2-klein', subfolder='transformer', torch_dtype=torch.bfloat16
)
transformer.to(device)
transformer.eval()

print("\n" + "="*80)
print("=== Transformer 结构分析 ===")
print("="*80)
print(transformer)

# 检查是否有哪些层可能会让 hidden_states 直接传递
# 例如: LayerNorm 可能会学到恒等映射

# 分析 Transformer 的各个组件
print("\n" + "="*80)
print("=== Transformer 子模块 (Transformer/Single Blocks) ===")
print("="*80)
for name, module in transformer.named_modules():
    if 'transformer_blocks' in name or 'single_transformer_blocks' in name:
        print(f"  {name}: {type(module).__name__}")

# 创建一个简单的测试: 追踪 hidden_states 的变化
print("\n" + "="*80)
print("=== 测试 hidden_states 变化 ===")
print("="*80)

B, L, C = 1, 1024, 128
hidden_states = torch.randn(B, L, C, device=device, dtype=torch.bfloat16)
timestep = torch.tensor([0.9], device=device, dtype=torch.bfloat16)
guidance = torch.tensor([3.5], device=device, dtype=torch.bfloat16)
encoder_hidden_states = torch.randn(B, 120, 4096, device=device, dtype=torch.bfloat16)
txt_ids = torch.randn(B, 121, 3, device=device, dtype=torch.bfloat16)
img_ids = torch.randn(B, 1024, 3, device=device, dtype=torch.bfloat16)
deg_emb = torch.randn(B, 1, 768, device=device, dtype=torch.bfloat16)
lq_tensor = torch.randn(B, 3, 512, 512, device=device, dtype=torch.bfloat16)

# Hook 注册：追踪中间输出
intermediate_outputs = {}

def hook_fn(name):
    def _hook(module, input, output):
        if isinstance(output, tuple):
            intermediate_outputs[name] = output[0].detach()
        else:
            intermediate_outputs[name] = output.detach()
    return _hook

hooks = []
for name, module in transformer.named_modules():
    if 'norm' in name.lower() and 'single' in name:
        h = module.register_forward_hook(hook_fn(name))
        hooks.append(h)

with torch.no_grad():
    out = transformer(
        hidden_states=hidden_states,
        timestep=timestep,
        guidance=guidance,
        encoder_hidden_states=encoder_hidden_states,
        txt_ids=txt_ids,
        img_ids=img_ids,
        deg_emb=deg_emb,
        return_dict=False,
        lq_tensor=lq_tensor,
    )[0]

print(f"输入 hidden_states mean: {hidden_states.mean().item():.6f}")
print(f"输出 hidden_states mean: {out.mean().item():.6f}")
print(f"输出 vs 输入: diff={(out - hidden_states).abs().mean().item():.6f}")

# 清理 hooks
for h in hooks:
    h.remove()

# 关键: 分析 forward 中 hidden_states 的完整流程
print("\n\n" + "="*80)
print("=== 分析 forward 中的 hidden_states 处理 ===")
print("="*80)

# 读取 forward 代码
import inspect
source = inspect.getsource(transformer.forward)
print("forward 方法前 150 行:")
lines = source.split('\n')
for i, line in enumerate(lines[:150]):
    print(f"{i:3d}: {line}")

# 重点搜索可能改变 hidden_states 的操作
print("\n\n" + "="*80)
print("=== 搜索关键操作 ===")
print("="*80)
keywords = ['hidden_states', 'lq_tensor', 'deg_emb', 'img_ids', 'skip', 'residual', 'output', 'return', 'x_embedder', 'proj_out', 'norm']
for i, line in enumerate(lines):
    for kw in keywords:
        if kw in line.lower():
            print(f"{i:3d}: {line.strip()}")
            break
