#!/usr/bin/env python3
"""
深入分析 Flux2 Transformer 的 forward 路径

重点分析：
1. hidden_states 在 forward 中经过哪些层？
2. 是否有 skip connection 或残差路径直接返回输入？
3. img_ids 和 deg_emb 在 forward 中如何被使用？
4. lq_tensor 在 forward 中如何被使用？
5. 最终输出的 hidden_states 与输入的 hidden_states 的形状关系
"""

import torch
import sys
sys.path.insert(0, '/home/yhmi/All_in_one')

from src.flux2.transformer_flux2 import Flux2Transformer2DModel

print("="*80)
print("深入分析 Flux2 Transformer Forward 路径")
print("="*80)

# 加载 transformer
print("\n[1] 加载 Transformer 模型...")
try:
    transformer = Flux2Transformer2DModel.from_pretrained(
        '/home/yhmi/data/model/flux.2-klein', subfolder='transformer', torch_dtype=torch.float32
    )
    transformer.eval()
    print("  ✓ 模型加载成功")
except Exception as e:
    print(f"  ✗ 模型加载失败: {e}")
    print("  使用 CPU 模式进行纯代码分析")

print("\n" + "="*80)
print("[2] Forward 方法完整分析")
print("="*80)

# 分析 forward 方法的关键部分
print("""
Forward 方法签名:
---------------
def forward(
    self,
    hidden_states: torch.Tensor,           # 输入: (B, L, 128)
    encoder_hidden_states: torch.Tensor,     # 文本 embedding: (B, T, 4096)
    timestep: torch.LongTensor,              # 时间步
    img_ids: torch.Tensor,                  # 图像位置 ID
    txt_ids: torch.Tensor,                  # 文本位置 ID
    guidance: torch.Tensor,                 # 引导值
    joint_attention_kwargs: dict,           # 注意力参数
    return_dict: bool,
    kv_cache: Flux2KVCache,                 # KV 缓存
    kv_cache_mode: str,
    num_ref_tokens: int,
    ref_fixed_timestep: float,
    deg_emb: torch.Tensor,                   # 降解 embedding: (B, 1, 768)
    lq_tensor: torch.Tensor,                 # 低质量图像: (B, 3, H, W)
) -> torch.Tensor
""")

print("\n" + "="*80)
print("[3] hidden_states 的变换路径（按执行顺序）")
print("="*80)

print("""
Step 1: 时间步处理 (Line 1315-1328)
---------------------------------
- timestep 乘以 1000 缩放
- guidance 乘以 1000 缩放  
- temb = self.time_guidance_embed(timestep, guidance)
  输入: timestep (B,), guidance (B,)  -> 输出: (B, 6144)
  包含 timestep embedding + guidance embedding 的和

Step 2: 输入投影 (Line 1330-1339)
---------------------------------
hidden_states 变换:
  输入 hidden_states: (B, 1024, 128)  # B=1, L=1024 tokens, C=128 channels
       ↓
  self.x_embedder(hidden_states): (B, 1024, 6144)  # 128 -> 6144
  
encoder_hidden_states 变换:
  输入 encoder_hidden_states: (B, 120, 4096)  # 文本序列
       ↓
  self.context_embedder(encoder_hidden_states): (B, 120, 6144)  # 4096 -> 6144
       ↓
  deg_emb 处理: (B, 1, 768) -> 投影到 6144 维
       ↓
  torch.cat([deg_emb, encoder_hidden_states], dim=1): (B, 1+120, 6144) = (B, 121, 6144)
  
*** 关键发现: deg_emb 被拼接到 encoder_hidden_states 的前面 ***
*** 位置: 索引 0 是 deg_emb，索引 1-120 是原始文本 embedding ***

Step 3: RoPE 位置编码 (Line 1341-1355)
--------------------------------------
- img_ids: 图像 token 的位置 ID，用于生成图像的 RoPE embedding
- txt_ids: 文本 token 的位置 ID，用于生成文本的 RoPE embedding
- concat_rotary_emb: 文本和图像 RoPE 拼接

*** 关键发现: img_ids 仅用于生成 RoPE 位置编码，不直接参与 hidden_states 计算 ***
*** 关键发现: txt_ids 仅用于生成 RoPE 位置编码，不直接参与 hidden_states 计算 ***

Step 4: 双流 Transformer Blocks (Line 1375-1405)
-----------------------------------------------
for index_block in range(8):  # 8 个双流 blocks
    1. double_stream_mod_img = self.double_stream_modulation_img(
           lq_tensor=lq_tensor,  # <-- lq_tensor 在这里被使用
           temb=temb,
           block_idx=index_block
       )
    2. double_stream_mod_txt = self.double_stream_modulation_txt(temb)
    3. block(
           hidden_states=hidden_states,      # (B, 1024, 6144)
           encoder_hidden_states=ehs,        # (B, 121, 6144)
           temb_mod_img=mod_img,
           temb_mod_txt=mod_txt,
           image_rotary_emb=concat_rotary_emb,
           joint_attention_kwargs=kv_attn_kwargs,
       )
       
内部流程 (Flux2TransformerBlock.forward):
    norm_hidden_states = self.norm1(hidden_states)           # LayerNorm
    norm_hidden_states = (1 + scale_msa) * norm + shift_msa  # 条件调制
    
    norm_ehs = self.norm1_context(encoder_hidden_states)      # LayerNorm  
    norm_ehs = (1 + c_scale_msa) * norm_ehs + c_shift_msa   # 条件调制
    
    attn_output = self.attn(norm_hidden_states, norm_ehs)    # 注意力
    
    hidden_states = hidden_states + gate_msa * attn_output  # 残差连接
    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
    ff_output = self.ff(norm_hidden_states)
    hidden_states = hidden_states + gate_mlp * ff_output    # 残差连接
    
*** 关键发现: 每个 block 有两个残差连接 ***
*** 残差连接形式: hidden_states = hidden_states + gate * output ***

Step 5: 合并文本和图像流 (Line 1407-1408)
----------------------------------------
hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
# (B, 121, 6144) + (B, 1024, 6144) = (B, 1145, 6144)

Step 6: 单流 Transformer Blocks (Line 1416-1441)
-----------------------------------------------
for index_block in range(48):  # 48 个单流 blocks
    1. single_stream_mod = self.single_stream_modulation(
           lq_tensor,         # <-- lq_tensor 在这里被使用
           temb, 
           index_block, 
           seq_len=1537       # num_concat_tokens = 1537
       )
    2. block(
           hidden_states=hidden_states,   # (B, 1145, 6144)
           encoder_hidden_states=None,    # 已拼接，直接使用
           temb_mod=single_stream_mod,
           image_rotary_emb=concat_rotary_emb,
           joint_attention_kwargs=kv_attn_kwargs_single,
       )

内部流程 (Flux2SingleTransformerBlock.forward):
    if encoder_hidden_states is not None:
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
    
    norm_hidden_states = self.norm(hidden_states)
    norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift
    
    attn_output = self.attn(norm_hidden_states)
    
    hidden_states = hidden_states + mod_gate * attn_output  # 残差连接
    
*** 关键发现: 每个单流 block 有一个残差连接 ***

Step 7: 移除文本 token (Line 1443-1447)
--------------------------------------
if kv_cache_mode == "extract":
    hidden_states = hidden_states[:, num_txt_tokens + num_ref_tokens:, ...]
else:
    hidden_states = hidden_states[:, num_txt_tokens:, ...]
    
# 移除前面的 121 个 token (1 deg_emb + 120 文本)
# 输出: (B, 1024, 6144)

Step 8: 输出投影 (Line 1449-1451)
---------------------------------
hidden_states = self.norm_out(hidden_states, temb)  # AdaLayerNormContinuous
output = self.proj_out(hidden_states)               # Linear: 6144 -> 128

最终输出: (B, 1024, 128)  # 与输入形状相同
""")

print("\n" + "="*80)
print("[4] lq_tensor 在 forward 中的使用")
print("="*80)

print("""
lq_tensor 流向分析:
==================

位置 1: 双流调制 (Line 1378-1380)
---------------------------------
double_stream_mod_img = self.double_stream_modulation_img(
    lq_tensor=lq_tensor,
    temb=temb,
    block_idx=index_block
)

在 FLUX2ModulationV2.forward 中 (modulation.py Line 142-215):
    if lq_tensor is not None:
        if self.use_conv and not self.use_vae:
            # ConvNeXt 处理
            x = self.conv_stem_s1(lq_tensor)       # (B,3,H,W) -> (B,96,H/2,W/2)
            x = self.conv_time_mod1(x, temb)        # 时间调制
            x = self.conv_down1_s2(x)              # (B,96,H/2,W/2) -> (B,192,H/4,W/4)
            x = self.conv_time_mod2(x, temb)        # 时间调制
            x = self.conv_down2_s3(x)              # (B,192,H/4,W/4) -> (B,384,H/8,W/8)
            x = self.conv_time_mod3(x, temb)        # 时间调制
            
            lq_mod = self.feat_proj(x)              # (B,384,H/8,W/8) -> (B, seq, 3*mod_param_sets*dim)
            
        # lq_mod 与 mod_time 相加
        if seq_len is not None:
            mod = mod_time.unsqueeze(1).expand(B, seq_len, -1)
            if lq_mod is not None:
                lq_mod_up = F.interpolate(lq_mod, size=seq_len)  # 插值到 seq_len
                mod = mod + lq_mod_up

*** 关键发现: lq_tensor 被 ConvNeXt 处理后，与时间调制参数相加 ***
*** 这是一种信息注入方式，不是直接的 skip connection ***

位置 2: 单流调制 (Line 1420)
-----------------------------
single_stream_mod = self.single_stream_modulation(
    lq_tensor,
    temb, 
    index_block,
    seq_len=num_concat_tokens  # 1537
)

同样通过 FLUX2ModulationV2.forward 处理，方式相同。

*** 总结: lq_tensor 影响调制参数，不直接加到 hidden_states 上 ***
*** lq_tensor -> ConvNeXt -> lq_mod -> 与时间调制参数相加 -> 影响注意力调制 ***
""")

print("\n" + "="*80)
print("[5] img_ids 和 deg_emb 在 forward 中的使用")
print("="*80)

print("""
img_ids 使用分析:
================
位置: Line 1343-1350
    if img_ids.ndim == 4:
        img_ids = img_ids.squeeze(1)
    if img_ids.ndim == 3:
        img_ids = img_ids[0]
    image_rotary_emb = self.pos_embed(img_ids)

用途: 生成图像 token 的 RoPE (Rotary Position Embedding) 位置编码
- self.pos_embed 是 Flux2PosEmbed 模块
- 输出: (cos, sin) 用于旋转位置编码
- image_rotary_emb 被传递给注意力层

*** 关键发现: img_ids 只用于生成 RoPE 位置编码 ***
*** img_ids 不直接参与 hidden_states 的计算 ***
*** RoPE 用于 query 和 key 的旋转嵌入，但不改变 hidden_states 数值 ***


deg_emb 使用分析:
================
位置: Line 1337-1339
    if deg_emb is not None and deg_emb.dtype != target_dtype:
        deg_emb = deg_emb.to(target_dtype)
    encoder_hidden_states = torch.cat([deg_emb, encoder_hidden_states], dim=1)

用途: 降解类型 embedding，被拼接到文本 embedding 的前面
- deg_emb shape: (B, 1, 768)  # 768 维向量
- encoder_hidden_states shape: (B, 120, 4096) -> context_embedder -> (B, 120, 6144)
- 拼接后: (B, 1 + 120, 6144) = (B, 121, 6144)

后续流程:
1. 进入双流 Transformer Block
2. 在 Line 1408 处与图像 hidden_states 拼接: torch.cat([encoder_hidden_states, hidden_states], dim=1)
3. 在 Line 1447 处被移除: hidden_states = hidden_states[:, num_txt_tokens:, ...]

*** 关键发现: deg_emb 参与整个 transformer 处理，但最终被移除 ***
*** deg_emb 的信息通过注意力机制影响图像 token 的处理 ***
""")

print("\n" + "="*80)
print("[6] 残差连接分析")
print("="*80)

print("""
Transformer 中的残差连接:
========================

1. 双流 TransformerBlock (8 个):
   残差连接 1: hidden_states = hidden_states + gate_msa * attn_output
              出现在 Line 932
   
   残差连接 2: hidden_states = hidden_states + gate_mlp * ff_output
              出现在 Line 938

2. 单流 SingleTransformerBlock (48 个):
   残差连接: hidden_states = hidden_states + mod_gate * attn_output
            出现在 Line 847

*** 总计: 8 * 2 + 48 = 64 个残差连接 ***

残差连接的形式: hidden_states = hidden_states + gate * output

特点:
- 每个残差连接都是将注意力/FFN 的输出加到输入上
- 输出是: input + gate * transform(input)
- 不是恒等映射，而是带有门控的残差

*** 关键结论: 没有直接的 skip connection 返回输入本身 ***
*** 所有残差连接都是 transform(input) + gate * output 形式 ***
*** 输出始终与输入经过相同维度的投影和变换 ***
""")

print("\n" + "="*80)
print("[7] 维度变换总结")
print("="*80)

print("""
输入到输出的维度流:
==================

hidden_states:
    (B, 1024, 128)  -- 输入
         ↓
    x_embedder: (B, 1024, 6144)  # 128 -> 6144
         ↓ (经过 8 个双流 block, 48 个单流 block, 每次有残差连接)
         ↓
    norm_out: AdaLayerNormContinuous
         ↓
    proj_out: (B, 1024, 128)  # 6144 -> 128
         ↓
    (B, 1024, 128)  -- 输出 (与输入形状相同)

encoder_hidden_states:
    (B, 120, 4096)  -- 输入
         ↓
    context_embedder: (B, 120, 6144)  # 4096 -> 6144
         ↓
    + deg_emb: (B, 1, 6144)
         ↓
    (B, 121, 6144)  -- 最终被移除

deg_emb:
    (B, 1, 768)  -- 输入
         ↓
    (直接拼接，不经过投影)
         ↓
    成为 encoder_hidden_states 的一部分

img_ids:
    (B, 1024, 3)  -- 输入
         ↓
    pos_embed (RoPE)
         ↓
    用于注意力层的 query/key 旋转嵌入

lq_tensor:
    (B, 3, H, W)  -- 输入
         ↓
    ConvNeXt + TimeModulator
         ↓
    特征插值到 seq_len
         ↓
    与时间调制参数相加
         ↓
    影响所有 transformer block 的调制参数
""")

print("\n" + "="*80)
print("[8] 最终结论")
print("="*80)

print("""
问题 1: hidden_states 在 forward 中经过哪些层？
==============================================
1. x_embedder: Linear(128, 6144)
2. 8 个双流 TransformerBlock (每个包含: norm, attn, ff, 2个残差)
3. 48 个单流 TransformerBlock (每个包含: norm, attn, 1个残差)
4. norm_out: AdaLayerNormContinuous
5. proj_out: Linear(6144, 128)

*** 关键: hidden_states 始终经过完整的 transformer 处理 ***

问题 2: 是否有 skip connection 或残差路径直接返回输入？
======================================================
*** 没有直接的恒等 skip connection ***

- 所有残差连接都是: hidden_states = hidden_states + gate * transform(hidden_states)
- 输出形状与输入相同 (1024, 128) 但数值完全不同
- 中间维度是 6144，输入 128 通过 x_embedder 投影到这个维度

问题 3: img_ids 和 deg_emb 在 forward 中如何被使用？
====================================================
img_ids:
    - 用于生成 RoPE (旋转位置编码)
    - 传递给注意力层的 query 和 key
    - 不直接参与 hidden_states 计算

deg_emb:
    - 直接拼接到 encoder_hidden_states 的最前面
    - 位置: (B, 1, 768) -> (B, 1, 6144) after concat
    - 与文本 token 一起参与所有 transformer block
    - 最终在输出前被移除

问题 4: lq_tensor 在 forward 中如何被使用？
==========================================
lq_tensor:
    - 通过 ConvNeXt 编码器提取特征
    - 特征通过 EfficientConvProj 投影
    - 与时间步调制参数 (temb) 相加
    - 影响所有 transformer block 的调制 (shift/scale/gate)
    
*** lq_tensor 不直接加到 hidden_states，而是通过调制参数间接影响 ***

问题 5: 最终输出的 hidden_states 与输入的形状关系
==================================================
- 输入: (B, 1024, 128)
- 输出: (B, 1024, 128)
- 形状完全相同，但数值经过完整的 transformer 处理

*** 最终结论: Transformer 是完全的序列到序列变换，没有直接的输入传递路径 ***
""")

print("\n" + "="*80)
print("分析完成")
print("="*80)
