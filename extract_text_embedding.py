import argparse
import os
import torch
from omegaconf import OmegaConf
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel

def main(args):
    # 1. 加载配置
    config = OmegaConf.load(args.config)
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16  # 统一用 fp16 提取，既不损失精度又节省硬盘

    # 2. 从配置获取模型路径
    model_path = config.network.pretrained_model_name_or_path
    revision = config.network.get("revision", None)
    variant = config.network.get("variant", None)

    print(f"Loading Text Encoders from: {model_path}")
    
    # 3. 初始化 Tokenizers 和 Encoders
    tokenizer = CLIPTokenizer.from_pretrained(
        model_path, subfolder="tokenizer", revision=revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_path, subfolder="text_encoder", revision=revision, variant=variant, torch_dtype=weight_dtype
    ).to(device)
    text_encoder.eval()

    tokenizer_2 = CLIPTokenizer.from_pretrained(
        model_path, subfolder="tokenizer_2", revision=revision
    )
    text_encoder_2 = T5EncoderModel.from_pretrained(
        model_path, subfolder="text_encoder_2", revision=revision, variant=variant, torch_dtype=weight_dtype
    ).to(device)
    text_encoder_2.eval()

    # 4. 解析配置文件中的 prompts
    datasets_cfg = config.data.get("datasets", {})
    
    # 按照 YAML 里的顺序遍历，这与你 train_restore.py 里的 ds_idx 分配逻辑完全一致
    with torch.no_grad():
        for ds_idx, (ds_name, ds_cfg) in enumerate(datasets_cfg.items()):
            prompt = ds_cfg.get("prompt", "")
            print(f"[{ds_name} | idx: {ds_idx}] Extracting prompt: '{prompt}'")
            
            # 编码 CLIP 特征
            text_inputs = tokenizer(
                [prompt] if prompt else [""],
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            pooled_prompt_embeds = text_encoder(**text_inputs).pooler_output

            # 编码 T5 特征
            t5_inputs = tokenizer_2(
                [prompt] if prompt else [""],
                padding="max_length",
                max_length=512,  # Flux 的 T5 默认长度
                truncation=True,
                return_tensors="pt",
            )
            t5_inputs = {k: v.to(device) for k, v in t5_inputs.items()}
            prompt_embeds = text_encoder_2(**t5_inputs).last_hidden_state

            # 生成对应的 text_ids (Flux 架构所需)
            text_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=weight_dtype)

            # 保存到硬盘
            save_dict = {
                "prompt_embeds": prompt_embeds[0].cpu(),
                "pooled_prompt_embeds": pooled_prompt_embeds[0].cpu(),
                "text_ids": text_ids[0].cpu()
            }
            save_path = os.path.join(save_dir, f"dataset_{ds_idx}_embeds.pt")
            torch.save(save_dict, save_path)
            
    print(f"\n✅ All done! Successfully extracted {len(datasets_cfg)} embeddings to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--save_dir", type=str, default="./cached_embeddings", help="Directory to save embeddings")
    args = parser.parse_args()
    main(args)