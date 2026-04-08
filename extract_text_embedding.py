import argparse
import os
import torch
from omegaconf import OmegaConf
from transformers import Qwen3ForCausalLM, Qwen2TokenizerFast


def get_weight_dtype(mixed_precision):
    """根据 mixed_precision 配置返回对应的 dtype"""
    if mixed_precision == "bf16":
        return torch.bfloat16
    elif mixed_precision == "fp16":
        return torch.float16
    else:
        return torch.float32


def prepare_text_ids(
    x: torch.Tensor,  # (B, L, D) or (L, D)
    t_coord: torch.Tensor | None = None,
):
    """
    准备文本的位置 ID，与 Flux2KleinPipeline._prepare_text_ids 保持一致
    """
    B, L, _ = x.shape
    out_ids = []

    for i in range(B):
        t = torch.arange(1) if t_coord is None else t_coord[i]
        h = torch.arange(1)
        w = torch.arange(1)
        l = torch.arange(L)

        coords = torch.cartesian_prod(t, h, w, l)
        out_ids.append(coords)

    return torch.stack(out_ids)


def get_qwen3_prompt_embeds(
    text_encoder: Qwen3ForCausalLM,
    tokenizer: Qwen2TokenizerFast,
    prompt: str | list[str],
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    max_sequence_length: int = 512,
    hidden_states_layers: list[int] = (9, 18, 27),
):
    """
    获取 prompt embeddings，与 Flux2KleinPipeline._get_qwen3_prompt_embeds 保持一致

    Args:
        text_encoder: Qwen3 文本编码器
        tokenizer: Qwen2 tokenizer
        prompt: 单个 prompt 或 prompt 列表
        dtype: 输出 dtype，默认使用 text_encoder 的 dtype
        device: 输出设备，默认使用 text_encoder 的设备
        max_sequence_length: 最大序列长度
        hidden_states_layers: 用于拼接的隐藏层索引

    Returns:
        prompt_embeds: (batch_size, seq_len, num_channels * hidden_dim)
    """
    dtype = text_encoder.dtype if dtype is None else dtype
    device = text_encoder.device if device is None else device

    prompt = [prompt] if isinstance(prompt, str) else prompt

    all_input_ids = []
    all_attention_masks = []

    for single_prompt in prompt:
        # 使用 apply_chat_template 包装 prompt，与 Pipeline 保持一致
        messages = [{"role": "user", "content": single_prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )

        all_input_ids.append(inputs["input_ids"])
        all_attention_masks.append(inputs["attention_mask"])

    input_ids = torch.cat(all_input_ids, dim=0).to(device)
    attention_mask = torch.cat(all_attention_masks, dim=0).to(device)

    # Forward pass through the model
    output = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )

    # 使用多层隐藏状态拼接，与 Pipeline 保持一致
    out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
    out = out.to(dtype=dtype, device=device)

    batch_size, num_channels, seq_len, hidden_dim = out.shape
    prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

    return prompt_embeds


def main(args):
    config = OmegaConf.load(args.config)
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mixed_precision = config.accelerator.get("mixed_precision", "no")
    weight_dtype = get_weight_dtype(mixed_precision)
    print(f"[Embedding] Using dtype: {weight_dtype}")

    model_path = config.network.pretrained_model_name_or_path
    revision = config.network.get("revision", None)
    variant = config.network.get("variant", None)

    # 支持从配置中指定 hidden_states_layers，默认使用 (9, 18, 27)
    hidden_states_layers = tuple(config.network.get("text_encoder_out_layers", [9, 18, 27]))
    max_sequence_length = config.network.get("tokenizer_max_length", 512)

    print(f"Loading Text Encoder (Qwen3) from: {model_path}")

    tokenizer = Qwen2TokenizerFast.from_pretrained(
        model_path, subfolder="tokenizer", revision=revision
    )
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        model_path, subfolder="text_encoder", revision=revision, variant=variant, torch_dtype=weight_dtype
    ).to(device)
    text_encoder.eval()

    datasets_cfg = config.data.get("datasets", {})

    with torch.no_grad():
        for ds_idx, (ds_name, ds_cfg) in enumerate(datasets_cfg.items()):
            prompt = ds_cfg.get("prompt", "")
            print(f"[{ds_name} | idx: {ds_idx}] Extracting: '{prompt}'")

            # 使用与 Flux2KleinPipeline 一致的 embedding 提取方式
            prompt_embeds = get_qwen3_prompt_embeds(
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                prompt=prompt if prompt else "",
                hidden_states_layers=hidden_states_layers,
                max_sequence_length=max_sequence_length,
            )  # (1, seq_len, num_channels * hidden_dim)

            seq_len = prompt_embeds.shape[1]

            # 准备 text_ids，与 Pipeline 保持一致
            text_ids = prepare_text_ids(prompt_embeds)

            # 使用最后一个有效 token 作为 pooled 表示
            pooled_prompt_embeds = prompt_embeds[0, -1]  # (num_channels * hidden_dim,)

            save_dict = {
                "prompt_embeds": prompt_embeds.cpu(),
                "pooled_prompt_embeds": pooled_prompt_embeds.cpu(),
                "text_ids": text_ids.cpu(),
            }
            save_path = os.path.join(save_dir, f"dataset_{ds_idx}_embeds.pt")
            torch.save(save_dict, save_path)

    print(f"\nAll done! Extracted {len(datasets_cfg)} embeddings to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--save_dir", type=str, default="./cached_embeddings", help="Directory to save embeddings")
    args = parser.parse_args()
    main(args)
