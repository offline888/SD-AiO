import torch


def compute_text_embeddings(prompt, pipeline):
    with torch.no_grad():
        prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=prompt,
            max_sequence_length=512,
        )
    return prompt_embeds, text_ids
