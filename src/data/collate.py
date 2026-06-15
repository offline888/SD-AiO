import torch

def collate_fn(examples):
    return {
        "hq_pixel_values": torch.stack([e["hq_pixel_values"] for e in examples]),
        "lq_pixel_values": torch.stack([e["lq_pixel_values"] for e in examples]),
        "prompts": [e["prompt"] for e in examples],
        "dataset_indices": torch.tensor(
            [e["dataset_idx"] for e in examples], dtype=torch.long
        ),
        "deg_types": [e["deg_type"] for e in examples],
    }
