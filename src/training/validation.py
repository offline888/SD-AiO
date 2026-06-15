import os

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from src.flux2.pipelines.flux2_klein import Flux2KleinIRPipeline
from src.models.deg_extractor import DegFeatExtractor


IMGNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMGNET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def validate(
    pipeline,
    val_items: list,
    guidance_scale: float,
    fixed_timestep: int,
    device: torch.device,
    output_dir: str,
    global_step: int,
):
    os.makedirs(output_dir, exist_ok=True)
    step_dir = os.path.join(output_dir, f"step_{global_step:06d}")
    os.makedirs(step_dir, exist_ok=True)

    pipeline.eval()

    mean = IMGNET_MEAN.to(device)
    std  = IMGNET_STD.to(device)

    with torch.no_grad():
        for item in val_items:
            lq_tensor = item["lq_pixel_values"].to(device)
            hq_tensor = item["hq_pixel_values"].to(device)

            lq_pil = TF.to_pil_image((lq_tensor * std + mean).clamp(0, 1))

            pred_output = pipeline(
                lq_image=lq_pil,
                prompt_embeds=item["prompt_embeds"].unsqueeze(0).to(device),
                num_inference_steps=1,
                guidance_scale=guidance_scale,
                fixed_timestep=fixed_timestep,
                output_type="np",
                return_dict=True,
            )
            pred_np = pred_output.images[0]
            pred_tensor = torch.from_numpy(pred_np).permute(2, 0, 1)

            lq_vis = (lq_tensor.to(device) * std + mean).clamp(0, 1).cpu()
            hq_vis = (hq_tensor.float().to(device) * std + mean).clamp(0, 1).cpu()
            pred_vis = pred_tensor.clamp(0, 1)

            grid = torch.cat([lq_vis, pred_vis, hq_vis], dim=2)
            dataset_label = item.get("label", "unknown")
            Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).astype("uint8")).save(
                os.path.join(step_dir, f"{dataset_label}.png")
            )
