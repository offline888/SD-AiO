import argparse
import os
import sys

import torch
from diffusers import Flux2KleinIRPipeline
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel
from PIL import Image
from train_flux2_ir import DegFeatExtractor, FLUX2ModulationV2

_script_dir = os.path.dirname(os.path.abspath(__file__))
for _p in [_script_dir, os.path.dirname(_script_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Flux.2 single-step image restoration inference."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/home/yhmi/data/model/flux.2-klein",
    )
    parser.add_argument(
        "--modulation_weights",
        type=str,
        default="data/output/flux2_lora/checkpoint-50000/modulation_weights.pt",
    )
    parser.add_argument("--lq_image", type=str, default=None)
    parser.add_argument(
        "--prompt", type=str, default="A cat holding a sign that says hello world"
    )
    parser.add_argument("--output_dir", type=str, default="outputs/ir_test")
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--num_inference_steps", type=int, default=1)
    parser.add_argument("--fixed_timestep", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--degradation_classifier_path", type=str, default=None)
    parser.add_argument("--dino_type", type=str, default="vits14")
    parser.add_argument("--num_deg_types", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--disable_cpu_offload",
        action="store_true",
        help="Disable sequential CPU offload for single-step inference (faster, more VRAM)",
    )
    parser.add_argument(
        "--resize_to",
        type=int,
        default=None,
        help="Resize image to target size before inference (e.g. 512). "
        "Default: keep original size (must be multiple of 64).",
    )

    if input_args is not None:
        return parser.parse_args(input_args)
    return parser.parse_args()


def build_inference_pipeline(
    pretrained_model_path: str,
    modulation_weights_path: str | None,
    degradation_classifier_path: str | None,
    dino_type: str,
    num_deg_types: int,
    device: str,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> Flux2KleinIRPipeline:
    pipe = Flux2KleinIRPipeline.from_pretrained(
        pretrained_model_path, torch_dtype=dtype
    )

    if getattr(args, "disable_cpu_offload", False):
        pipe.to(device)
    else:
        pipe.enable_sequential_cpu_offload()

    pipe.eval()

    pipe.vae = pipe.vae.to(dtype)

    _replace_modulation(pipe.transformer, device, dtype)

    class_emb = None
    if modulation_weights_path and os.path.exists(modulation_weights_path):
        state_dict = torch.load(
            modulation_weights_path, map_location="cpu", weights_only=True
        )
        transf_dict = {
            k.replace("transformer.", ""): v
            for k, v in state_dict.items()
            if "transformer." in k
        }
        pipe.transformer.load_state_dict(transf_dict, strict=False)
        print(f"[INFO] Loaded {len(transf_dict)} modulation tensors into transformer")

        for k, v in state_dict.items():
            if "class_embedding_U" in k:
                class_emb = v
                print(f"[INFO] Loaded class_embedding_U: shape={class_emb.shape}")
                break

    pipe.deg_extractor = DegFeatExtractor(
        transformer=pipe.transformer,
        num_deg_types=num_deg_types,
        weight_dtype=dtype,
        device=device,
        args=argparse.Namespace(
            degradation_classifier_path=degradation_classifier_path,
            dino_type=dino_type,
        ),
        class_embedding_U=class_emb,
    )
    return pipe


def _replace_modulation(transformer: Flux2Transformer2DModel, device, dtype):
    if not hasattr(transformer, "double_stream_modulation_img"):
        return
    orig = transformer.double_stream_modulation_img
    if isinstance(orig, FLUX2ModulationV2):
        return
    n_blocks = transformer.config.num_layers
    new_mod = FLUX2ModulationV2(
        dim=orig.linear.in_features,
        mod_param_sets=orig.mod_param_sets,
        bias=orig.linear.bias is not None,
        n_blocks=n_blocks,
    ).to(device=device, dtype=dtype)
    transformer.double_stream_modulation_img = new_mod
    print(f"[INFO] Replaced modulation with FLUX2ModulationV2 (n_blocks={n_blocks})")


def main(args):
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(args.dtype, torch.bfloat16)

    pipe = build_inference_pipeline(
        pretrained_model_path=args.pretrained_model_name_or_path,
        modulation_weights_path=args.modulation_weights,
        degradation_classifier_path=args.degradation_classifier_path,
        dino_type=args.dino_type,
        num_deg_types=args.num_deg_types,
        device=args.device,
        dtype=dtype,
        args=args,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    if args.lq_image:
        lq_image = Image.open(args.lq_image).convert("RGB")
        if args.resize_to is not None:
            lq_image = lq_image.resize((args.resize_to, args.resize_to))
    else:
        lq_image = Image.new("RGB", (512, 512), color=(128, 128, 128))

    generator = (
        torch.Generator(device=args.device).manual_seed(args.seed)
        if args.seed
        else None
    )
    output = pipe(
        lq_image=lq_image,
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        fixed_timestep=args.fixed_timestep,
        generator=generator,
    )

    out_path = os.path.join(args.output_dir, "output.png")
    output.images[0].save(out_path)
    print(f"[INFO] Saved to {out_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
