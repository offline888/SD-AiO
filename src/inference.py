import argparse
import os
import re

import torch
from PIL import Image
import yaml

from src.models import FLUX2ModulationV2, DegFeatExtractor
from src.flux2.pipelines.flux2_klein import Flux2KleinIRPipeline
from src.flux2 import Flux2Transformer2DModel


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
        "--data_yaml",
        type=str,
        default=None,
        help="Path to YAML config file (e.g. options/train/data.yaml). "
        "Inference will be run on all images under the lq_path of each ValDataset entry, "
        "saved under output_dir/<deg_type>/<image_name>.",
    )
    parser.add_argument(
        "--prompt", type=str, default="A cat holding a sign that says hello world"
    )
    parser.add_argument("--output_dir", type=str, default="outputs/ir_test")
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--num_inference_steps", type=int, default=1)
    parser.add_argument("--fixed_timestep", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--degradation_classifier_path", type=str, default=None, required=True)
    parser.add_argument("--dino_type", type=str, default="vits14")
    parser.add_argument("--num_deg_types", type=int, default=4)
    parser.add_argument(
        "--mod_lq_type",
        type=str,
        default="convnext",
        choices=["convnext", "vae"],
        help="Backbone type for FLUX2ModulationV2",
    )
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


def _nat_sort_key(s: str):
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", s)]


def _replace_modulation(transformer, vae_path, device, dtype, mod_lq_type: str):
    if not hasattr(transformer, "double_stream_modulation_img"):
        return
    orig = transformer.double_stream_modulation_img
    if isinstance(orig, FLUX2ModulationV2):
        return
    use_conv = (mod_lq_type == "convnext")
    use_vae = (mod_lq_type == "vae")
    new_mod = FLUX2ModulationV2(
        dim=orig.linear.in_features,
        mod_param_sets=orig.mod_param_sets,
        bias=orig.linear.bias is not None,
        use_block_emb=True,
        use_conv=use_conv,
        use_vae=use_vae,
        vae_path=vae_path,
    ).to(device=device, dtype=dtype)
    transformer.double_stream_modulation_img = new_mod
    print(f"[INFO] Replaced modulation with FLUX2ModulationV2 ({mod_lq_type})")


def build_inference_pipeline(args):
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(args.dtype, torch.bfloat16)

    pipe = Flux2KleinIRPipeline.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=dtype
    )

    pipe.vae = pipe.vae.to(dtype=dtype)
    pipe.transformer = pipe.transformer.to(dtype=dtype)
    pipe.transformer = pipe.transformer.to(device="cpu", dtype=dtype)

    if getattr(args, "disable_cpu_offload", False):
        pipe.to(args.device)
        pipe.transformer = pipe.transformer.to(device=args.device, dtype=dtype)
        pipe.vae = pipe.vae.to(device=args.device, dtype=dtype)
        if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
            pipe.text_encoder = pipe.text_encoder.to(device=args.device, dtype=dtype)
    else:
        pipe.enable_sequential_cpu_offload()

    pipe.eval()

    _replace_modulation(
        pipe.transformer,
        args.pretrained_model_name_or_path,
        args.device,
        dtype,
        mod_lq_type=args.mod_lq_type,
    )

    class_emb = None
    if args.modulation_weights and os.path.exists(args.modulation_weights):
        state_dict = torch.load(
            args.modulation_weights, map_location="cpu", weights_only=True
        )
        loaded_mod_count = 0
        for k, v in state_dict.items():
            if k.startswith("double_stream_") or k.startswith("single_"):
                pipe.transformer.state_dict()[k].copy_(v)
                loaded_mod_count += 1
            elif k == "deg_embedding":
                pipe.transformer.state_dict()[k].copy_(v)
                print(f"[INFO] Loaded deg_embedding: shape={v.shape}")
            elif "class_embedding_U" in k:
                class_emb = v
                print(f"[INFO] Loaded class_embedding_U: shape={class_emb.shape}")
        print(f"[INFO] Loaded {loaded_mod_count} modulation tensors into transformer")

    mod_img = pipe.transformer.double_stream_modulation_img
    print("\n[INFO] double_stream_modulation_img structure:")
    print(f"  Type: {type(mod_img).__name__}")
    print(f"  dim: {mod_img.dim}, mod_param_sets: {mod_img.mod_param_sets}")
    for name, param in sorted(mod_img.named_parameters(), key=lambda x: x[0]):
        print(f"  {name}: {param.shape}")
    print()

    pipe.deg_extractor = DegFeatExtractor(
        inner_dim=pipe.transformer.inner_dim,
        num_deg_types=args.num_deg_types,
        weight_dtype=dtype,
        args=argparse.Namespace(
            degradation_classifier_path=args.degradation_classifier_path,
            dino_type=args.dino_type,
        ),
        deg_embedding=None,
    )
    pipe.transformer.register_parameter("deg_embedding", pipe.deg_extractor.deg_embedding)
    return pipe


def run_inference_single(
    pipe, lq_image: Image.Image, prompt: str, args, generator: torch.Generator | None
) -> Image.Image:
    output = pipe(
        lq_image=lq_image,
        prompt=prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        fixed_timestep=args.fixed_timestep,
        generator=generator,
    )
    return output.images[0]


def run_inference_from_yaml(pipe, yaml_path: str, args):
    with open(yaml_path, "r") as f:
        yaml_cfg = yaml.safe_load(f)

    val_keys = sorted(
        [k for k in yaml_cfg if k.startswith("ValDataset")],
        key=lambda k: int(re.search(r"\d+", k).group()),
    )
    if not val_keys:
        print("[WARN] No ValDataset entries found in YAML.")
        return

    total_images = 0
    for ds_key in val_keys:
        ds = yaml_cfg[ds_key]
        lq_path: str = ds.get("lq_path", "")
        prompt: str = ds.get("prompt", args.prompt)
        deg_type: str = ds.get("deg_type", os.path.basename(os.path.normpath(lq_path)))

        if not os.path.isdir(lq_path):
            print(f"[WARN] Skipping {ds_key}: lq_path '{lq_path}' is not a directory.")
            continue

        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}
        img_files = sorted(
            [f for f in os.listdir(lq_path) if os.path.splitext(f.lower())[1] in exts],
            key=_nat_sort_key,
        )
        if not img_files:
            print(f"[WARN] No images found in {lq_path}.")
            continue

        ds_output_dir = os.path.join(args.output_dir, deg_type)
        os.makedirs(ds_output_dir, exist_ok=True)

        print(f"\n[{ds_key}] deg_type={deg_type}, lq_path={lq_path}, {len(img_files)} images")
        for i, fname in enumerate(img_files):
            lq_full_path = os.path.join(lq_path, fname)
            lq_image = Image.open(lq_full_path).convert("RGB")
            if args.resize_to is not None:
                lq_image = lq_image.resize((args.resize_to, args.resize_to))

            seed = args.seed + i if args.seed else None
            generator = (
                torch.Generator(device=args.device).manual_seed(seed)
                if seed
                else None
            )

            result = run_inference_single(pipe, lq_image, prompt, args, generator)
            out_path = os.path.join(ds_output_dir, fname)
            result.save(out_path)
            total_images += 1
            print(f"  [{i+1}/{len(img_files)}] {fname} -> {out_path}")

    print(f"\n[INFO] YAML inference complete. Total images processed: {total_images}")


def run_inference(args):
    pipe = build_inference_pipeline(args)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.data_yaml:
        run_inference_from_yaml(pipe, args.data_yaml, args)
    elif args.lq_image:
        lq_image = Image.open(args.lq_image).convert("RGB")
        if args.resize_to is not None:
            lq_image = lq_image.resize((args.resize_to, args.resize_to))

        generator = (
            torch.Generator(device=args.device).manual_seed(args.seed)
            if args.seed
            else None
        )
        output = run_inference_single(pipe, lq_image, args.prompt, args, generator)
        out_path = os.path.join(args.output_dir, "output.png")
        output.save(out_path)
        print(f"[INFO] Saved to {out_path}")
    else:
        lq_image = Image.new("RGB", (512, 512), color=(128, 128, 128))
        generator = (
            torch.Generator(device=args.device).manual_seed(args.seed)
            if args.seed
            else None
        )
        output = run_inference_single(pipe, lq_image, args.prompt, args, generator)
        out_path = os.path.join(args.output_dir, "output.png")
        output.save(out_path)
        print(f"[INFO] Saved to {out_path}")
