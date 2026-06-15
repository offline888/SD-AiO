import sys
sys.dont_write_bytecode = True
import argparse
import lpips
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from diffusers import Flux2KleinPipeline
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio as psnr_fn
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
from PIL import Image
from tqdm import tqdm
from src.data.dataset import PairedDataset

class ZeroShotTester:
    def __init__(self, config_path: str, output_dir: str = None):
        self.config = OmegaConf.load(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir or self.config.val.get("output_dir", "outputs/vis")
        self._load_pipeline()
        self._build_datasets()
        self._init_metrics()

    def _load_pipeline(self):
        model_path = self.config.network.model_path
        self.pipe = Flux2KleinPipeline.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        )
        self.pipe.to(self.device)
        print("Pipeline loaded successfully!")

    def _build_datasets(self):
        resolution = self.config.data.get("resolution", 512)
        datasets_config = self.config.data.datasets

        self.val_datasets = []
        self.dataset_prompts = []
        self.dataset_names = []

        val_keys = [k for k in datasets_config.keys() if k.startswith("ValDataset")]
        for ds_key in val_keys:
            ds_cfg = datasets_config[ds_key]
            ds = PairedDataset(
                lq_path=ds_cfg.lq_path,
                hq_path=ds_cfg.hq_path,
                resolution=resolution,
                prompt=ds_cfg.prompt,
                dataset_idx=len(self.val_datasets),
            )
            self.val_datasets.append(ds)
            self.dataset_prompts.append(ds_cfg.prompt)
            self.dataset_names.append(ds_key)
            print(f"Loaded dataset: {ds_key} with {len(ds)} samples")

        print(f"Total datasets: {len(self.val_datasets)}")

    def _init_metrics(self):
        self.lpips_model = lpips.LPIPS(net="vgg").to(self.device)
        self.lpips_model.requires_grad_(False)

    @torch.no_grad()
    def test(self, max_samples_per_dataset=None):
        results = {}
        guidance_scale = self.config.val.get("guidance_scale", 3.0)
        num_inference_steps = 1
        start_timestep = self.config.val.get("start_timestep", None)

        for ds_idx, dataset in enumerate(self.val_datasets):
            ds_name = self.dataset_names[ds_idx]
            prompt = self.dataset_prompts[ds_idx]
            print(f"\nTesting dataset: {ds_name}")
            print(f"Prompt: '{prompt}'")

            dataloader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )

            ssim_sum, psnr_sum, lpips_sum, count = 0.0, 0.0, 0.0, 0
            vis_save_dir = os.path.join(self.output_dir, ds_name)
            os.makedirs(vis_save_dir, exist_ok=True)
            vis_count = 0

            pbar = tqdm(dataloader, desc=f"{ds_name}")
            for batch in pbar:
                if max_samples_per_dataset and count >= max_samples_per_dataset:
                    break

                lq_tensor = batch["cond_pixel_values"].to(self.device)
                hq_tensor = batch["pixel_values"].to(self.device)

                # Convert LQ tensor to PIL
                lq_img = (
                    (lq_tensor[0].cpu() * 127.5 + 128)
                    .clamp(0, 255)
                    .permute(1, 2, 0)
                    .numpy()
                    .astype("uint8")
                )
                lq_pil = Image.fromarray(lq_img).convert("RGB")

                # Inference
                pipe_kwargs = dict(
                    prompt=prompt,
                    image=lq_pil,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                )
                if start_timestep is not None:
                    pipe_kwargs["sigmas"] = np.linspace(
                        start_timestep / 1000, 1 / num_inference_steps, num_inference_steps
                    ).tolist()

                output = self.pipe(**pipe_kwargs)
                gen_pil = output.images[0]

                # Convert generated PIL to tensor (before closing gen_pil)
                gen_np = (
                    torch.from_numpy(np.array(gen_pil)).float() / 255.0
                ).permute(2, 0, 1)
                gen_tensor = gen_np.unsqueeze(0).to(self.device)

                # Save visualization immediately (max 100 per dataset)
                if vis_count < 100:
                    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                    axes[0].imshow(lq_img)
                    axes[0].set_title(f"LQ #{count}", fontsize=12)
                    axes[0].axis("off")
                    axes[1].imshow(np.array(gen_pil))
                    axes[1].set_title(f"Pred #{count}", fontsize=12)
                    axes[1].axis("off")
                    plt.tight_layout()
                    save_path = os.path.join(vis_save_dir, f"vis_{vis_count:03d}.png")
                    plt.savefig(save_path, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    vis_count += 1

                lq_pil.close()
                gen_pil.close()

                # Normalize to [0, 1]
                pred_tensor = gen_tensor.clamp(0, 1)
                gt_tensor = (hq_tensor + 1.0) / 2.0

                # Metrics
                ssim_val = ssim_fn(pred_tensor, gt_tensor, data_range=1.0).item()
                psnr_val = psnr_fn(pred_tensor, gt_tensor, data_range=1.0).item()
                lpips_val = self.lpips_model(
                    pred_tensor * 2 - 1, gt_tensor * 2 - 1
                ).item()

                ssim_sum += ssim_val
                psnr_sum += psnr_val
                lpips_sum += lpips_val
                count += 1

            if count > 0:
                results[ds_name] = {
                    "ssim": ssim_sum / count,
                    "psnr": psnr_sum / count,
                    "lpips": lpips_sum / count,
                    "num_samples": count,
                    "prompt": prompt,
                }

        return results

    def print_summary(self, results):
        print("\n" + "=" * 60)
        print("Zero-Shot Test Results Summary")
        print("=" * 60)

        total_ssim, total_psnr, total_lpips = 0.0, 0.0, 0.0
        total_count = 0

        for ds_name, metrics in results.items():
            print(f"\n{ds_name}:")
            print(f"  Prompt: '{metrics['prompt']}'")
            print(f"  SSIM:  {metrics['ssim']:.4f}")
            print(f"  PSNR:  {metrics['psnr']:.2f} dB")
            print(f"  LPIPS: {metrics['lpips']:.4f}")
            print(f"  Samples: {metrics['num_samples']}")

            total_ssim += metrics['ssim'] * metrics['num_samples']
            total_psnr += metrics['psnr'] * metrics['num_samples']
            total_lpips += metrics['lpips'] * metrics['num_samples']
            total_count += metrics['num_samples']

        if total_count > 0:
            print("\n" + "-" * 60)
            print(f"Average (Total {total_count} samples):")
            print(f"  SSIM:  {total_ssim/total_count:.4f}")
            print(f"  PSNR:  {total_psnr/total_count:.2f} dB")
            print(f"  LPIPS: {total_lpips/total_count:.4f}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Zero-shot Flux2 Image Restoration Test")
    parser.add_argument(
        "--config", type=str,
        default="/home/yhmi/All_in_one/options/train/finetune_flux2.yaml",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Max samples per dataset (for quick test)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save visualizations"
    )
    args = parser.parse_args()

    tester = ZeroShotTester(args.config, output_dir=args.output_dir)
    results = tester.test(max_samples_per_dataset=args.max_samples)
    tester.print_summary(results)


if __name__ == "__main__":
    main()
