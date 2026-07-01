"""Compute LQ vs GT baseline metrics for 3D test sets.

PSNR(Y), SSIM(Y), LPIPS — no model involved.
Usage:
  python compute_baseline.py --config configs/tasks_3d.yaml        # all metrics
  python compute_baseline.py --config configs/tasks_3d.yaml --no_lpips  # fast: skip LPIPS
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))
from utils.dataset import build_datasets
from utils.image_utils import rgb2ycbcr, calculate_psnr, calculate_ssim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tasks_3d.yaml")
    parser.add_argument("--no_lpips", action="store_true", help="Skip LPIPS (faster)")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id for LPIPS, -1=cpu")
    parser.add_argument("--json_out", default="", help="Save results JSON to this path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    test_tasks = cfg["test"]
    # Build per-task-entry (not grouped by deg_type) so denoise sigma levels are separate
    from utils.dataset import _DATASET_CLASS, PairedRestorationDataset
    per_task_ds = []
    for t in test_tasks:
        cls = _DATASET_CLASS.get(t['deg_type'], PairedRestorationDataset)
        ds = cls(t, image_size=512, training=False)
        per_task_ds.append(ds)

    device = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"

    if not args.no_lpips:
        import lpips
        net_lpips = lpips.LPIPS(net="vgg").to(device)
        net_lpips.eval()

    print(f"\n{'='*70}")
    header = "PSNR-Y ↑, SSIM-Y ↑, LPIPS ↓" if not args.no_lpips else "PSNR-Y ↑, SSIM-Y ↑"
    print(f"  Baseline: LQ vs GT  ({header})")
    print(f"{'='*70}")

    all_psnr, all_ssim, all_lpips = [], [], []
    results = {}

    for ds in per_task_ds:
        task_psnr, task_ssim, task_lpips = [], [], []
        task_name = ds.name

        for pair in ds.pairs:
            lq = Image.open(pair["lq"]).convert("RGB")
            gt = Image.open(pair["gt"]).convert("RGB")

            # PSNR / SSIM on Y channel (numpy, fast)
            lq_np = np.array(lq).astype(np.uint8)
            gt_np = np.array(gt).astype(np.uint8)
            lq_y = rgb2ycbcr(lq_np, only_y=True)
            gt_y = rgb2ycbcr(gt_np, only_y=True)
            psnr_v = calculate_psnr(gt_y, lq_y, ycbcr=False)
            ssim_v = calculate_ssim(gt_y, lq_y, ycbcr=False)

            # LPIPS (resize to 512 to limit GPU memory)
            if not args.no_lpips:
                lq_t = torch.from_numpy(lq_np.astype(np.float32) / 127.5 - 1.0)
                lq_t = lq_t.permute(2, 0, 1).unsqueeze(0)  # 1CHW
                gt_t = torch.from_numpy(gt_np.astype(np.float32) / 127.5 - 1.0)
                gt_t = gt_t.permute(2, 0, 1).unsqueeze(0)
                # LPIPS on CPU: resize to 256 (fast, metric is roughly scale-invariant)
                _, _, h, w = lq_t.shape
                if max(h, w) > 256:
                    scale = 256 / max(h, w)
                    lq_t = torch.nn.functional.interpolate(lq_t, scale_factor=scale, mode='bilinear', align_corners=False)
                    gt_t = torch.nn.functional.interpolate(gt_t, scale_factor=scale, mode='bilinear', align_corners=False)
                with torch.no_grad():
                    lpips_v = net_lpips(lq_t, gt_t).item()
                task_lpips.append(lpips_v)

            task_psnr.append(psnr_v)
            task_ssim.append(ssim_v)

        mean_p = np.mean(task_psnr)
        mean_s = np.mean(task_ssim)
        r = {"psnr": float(mean_p), "ssim": float(mean_s), "n": len(task_psnr)}
        if task_lpips:
            mean_l = np.mean(task_lpips)
            r["lpips"] = float(mean_l)
            print(f"  {task_name:30s}  PSNR={mean_p:6.2f}  SSIM={mean_s:.4f}  LPIPS={mean_l:.4f}  (n={len(task_psnr)})")
        else:
            print(f"  {task_name:30s}  PSNR={mean_p:6.2f}  SSIM={mean_s:.4f}  (n={len(task_psnr)})")

        results[task_name] = r
        all_psnr.extend(task_psnr)
        all_ssim.extend(task_ssim)
        all_lpips.extend(task_lpips)

    print(f"  {'─'*68}")
    o = {"psnr": float(np.mean(all_psnr)), "ssim": float(np.mean(all_ssim)), "n": len(all_psnr)}
    if all_lpips:
        o["lpips"] = float(np.mean(all_lpips))
        print(f"  {'OVERALL':30s}  PSNR={o['psnr']:6.2f}  SSIM={o['ssim']:.4f}  LPIPS={o['lpips']:.4f}  (n={o['n']})")
    else:
        print(f"  {'OVERALL':30s}  PSNR={o['psnr']:6.2f}  SSIM={o['ssim']:.4f}  (n={o['n']})")
    results["OVERALL"] = o
    print(f"{'='*70}\n")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.json_out}")


if __name__ == "__main__":
    main()
