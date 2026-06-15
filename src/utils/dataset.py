"""Paired LQ-GT dataset for Stage 2 restoration training.
Expects task entries with lq_path and gt_path pointing to directories.
"""

import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}


def _scan_dir(directory):
    d = Path(directory)
    if not d.exists():
        return {}
    return {e.stem: e.suffix for e in d.iterdir()
            if e.is_file() and e.suffix.lower() in _IMAGE_EXTS}


class PairedRestorationDataset(Dataset):

    def __init__(self, task_entries, image_size=512, training=True):
        self.image_size = image_size
        self.training = training
        self.pairs = []

        for task in task_entries:
            lq_map = _scan_dir(task['lq_path'])
            gt_map = _scan_dir(task['gt_path'])
            common = set(lq_map) & set(gt_map)

            for stem in sorted(common):
                self.pairs.append({
                    'lq_path': str(Path(task['lq_path']) / f"{stem}{lq_map[stem]}"),
                    'gt_path': str(Path(task['gt_path']) / f"{stem}{gt_map[stem]}"),
                    'task': task['name'],
                })

            status = f"{len(common)} pairs" if common else "WARNING: no pairs"
            print(f"  [{task['name']}] {status}  lq={task['lq_path']}  gt={task['gt_path']}")

        print(f"Total: {len(self.pairs)} paired images")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        lq = Image.open(pair['lq_path']).convert('RGB')
        gt = Image.open(pair['gt_path']).convert('RGB')

        if self.training:
            if random.random() < 0.5:
                lq = TF.hflip(lq)
                gt = TF.hflip(gt)
            lq, gt = self._random_crop(lq, gt)
        else:
            lq, gt = self._center_crop(lq, gt)

        return {
            'lq': TF.to_tensor(lq) * 2.0 - 1.0,
            'gt': TF.to_tensor(gt) * 2.0 - 1.0,
            'task': pair['task'],
        }

    def _random_crop(self, lq, gt):
        w, h = lq.size
        if min(h, w) < self.image_size:
            scale = self.image_size / min(h, w)
            nh, nw = int(h * scale), int(w * scale)
            lq = TF.resize(lq, (nh, nw))
            gt = TF.resize(gt, (nh, nw))
            h, w = nh, nw
        top = random.randint(0, max(0, h - self.image_size))
        left = random.randint(0, max(0, w - self.image_size))
        lq = TF.crop(lq, top, left, self.image_size, self.image_size)
        gt = TF.crop(gt, top, left, self.image_size, self.image_size)
        return lq, gt

    def _center_crop(self, lq, gt):
        w, h = lq.size
        if h < self.image_size or w < self.image_size:
            lq = TF.resize(lq, (self.image_size, self.image_size))
            gt = TF.resize(gt, (self.image_size, self.image_size))
        else:
            top = (h - self.image_size) // 2
            left = (w - self.image_size) // 2
            lq = TF.crop(lq, top, left, self.image_size, self.image_size)
            gt = TF.crop(gt, top, left, self.image_size, self.image_size)
        return lq, gt
