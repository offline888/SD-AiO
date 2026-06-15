"""Classification dataset for Stage 1 — multi-label degradation classifier.
Expects task entries with lq_path pointing to a directory of images,
and deg_type mapping to the canonical degradation index.
"""

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

# Degradation type → label index.  Order must match num_deg_types used in training.
DEG_TYPES = ["rain", "haze", "noise"]


def _scan_dir(directory):
    d = Path(directory)
    if not d.exists():
        return {}
    return {e.stem: e.suffix for e in d.iterdir()
            if e.is_file() and e.suffix.lower() in _IMAGE_EXTS}


def deg_to_label(deg_type: str, num_deg_types: int) -> torch.Tensor:
    """Convert a deg_type string to a one-hot label of length num_deg_types."""
    label = torch.zeros(num_deg_types)
    if deg_type in DEG_TYPES[:num_deg_types]:
        label[DEG_TYPES.index(deg_type)] = 1.0
    return label


class ClassificationDataset(Dataset):
    """Returns (lq_image, label) for classifier training."""

    def __init__(self, task_entries, num_deg_types, image_size=256, training=True):
        self.image_size = image_size
        self.training = training
        self.num_deg_types = num_deg_types
        self.samples = []

        for task in task_entries:
            lq_map = _scan_dir(task['lq_path'])
            label = deg_to_label(task['deg_type'], num_deg_types)

            for stem in sorted(lq_map):
                self.samples.append({
                    'path': str(Path(task['lq_path']) / f"{stem}{lq_map[stem]}"),
                    'label': label,
                })
            print(f"  [{task['name']}] {len(lq_map)} images  deg={task['deg_type']}  label={label.tolist()}")

        print(f"Total: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s['path']).convert('RGB')
        w, h = img.size

        if self.training:
            if min(h, w) < self.image_size:
                scale = self.image_size / min(h, w)
                img = TF.resize(img, (int(h * scale), int(w * scale)))
                h, w = img.size[1], img.size[0]
            top = random.randint(0, max(0, h - self.image_size))
            left = random.randint(0, max(0, w - self.image_size))
            img = TF.crop(img, top, left, self.image_size, self.image_size)
            if random.random() < 0.5:
                img = TF.hflip(img)
        else:
            if h < self.image_size or w < self.image_size:
                img = TF.resize(img, (self.image_size, self.image_size))
            else:
                top = (h - self.image_size) // 2
                left = (w - self.image_size) // 2
                img = TF.crop(img, top, left, self.image_size, self.image_size)

        return {
            'lq': TF.to_tensor(img) * 2.0 - 1.0,
            'label': s['label'],
        }
