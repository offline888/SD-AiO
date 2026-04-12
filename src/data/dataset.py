import os
from typing import Callable

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class PairedDataset(Dataset):
    def __init__(
        self,
        lq_path: str,
        hq_path: str,
        resolution: int = 512,  
        prompt: str = "",
        dataset_idx: int = 0,
        transforms: Callable = None,
        enlarge_ratio: float = 1.0,
        deg_type: str = None,
    ):
        super().__init__()
        self.prompt = prompt
        self.resolution = resolution
        self.dataset_idx = dataset_idx
        self.enlarge_ratio = enlarge_ratio
        self.deg_type = deg_type if deg_type is not None else None

        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((resolution, resolution)),  
                # PIL [0,255] → Tensor [0,1]
                transforms.ToTensor(),
                # [0,1] → [-1,1]
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
            ])

        self.lq_dir = os.path.abspath(lq_path)
        self.hq_dir = os.path.abspath(hq_path)

        assert os.path.exists(self.lq_dir), f"LQ dir not found: {self.lq_dir}"
        assert os.path.exists(self.hq_dir), f"HQ dir not found: {self.hq_dir}"

        self.pairs = self._load_pairs()
        self.num_pairs = len(self.pairs)
        self.total_num_pairs = int(self.num_pairs * self.enlarge_ratio)

    def _load_pairs(self):
        valid_ext = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
        
        def get_files(directory):
            return {
                os.path.splitext(f)[0]: f
                for f in os.listdir(directory)
                if f.lower().endswith(valid_ext)
            }
        
        hq_files = get_files(self.hq_dir)
        lq_files = get_files(self.lq_dir)
        
        common = sorted(set(hq_files) & set(lq_files))
        return [
            (os.path.join(self.hq_dir, hq_files[k]),
             os.path.join(self.lq_dir, lq_files[k]))
            for k in common
        ]

    def _load_image(self, path: str) -> Image.Image:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img)

        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load: {path}") from e

    def __len__(self):
        return self.total_num_pairs

    def __getitem__(self, idx: int):
        pair_idx = idx % self.num_pairs
        hq_path, lq_path = self.pairs[pair_idx]

        hq_img = self._load_image(hq_path)  # PIL
        lq_img = self._load_image(lq_path)  

        hq_tensor = self.transforms(hq_img)
        lq_tensor = self.transforms(lq_img)

        return {
            "hq_pixel_values": hq_tensor,  # [3, H, W]
            "lq_pixel_values": lq_tensor,
            "prompt": self.prompt,
            "dataset_idx": self.dataset_idx,
            "deg_type": self.deg_type,
        }