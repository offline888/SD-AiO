import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class PairedDataset(Dataset):
    def __init__(
        self,
        lq_path: str,
        hq_path: str,
        resolution: int = 1024,
        prompt: str = "",
        dataset_idx: int = 0,
        transforms=None,
        enlarge_ratio: float = 1.0,
    ):
        super().__init__()
        self.resolution = resolution
        self.prompt = prompt
        self.dataset_idx = dataset_idx
        self.transforms = transforms
        self.enlarge_ratio = enlarge_ratio

        self.lq_dir = os.path.abspath(lq_path)
        self.hq_dir = os.path.abspath(hq_path)
        self.custom_instance_prompts = None  # for compatibility with diffusers_flux2.py

        if not os.path.exists(self.lq_dir):
            raise ValueError(f"LQ directory does not exist: {self.lq_dir}")
        if not os.path.exists(self.hq_dir):
            raise ValueError(f"HQ directory does not exist: {self.hq_dir}")

        self.pairs = self._load_pairs()
        self.num_pairs = len(self.pairs)
        self.total_length = int(self.num_pairs * self.enlarge_ratio)

    def _load_pairs(self):
        valid_ext = ('.jpg', '.jpeg', '.png', '.webp')

        hq_files = {
            os.path.splitext(f)[0]: f
            for f in os.listdir(self.hq_dir)
            if f.lower().endswith(valid_ext)
        }
        lq_files = {
            os.path.splitext(f)[0]: f
            for f in os.listdir(self.lq_dir)
            if f.lower().endswith(valid_ext)
        }

        common_keys = sorted(set(hq_files.keys()) & set(lq_files.keys()))
        pairs = [
            (os.path.join(self.hq_dir, hq_files[k]),
             os.path.join(self.lq_dir, lq_files[k]))
            for k in common_keys
        ]
        return pairs

    def _load_image(self, img_path: str) -> np.ndarray:
        try:
            import cv2
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        except Exception:
            img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

        if img.shape[0] != self.resolution or img.shape[1] != self.resolution:
            import cv2
            img = cv2.resize(img, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)

        return img

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        tensor = (img.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(tensor)).float()

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        pair_idx = idx % self.num_pairs
        hq_path, lq_path = self.pairs[pair_idx]

        hq_img = self._load_image(hq_path)
        lq_img = self._load_image(lq_path)

        hq_tensor = self._to_tensor(hq_img)
        lq_tensor = self._to_tensor(lq_img)

        if self.transforms is not None:
            hq_tensor = self.transforms(hq_tensor)
            lq_tensor = self.transforms(lq_tensor)

        return {
            "hq_pixel_values": hq_tensor,
            "lq_pixel_values": lq_tensor,
            "instance_prompt": self.prompt,
            "dataset_idx": self.dataset_idx,
        }
