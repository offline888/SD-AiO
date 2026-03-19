import torch
import os
import pickle
from torch.utils.data import IterableDataset
import cv2
import numpy as np
from PIL import Image

_cache_dir = "/tmp/dataset_cache"
os.makedirs(_cache_dir, exist_ok=True)


def _get_cache_path(root):
    """Get cache file path for a dataset root"""
    return os.path.join(_cache_dir, f"{hash(root)}.pkl")


def _scan_directory(root):
    """Scan directory and return sorted file list"""
    valid_extensions = ('.jpg', '.jpeg', '.png')
    img_list = []
    
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(valid_extensions):
                img_list.append(os.path.join(r, f))
    
    img_list.sort()
    return img_list


def _load_or_scan(root):
    """Load from cache or scan directory"""
    cache_path = _get_cache_path(root)
    
    # Try to load from cache first (fast startup)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            # Verify cache is valid
            if cached and os.path.exists(cached[0] if cached else ""):
                return cached
        except:
            pass
    
    # Scan directory if no cache
    img_list = _scan_directory(root)
    
    # Save to cache for next time
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(img_list, f)
    except:
        pass
    
    return img_list


class MultiLabelClassification(IterableDataset):

    def __init__(self, dataset_cfg, class_names, transforms=None):
        self.num_classes = len(class_names)
        self.class_to_idx = {cls: i for i, cls in enumerate(class_names)}
        self.transforms = transforms 

        # Parse degradation types
        deg_config = dataset_cfg.get('degradation', [])
        deg_types = [str(deg_config)] if isinstance(deg_config, (str, int)) else [str(d) for d in deg_config]
        
        # Create base label (reused for all samples in this dataset)
        # Shape: [num_classes, 2] - each row is [exists, not_exists]
        # Exists: [1, 0], Not exists: [0, 1]
        self.base_label = torch.zeros(self.num_classes, 2, dtype=torch.float32)
        for deg in deg_types:
            if deg in self.class_to_idx:
                i = self.class_to_idx[deg]
                self.base_label[i, 0] = 1.0   # exists: [1, 0]
        
        # Fill "not exists" column for classes that are not in deg_types
        for i in range(self.num_classes):
            if self.base_label[i, 0] == 0:
                self.base_label[i, 1] = 1.0   # not exists: [0, 1]

        root = os.path.abspath(dataset_cfg['path'])
        
        # Load or scan file list (cached)
        self.img_list = _load_or_scan(root)

    def _load_image(self, img_path):
        """Load image using cv2 with optimization"""
        # Use IMREAD_UNCHANGED + faster BGR→RGB conversion
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            # Fallback to PIL for unusual formats
            from PIL import Image
            img = Image.open(img_path).convert("RGB")
            img = np.array(img, dtype=np.uint8)
        else:
            # Fast in-place BGR→RGB conversion using cv2.cvtColor
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __len__(self):
        return len(self.img_list)

    def __iter__(self):
        """
        DataLoader will handle worker-based sharding automatically.
        Each worker gets a different slice of img_list.
        """
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # Single worker mode
            img_iter = iter(self.img_list)
        else:
            # Multiple workers - DataLoader handles sharding
            per_worker = int(np.ceil(len(self.img_list) / worker_info.num_workers))
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(self.img_list))
            img_iter = iter(self.img_list[start:end])
        
        for img_path in img_iter:
            img = self._load_image(img_path)
            # Use ascontiguousarray + permute in one go, then numpy directly
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            img = torch.from_numpy(img).float()
            
            if self.transforms:
                img = self.transforms(img)
            
            yield img, self.base_label


class InterleavedShuffleDataset(IterableDataset):

    def __init__(self, datasets, buffer_size: int = 2000, seed: int = 42):
        self.datasets = datasets
        self.buffer_size = buffer_size
        self.seed = seed

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        rng = np.random.default_rng(seed=self.seed + worker_id)

        def _interleave():
            """Round-robin across all sub-datasets until all are exhausted."""
            iters = [iter(d) for d in self.datasets]
            exhausted = [False] * len(iters)
            while not all(exhausted):
                for i, it in enumerate(iters):
                    if exhausted[i]:
                        continue
                    try:
                        yield next(it)
                    except StopIteration:
                        exhausted[i] = True

        # Reservoir / buffer shuffle over the interleaved stream
        buffer = []
        for item in _interleave():
            if len(buffer) < self.buffer_size:
                buffer.append(item)
            else:
                idx = int(rng.integers(0, self.buffer_size))
                yield buffer[idx]
                buffer[idx] = item

        # Flush remaining buffer in shuffled order
        indices = np.arange(len(buffer))
        rng.shuffle(indices)
        for i in indices:
            yield buffer[i]

class PairedDataset(IterableDataset):
    
    def __init__(self, lq_path: str, hq_path: str,
                 resolution: int = 1024,
                 prompt: str = "",
                 dataset_idx: int = 0,
                 transforms=None):

        super().__init__()
        self.resolution = resolution
        self.prompt = prompt
        self.dataset_idx = dataset_idx  # 用于预计算 prompt embedding 缓存的索引
        self.transforms = transforms

        lq_path = os.path.abspath(lq_path)
        hq_path = os.path.abspath(hq_path)

        self.lq_dir = lq_path
        self.hq_dir = hq_path

        if not os.path.exists(self.lq_dir):
            raise ValueError(f"LQ directory does not exist: {self.lq_dir}")
        if not os.path.exists(self.hq_dir):
            raise ValueError(f"HQ directory does not exist: {self.hq_dir}")

        # Scan and match images
        hq_files = {os.path.splitext(f)[0]: f for f in os.listdir(self.hq_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))}
        lq_files = {os.path.splitext(f)[0]: f for f in os.listdir(self.lq_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))}

        # Find common files
        common_keys = sorted(set(hq_files.keys()) & set(lq_files.keys()))

        self.pairs = [(os.path.join(self.hq_dir, hq_files[k]),
                       os.path.join(self.lq_dir, lq_files[k]))
                      for k in common_keys]

    def _load_image(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            img = Image.open(img_path).convert("RGB")
            img = np.array(img, dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __len__(self):
        return len(self.pairs)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            pairs_iter = iter(self.pairs)
        else:
            per_worker = int(np.ceil(len(self.pairs) / worker_info.num_workers))
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(self.pairs))
            pairs_iter = iter(self.pairs[start:end])

        for hq_path, lq_path in pairs_iter:
            hq_image = self._load_image(hq_path)
            lq_image = self._load_image(lq_path)

            # Resize
            hq_image = cv2.resize(hq_image, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)
            lq_image = cv2.resize(lq_image, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)

            # To tensor and normalize
            hq_tensor = torch.from_numpy(hq_image.transpose(2, 0, 1)).float() / 255.0 * 2.0 - 1.0
            lq_tensor = torch.from_numpy(lq_image.transpose(2, 0, 1)).float() / 255.0 * 2.0 - 1.0

            yield {
                "pixel_values": hq_tensor,
                "conditioning_pixel_values": lq_tensor,
                "captions": self.prompt,
                "dataset_idx": self.dataset_idx,
            }