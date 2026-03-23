import torch
import os
import pickle
import hashlib
from torch.utils.data import IterableDataset
import cv2
import numpy as np
from PIL import Image

_cache_dir = "/tmp/dataset_cache"
os.makedirs(_cache_dir, exist_ok=True)


def _get_cache_path(root):
    """Get cache file path for a dataset root"""
    # Use hashlib for consistent cross-session and cross-platform hash
    hash_value = hashlib.md5(root.encode('utf-8')).hexdigest()
    return os.path.join(_cache_dir, f"{hash_value}.pkl")


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

    def __init__(self, datasets, buffer_size: int = 3000, seed: int = 42):
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

        cache_key = f"paired_{hash(lq_path)}_{hash(hq_path)}"
        cache_path = os.path.join(_cache_dir, f"{cache_key}.pkl")

        self.pairs = None
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    self.pairs = pickle.load(f)
                if self.pairs and os.path.exists(self.pairs[0][0]):
                    print(f"[PairedDataset] Loaded {len(self.pairs)} pairs from cache: {cache_path}")
                else:
                    self.pairs = None
            except Exception as e:
                print(f"[PairedDataset] Cache corrupted ({e}), rescanning...")
                self.pairs = None

        if self.pairs is None:
            hq_files = {os.path.splitext(f)[0]: f for f in os.listdir(self.hq_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))}
            lq_files = {os.path.splitext(f)[0]: f for f in os.listdir(self.lq_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))}
            common_keys = sorted(set(hq_files.keys()) & set(lq_files.keys()))
            self.pairs = [(os.path.join(self.hq_dir, hq_files[k]),
                           os.path.join(self.lq_dir, lq_files[k]))
                          for k in common_keys]
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(self.pairs, f)
                print(f"[PairedDataset] Cached {len(self.pairs)} pairs to {cache_path}")
            except Exception as e:
                print(f"[PairedDataset] Failed to write cache: {e}")

    def _load_and_resize_image(self, img_path: str, resolution: int) -> np.ndarray:
        """
        Load image and resize in one step.
        
        Optimizations:
        - Uses cv2.IMREAD_COLOR for RGB (faster than IMREAD_UNCHANGED)
        - Uses INTER_AREA for downsampling (best quality for shrinking)
        - Direct resize to target resolution
        """
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        else:
            # Fast BGR→RGB conversion
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize using INTER_AREA (best for downsampling)
        img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_AREA)
        return img

    def _numpy_to_tensor(self, img: np.ndarray, to_bfloat16: bool = False) -> torch.Tensor:
        """
        Convert numpy array to tensor with memory optimization.
        
        Args:
            img: HWC numpy array (uint8)
            to_bfloat16: If True, convert to bfloat16 to save GPU memory
        
        Returns:
            CHW tensor in range [-1, 1], float32 or bfloat16
        """
        # Transpose HWC → CHW and normalize to [-1, 1] in one step
        # (img.astype(np.float32) / 127.5 - 1.0) is faster than separate operations
        tensor = (img.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)
        
        # Create tensor from contiguous array
        tensor = torch.from_numpy(np.ascontiguousarray(tensor)).float()
        
        # Convert to bfloat16 if requested (saves 50% memory vs float32)
        if to_bfloat16:
            tensor = tensor.to(torch.bfloat16)
        
        return tensor

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
            # Load and resize both images
            hq_image = self._load_and_resize_image(hq_path, self.resolution)
            lq_image = self._load_and_resize_image(lq_path, self.resolution)

            # Convert to tensors immediately to release numpy memory
            hq_tensor = self._numpy_to_tensor(hq_image)
            del hq_image  # Explicitly release numpy memory
            
            lq_tensor = self._numpy_to_tensor(lq_image)
            del lq_image  # Explicitly release numpy memory

            yield {
                "pixel_values": hq_tensor,
                "conditioning_pixel_values": lq_tensor,
                "captions": self.prompt,
                "dataset_idx": self.dataset_idx,
            }


class PairedDatasetMemoryOpt(PairedDataset):
    """
    Memory-optimized variant of PairedDataset.
    
    Additional optimizations for low-memory scenarios:
    - Converts tensors to bfloat16 (50% memory savings)
    - Uses lower precision for preprocessing
    - Keeps only one image in memory at a time during iteration
    """
    
    def __init__(self, lq_path: str, hq_path: str,
                 resolution: int = 1024,
                 prompt: str = "",
                 dataset_idx: int = 0,
                 transforms=None,
                 use_bfloat16: bool = True):
        
        super().__init__(lq_path, hq_path, resolution, prompt, dataset_idx, transforms)
        self.use_bfloat16 = use_bfloat16

    def _numpy_to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """
        Convert with bfloat16 optimization for GPU memory savings.
        
        bfloat16 uses same memory as float16 but has better numerical stability
        for deep learning training (same exponent bits as float32).
        """
        # Fast normalization to [-1, 1]
        tensor = (img.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)
        tensor = torch.from_numpy(np.ascontiguousarray(tensor)).float()
        
        # Convert to bfloat16 if requested (50% memory savings)
        if self.use_bfloat16:
            tensor = tensor.to(torch.bfloat16)
        
        return tensor

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
            # Load and resize both images
            hq_image = self._load_and_resize_image(hq_path, self.resolution)
            lq_image = self._load_and_resize_image(lq_path, self.resolution)

            # Convert to bfloat16 tensors immediately
            hq_tensor = self._numpy_to_tensor(hq_image)
            lq_tensor = self._numpy_to_tensor(lq_image)
            
            # Release numpy arrays immediately
            del hq_image, lq_image

            yield {
                "pixel_values": hq_tensor,
                "conditioning_pixel_values": lq_tensor,
                "captions": self.prompt,
                "dataset_idx": self.dataset_idx,
            }
