import torch
import os
import pickle
from torch.utils.data import IterableDataset
import cv2
import numpy as np
from copy import deepcopy


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
    """
    Optimized IterableDataset for large-scale image classification.
    
    Key optimizations:
    1. File list caching - avoid rescanning on each startup
    2. Uses DataLoader worker-based sharding (not manual rank/world_size)
    3. Fast cv2 image loading
    4. Zero-copy label tensor (read-only, no clone)
    """

    def __init__(self, dataset_cfg, class_names, transforms=None):
        self.num_classes = len(class_names)
        self.class_to_idx = {cls: i for i, cls in enumerate(class_names)}
        self.transforms = transforms 

        # Parse degradation types
        deg_config = dataset_cfg.get('degradation', [])
        deg_types = [str(deg_config)] if isinstance(deg_config, (str, int)) else [str(d) for d in deg_config]
        
        # Create base label (reused for all samples in this dataset)
        self.base_label = torch.zeros(self.num_classes, dtype=torch.float32)
        for deg in deg_types:
            if deg in self.class_to_idx:
                self.base_label[self.class_to_idx[deg]] = 1.0

        root = os.path.abspath(dataset_cfg['path'])
        
        # Load or scan file list (cached)
        self.img_list = _load_or_scan(root)

    def _load_image(self, img_path):
        """Load image using cv2 (faster than PIL)"""
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            from PIL import Image
            img = Image.open(img_path).convert("RGB")
            img = np.array(img)
        else:
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
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            
            if self.transforms:
                img = self.transforms(img)
            
            yield img, self.base_label


class InterleavedShuffleDataset(IterableDataset):
    """
    Interleaves multiple IterableDatasets round-robin, then applies buffer shuffle.

    Solves two problems with plain ChainDataset:
    1. Cross-dataset mixing: samples from different degradation types are interleaved,
       preventing the model from seeing all-Blur then all-Rain (which causes loss spikes).
    2. Within-shard shuffle: reservoir sampling shuffles the interleaved stream so the
       model never sees the same ordering within an epoch.

    Worker sharding: each sub-dataset's __iter__ reads worker_info and shards itself,
    so no data is duplicated or lost across workers.
    """

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
