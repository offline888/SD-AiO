import os
import random
import re
import zlib
import yaml
from pathlib import Path
from collections import OrderedDict

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler
import torchvision.transforms.functional as TF

_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}

def scan_images(directory):
    directory = Path(directory)
    if not directory.exists():
        return {}
    return {e.stem: e.suffix for e in directory.iterdir()
            if e.is_file() and e.suffix.lower() in _IMAGE_EXTS}

def is_denoise_task(task_config):
    return (task_config.get('deg_type') == 'noise'
            and task_config.get('lq_path') == task_config.get('gt_path'))

def add_noise_tensor(lq_tensor, sigma, seed=None):
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
        noise = torch.randn_like(lq_tensor, generator=generator) * (sigma / 127.5)
    else:
        noise = torch.randn_like(lq_tensor) * (sigma / 127.5)
    return (lq_tensor + noise).clamp(-1, 1)


class PairedTransform:
    """
    Paired LQ/GT augmentation — identical transforms on both images.

    Training crop (always outputs target_size × target_size square):
      - short edge < target_size → resize short edge to target_size → random crop
      - short edge >= target_size → direct random crop from original

    Optional geometric augmentations from YAML aug_config:
      hflip_prob (default 0.5), vflip_prob (default 0.0), rot90_prob (default 0.0)
    """
    def __init__(self, aug_config, target_size, is_train=True, full_image_eval=False):
        self.is_train = is_train
        self.target_size = target_size
        self.full_image_eval = full_image_eval
        cfg = aug_config or {}
        self.hflip_prob = cfg.get('hflip_prob', 0.5)
        self.vflip_prob = cfg.get('vflip_prob', 0.0)
        self.rot90_prob = cfg.get('rot90_prob', 0.0)

    def __call__(self, lq_img, gt_img):
        if not self.is_train and self.full_image_eval:
            # Eval on full image: no crop, no aug, just tensor conversion
            return TF.to_tensor(lq_img) * 2.0 - 1.0, TF.to_tensor(gt_img) * 2.0 - 1.0
        lq_img, gt_img = self._crop_square(lq_img, gt_img)
        if self.is_train:
            if random.random() < self.hflip_prob:
                lq_img, gt_img = TF.hflip(lq_img), TF.hflip(gt_img)
            if random.random() < self.vflip_prob:
                lq_img, gt_img = TF.vflip(lq_img), TF.vflip(gt_img)
            if random.random() < self.rot90_prob:
                angle = random.choice([90, -90])
                lq_img, gt_img = TF.rotate(lq_img, angle), TF.rotate(gt_img, angle)
        return TF.to_tensor(lq_img) * 2.0 - 1.0, TF.to_tensor(gt_img) * 2.0 - 1.0

    def _crop_square(self, lq_img, gt_img):
        w, h = gt_img.size
        t = self.target_size

        if min(h, w) < t:
            scale = t / min(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            lq_img = TF.resize(lq_img, (new_h, new_w), TF.InterpolationMode.BICUBIC)
            gt_img = TF.resize(gt_img, (new_h, new_w), TF.InterpolationMode.BICUBIC)
            h, w = new_h, new_w

        if not self.is_train:
            top = (h - t) // 2
            left = (w - t) // 2
        else:
            top = random.randint(0, max(0, h - t))
            left = random.randint(0, max(0, w - t))

        return TF.crop(lq_img, top, left, t, t), TF.crop(gt_img, top, left, t, t)

class SingleTaskDataset(Dataset):
    def __init__(self, task_config, transform):
        super().__init__()
        self.task_name = task_config['name']
        self.deg_type = task_config['deg_type']
        self.prompt = task_config['prompt']
        self.transform = transform
        self.repeat_ratio = task_config.get('repeat_ratio', 1)

        self.is_denoise = is_denoise_task(task_config)
        self.sigma = 0
        if self.is_denoise:
            m = re.search(r'_(\d+)$', self.task_name)
            self.sigma = int(m.group(1)) if m else 0

        self.lq_paths = self._list_images(task_config['lq_path'])
        self.gt_paths = self._list_images(task_config['gt_path'])

        self.prefix_len = task_config.get('match_prefix_len', 0)
        if self.prefix_len:
            # Prefix pairing (OTS_BETA-style: 72k haze variants matched to 2k clear by prefix)
            self.gt_by_prefix = {p.stem[:self.prefix_len]: p for p in self.gt_paths}
            for lp in self.lq_paths:
                assert lp.stem[:self.prefix_len] in self.gt_by_prefix, \
                    f"[{self.task_name}] No GT prefix match for {lp.name}"
        else:
            assert len(self.lq_paths) == len(self.gt_paths), \
                f"[{self.task_name}] LQ ({len(self.lq_paths)}) != GT ({len(self.gt_paths)})"
            self.gt_by_prefix = None

        self.indices = list(range(len(self.lq_paths))) * self.repeat_ratio

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        try:
            lq_img = Image.open(self.lq_paths[real_idx]).convert('RGB')
            if self.gt_by_prefix:
                prefix = self.lq_paths[real_idx].stem[:self.prefix_len]
                gt_img = Image.open(self.gt_by_prefix[prefix]).convert('RGB')
            else:
                gt_img = Image.open(self.gt_paths[real_idx]).convert('RGB')
        except Exception as e:
            print(f"Warning: failed to read {self.lq_paths[real_idx]}: {e}. Retrying random sample.")
            return self.__getitem__(random.randint(0, len(self) - 1))

        lq_tensor, gt_tensor = self.transform(lq_img, gt_img)

        if self.is_denoise and self.sigma > 0:
            if not self.transform.is_train:
                seed = zlib.crc32(str(real_idx).encode()) & 0x7fffffff
                lq_tensor = add_noise_tensor(lq_tensor, self.sigma, seed=seed)
            else:
                lq_tensor = add_noise_tensor(lq_tensor, self.sigma)

        return {
            'conditioning_pixel_values': lq_tensor,
            'output_pixel_values': gt_tensor,
            'prompt': self.prompt,
            'task_name': self.task_name,
            'deg_type': self.deg_type,
            'online_noise': self.sigma if self.is_denoise else 0,
        }

    @staticmethod
    def _list_images(dir_path):
        d = Path(dir_path)
        if not d.exists():
            raise FileNotFoundError(f"Image directory not found: {d}")
        return sorted(p for p in d.iterdir()
                     if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)

def build_deg_types(task_entries):
    seen, types = set(), []
    for task in task_entries:
        dt = task['deg_type']
        if dt not in seen:
            seen.add(dt)
            types.append(dt)
    return types

def deg_to_label(deg_type: str, deg_types: list, num_deg_types: int) -> torch.Tensor:
    label = torch.zeros(num_deg_types)
    if deg_type in deg_types[:num_deg_types]:
        label[deg_types.index(deg_type)] = 1.0  # type: ignore
    return label

class ClassificationDataset(Dataset):
    def __init__(self, task_entries, num_deg_types, image_size=256, training=True):
        self.image_size = image_size
        self.training = training
        self.num_deg_types = num_deg_types
        self.samples = []

        deg_types = build_deg_types(task_entries)
        print(f"  Degradation types: {deg_types[:num_deg_types]}")

        for task in task_entries:
            lq_map = scan_images(task['lq_path'])
            label = deg_to_label(task['deg_type'], deg_types, num_deg_types)
            repeat = int(task.get('repeat_ratio', 1))
            for stem in sorted(lq_map):
                for _ in range(repeat):
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
        t = self.image_size

        if min(h, w) < t:
            scale = t / min(h, w)
            img = TF.resize(img, (int(h * scale), int(w * scale)))
            h, w = img.size[1], img.size[0]

        if not self.training:
            top, left = (h - t) // 2, (w - t) // 2
        else:
            top = random.randint(0, max(0, h - t))
            left = random.randint(0, max(0, w - t))

        img = TF.crop(img, top, left, t, t)
        if self.training and random.random() < 0.5:
            img = TF.hflip(img)

        return {
            'lq': TF.to_tensor(img) * 2.0 - 1.0,
            'label': s['label'],
        }

def collate_fn(batch):
    return {
        'conditioning_pixel_values': torch.stack([s['conditioning_pixel_values'] for s in batch]),
        'output_pixel_values': torch.stack([s['output_pixel_values'] for s in batch]),
        'prompt': [s['prompt'] for s in batch],
        'task_name': [s['task_name'] for s in batch],
        'deg_type': [s['deg_type'] for s in batch],
        'online_noise': [s['online_noise'] for s in batch],
    }

class DegTypeRoundRobinSampler(Sampler):
    """Round-robin across degradation-type groups. With batch_size == num_groups,
    each batch contains exactly one sample from each degradation type."""

    def __init__(self, group_boundaries):
        self.group_boundaries = group_boundaries  # [(0, N1), (N1, N1+N2), ...]
        self.num_groups = len(group_boundaries)
        self.group_sizes = [end - start for start, end in group_boundaries]

    def __iter__(self):
        per_group = []
        for start, end in self.group_boundaries:
            indices = list(range(start, end))
            random.shuffle(indices)
            per_group.append(indices)

        batch_count = min(self.group_sizes)
        batches = []
        for slot in range(batch_count):
            for g in range(self.num_groups):
                batches.append(per_group[g][slot])
        return iter(batches)

    def __len__(self):
        return min(self.group_sizes) * self.num_groups


def build_dataloaders(yaml_path, train_batch_size=None, eval_batch_size=1,
                      num_workers=None, train_image_size=None, test_image_size=None,
                      full_image_eval=False, round_robin=False):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    global_cfg = config.get('global_config', {})
    aug_cfg = global_cfg.get('augmentation', {})
    dl_cfg = global_cfg.get('dataloader', {})
    rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', 0)))
    is_main = (rank == 0)

    train_loader = None
    if 'train' in config:
        tsize = train_image_size if train_image_size else global_cfg.get('train_image_size', 256)
        train_transform = PairedTransform(aug_config=aug_cfg, target_size=tsize, is_train=True)

        if round_robin:
            # Group by deg_type for round-robin sampling
            tasks_by_deg = OrderedDict()
            for task in config['train']:
                dt = task['deg_type']
                tasks_by_deg.setdefault(dt, []).append(task)

            train_datasets = []
            group_boundaries = []
            offset = 0
            for dt, tasks in tasks_by_deg.items():
                start = offset
                for task in tasks:
                    ds = SingleTaskDataset(task, train_transform)
                    train_datasets.append(ds)
                    offset += len(ds)
                group_boundaries.append((start, offset))
                if is_main:
                    names = ', '.join(t['name'] for t in tasks)
                    print(f"Train group [{dt}]: {names} | size={offset - start}")

            unified = ConcatDataset(train_datasets)
            bs = train_batch_size if train_batch_size is not None else dl_cfg.get('batch_size', 4)
            nw = num_workers if num_workers is not None else dl_cfg.get('num_workers', 4)
            sampler = DegTypeRoundRobinSampler(group_boundaries)
            train_loader = DataLoader(
                unified,
                batch_size=bs,
                sampler=sampler,
                num_workers=nw,
                pin_memory=dl_cfg.get('pin_memory', True),
                collate_fn=collate_fn,
                drop_last=True,
                persistent_workers=(nw > 0),
            )
        else:
            train_datasets = []
            for task in config['train']:
                ds = SingleTaskDataset(task, train_transform)
                train_datasets.append(ds)
                if is_main:
                    print(f"Train task: {task['name']} | size={len(ds)} (repeat={task.get('repeat_ratio', 1)})")

            unified = ConcatDataset(train_datasets)
            bs = train_batch_size if train_batch_size is not None else dl_cfg.get('batch_size', 4)
            nw = num_workers if num_workers is not None else dl_cfg.get('num_workers', 4)
            train_loader = DataLoader(
                unified,
                batch_size=bs,
                shuffle=True,
                num_workers=nw,
                pin_memory=dl_cfg.get('pin_memory', True),
                collate_fn=collate_fn,
                drop_last=True,
                persistent_workers=(nw > 0),
            )

    test_loaders = OrderedDict()
    if 'test' in config:
        esize = test_image_size if test_image_size else global_cfg.get('test_image_size', 512)
        test_transform = PairedTransform(aug_config=None, target_size=esize, is_train=False,
                                         full_image_eval=full_image_eval)
        nw = num_workers if num_workers is not None else dl_cfg.get('num_workers', 2)
        for task in config['test']:
            ds = SingleTaskDataset(task, test_transform)
            loader = DataLoader(
                ds,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=nw,
                collate_fn=collate_fn,
                persistent_workers=(nw > 0),
            )
            test_loaders[task['name']] = loader
            if is_main:
                print(f"Test task: {task['name']} | size={len(ds)}")

    return train_loader, test_loaders

def build_unified_train_dataset(train_tasks, image_size=256):
    transform = PairedTransform(aug_config={}, target_size=image_size, is_train=True)
    datasets = [SingleTaskDataset(t, transform) for t in train_tasks]
    return ConcatDataset(datasets)

def build_datasets(val_tasks, image_size=512, training=False):
    transform = PairedTransform(aug_config=None, target_size=image_size, is_train=training)
    return OrderedDict((t['name'], SingleTaskDataset(t, transform)) for t in val_tasks)
