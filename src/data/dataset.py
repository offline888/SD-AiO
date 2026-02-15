import torch, os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class MultiLabelClassification(Dataset):
    def __init__(self, dataset_cfg, class_names, transforms=None):
        self.data_root = os.path.abspath(dataset_cfg['path'])
        self.transforms = transforms 
        
        self.class_to_idx = {cls: i for i, cls in enumerate(class_names)}
        self.num_classes = len(class_names)
        
        deg_config = dataset_cfg['degradation']
        deg_types = [str(deg_config)] if isinstance(deg_config, str) else [str(d) for d in deg_config]
        self.fixed_label = torch.zeros(self.num_classes, dtype=torch.float32)
        for deg in deg_types:
            if deg in self.class_to_idx:
                self.fixed_label[self.class_to_idx[deg]] = 1.0

        valid_extensions = ('.jpg', '.png', '.jpeg', '.bmp', '.tiff', '.JPG', '.PNG', '.JPEG', '.BMP', '.TIFF')
        self.img_list = []
        for root, _, files in os.walk(self.data_root):
            for file in files:
                if file.endswith(valid_extensions):
                    self.img_list.append(os.path.join(root, file))
        self.img_list.sort()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.img_list[idx]).convert("RGB")
            if self.transforms:
                image = self.transforms(image)
            return image, self.fixed_label.clone()
        except Exception:
            return self.__getitem__((idx + 1) % len(self))