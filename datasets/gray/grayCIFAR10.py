# Wraping for the SHVN dataset

from typing import Callable, Optional

from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import transforms

class CIFAR10(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        self.dataset = torchvision.datasets.CIFAR10(root, train, None, None, download)
        self.transform = transform
        self.target_transform = target_transform
        self.classes = [i for i in self.dataset.classes]
        self.targets = []
        for cls in self.dataset.targets:
            self.targets.append(int(cls))

    def __getitem__(self, index):
        image, label = self.dataset.__getitem__(index)
        image = image.convert('L')
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    def __len__(self):
        return len(self.dataset)