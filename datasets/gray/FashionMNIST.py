# Wraping for the SHVN dataset

from typing import Callable, Optional

from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import transforms
from PIL import Image 

class FashionMNIST(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__()
        self.dataset = torchvision.datasets.FashionMNIST(root, train, None, None, download)
        self.transform = transform
        self.target_transform = target_transform
        self.classes = ["T-shirt/top", "Trouser", "Pullover", 'Dress', "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        self.targets = []
        for cls in self.dataset.targets:
            self.targets.append(int(cls))

    def __getitem__(self, index):
        image, label = self.dataset.__getitem__(index)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
        
    def __len__(self):
        return len(self.dataset)