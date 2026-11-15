# Wraping for the SHVN dataset

from typing import Callable, Optional

from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import transforms

class MNIST(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__()
        self.dataset = torchvision.datasets.MNIST(root, train, None, None, download)
        self.classes = [str(i) for i in range(10)]
        self.targets = []
        for cls in self.dataset.targets:
            self.targets.append(int(cls))
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image, label = self.dataset.__getitem__(index)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label
        
    def __len__(self):
        return len(self.dataset)