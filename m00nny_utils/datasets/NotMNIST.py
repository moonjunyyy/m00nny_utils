import os
import torch
import numpy as np
from PIL import Image
from typing import Any, Callable, Optional, Tuple, TypeVar
PathLike = TypeVar('PathLike', str, bytes, os.PathLike)
ImageLike = TypeVar('ImageLike', Image.Image, torch.Tensor, np.ndarray)

from torch.utils.data import Dataset, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

class NotMNIST(Dataset):
    def __init__(self,
                 root : PathLike,
                 train : bool=True,
                 transform : Callable[[ImageLike],torch.Tensor]|None=None,
                 target_transform : Callable[[ImageLike],torch.Tensor]|None=None,
                 download : bool=False
                 ) -> None:
        
        super().__init__()
        self.dataset = ImageFolder(root + '/notMNIST_large/', transforms.ToTensor() if transform is None else transform, target_transform)
        len_train    = int(len(self.dataset) * 0.8)
        len_val      = len(self.dataset) - len_train
        train, test  = random_split(self.dataset, [len_train, len_val], generator=torch.Generator().manual_seed(42))
        self.dataset = train if train else test
        self.classes = self.dataset.dataset.classes
        self.targets = []
        for i in self.dataset.indices:
            self.targets.append(self.dataset.dataset.targets[i])
        pass
    
    def __getitem__(self, index):
        image, label = self.dataset.__getitem__(index)
        return image.expand(3,-1,-1), label

    def __len__(self):
        return len(self.dataset)
