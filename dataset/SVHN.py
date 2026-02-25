from typing import Any, Callable, Optional, Tuple, TypeVar

import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torchvision.datasets import SVHN
from torchvision.transforms import transforms

from PIL import Image

PathLike = TypeVar('PathLike', str, bytes, os.PathLike)
ImageLike = TypeVar('ImageLike', Image.Image, torch.Tensor, np.ndarray)

class SVHN(Dataset):
    def ___init__(self,
                 root : PathLike,
                 train : bool=True,
                 transform : Callable[[ImageLike],torch.Tensor]|None=None,
                 target_transform : Callable[[ImageLike],torch.Tensor]|None=None,
                 download : bool=False
                 ) -> None:

        super().__init__()
        self.dataset = SVHN(root, "train" if train else "test", transforms.ToTensor() if transform is None else transform, target_transform, download)
        
        self.classes = [str(i) for i in range(10)]
        self.targets = []
        for cls in self.dataset.labels:
            self.targets.append(int(cls))

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)

    def __len__(self):
        return len(self.dataset)