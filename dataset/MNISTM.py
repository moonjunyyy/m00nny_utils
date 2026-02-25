import os
import os.path
import torch
import numpy as np
from PIL import Image
from typing import Any, Callable, Optional, Tuple, TypeVar
PathLike = TypeVar('PathLike', str, bytes, os.PathLike)
ImageLike = TypeVar('ImageLike', Image.Image, torch.Tensor, np.ndarray)

from torchvision.datasets.utils import download_and_extract_archive

class MNISTM(torch.utils.data.Dataset):
    resources = [
        ('https://github.com/liyxi/mnist-m/releases/download/data/mnist_m_train.pt.tar.gz',
         '191ed53db9933bd85cc9700558847391'),
        ('https://github.com/liyxi/mnist-m/releases/download/data/mnist_m_test.pt.tar.gz',
         'e11cb4d7fff76d7ec588b1134907db59')
    ]

    training_file = "mnist_m_train.pt"
    test_file = "mnist_m_test.pt"

    def __init__(self,
                 root : PathLike,
                 train : bool=True,
                 transform : Callable[[ImageLike],torch.Tensor]|None=None,
                 target_transform : Callable[[ImageLike],torch.Tensor]|None=None,
                 download : bool=False
                 ) -> None:
        root = os.path.join(root, 'MNIST-M')   
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        if self.train: data_file = self.training_file
        else: data_file = self.test_file
        self.classes = [i for i in range(10)]
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.squeeze().numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder, self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder, self.test_file)))

    def download(self):
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder,
                                         extract_root=self.processed_folder,
                                         filename=filename, md5=md5)
