from typing import Callable, Optional
import os

import torch
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url
import torchvision.transforms as transforms

# ImageNet-R dataset class
# Download code from https://github.com/JH-LEE-KR/ContinualDatasets/blob/main/continual_datasets/continual_datasets.py
# by JH-LEE-KR

class Imagenet_R(ImageFolder):
    def __init__(self, 
                 root             : str, 
                 train            : bool, 
                 transform        : Optional[Callable] = None, 
                 target_transform : Optional[Callable] = None, 
                 download         : bool = False
                 ) -> None:
        
        self.root = os.path.expanduser(root)
        self.url = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
        self.filename = 'imagenet-r.tar'

        try:
            fpath = os.path.join(self.root, self.filename)
            if not os.path.isfile(fpath):
                if not download:
                    raise RuntimeError('Dataset not found. You can use download=True to download it')
                else:
                    print('Downloading from '+ self.url)
                    download_url(self.url, self.root, filename=self.filename)
            if not os.path.exists(os.path.join(self.root, 'imagenet-r')):
                import tarfile
                with tarfile.open(fpath, 'r') as tf:
                    for member in tf.getmembers():
                        try:
                            tf.extract(member, root)
                        except tarfile.error as e:
                            pass

            self.path = self.root + '/imagenet-r/'
            super().__init__(self.path, None, None)
        except:
            print('Dataset failed')
            os.remove(os.path.join(self.root, self.filename))
            os.rmdir(os.path.join(self.root, 'imagenet-r'))

            fpath = os.path.join(self.root, self.filename)
            if not os.path.isfile(fpath):
                if not download:
                    raise RuntimeError('Dataset not found. You can use download=True to download it')
                else:
                    print('Downloading from '+ self.url)
                    download_url(self.url, self.root, filename=self.filename)
            if not os.path.exists(os.path.join(self.root, 'imagenet-r')):
                import tarfile
                with tarfile.open(fpath, 'r') as tf:
                    for member in tf.getmembers():
                        try:
                            tf.extract(member, root)
                        except tarfile.error as e:
                            pass

            self.path = self.root + '/imagenet-r/'
            super().__init__(self.path, None, None)
            raise
        generator = torch.Generator().manual_seed(42)
        len_train = int(len(self.samples) * 0.8)
        len_test = len(self.samples) - len_train
        self.train_sample = torch.randperm(len(self.samples), generator=generator)
        self.test_sample = self.train_sample[len_train:].sort().values.tolist()
        self.train_sample = self.train_sample[:len_train].sort().values.tolist()

        if train:
            self.classes = [i for i in range(200)]
            self.class_to_idx = [i for i in range(200)]
            samples = []
            for idx in self.train_sample:
                samples.append(self.samples[idx])
            self.targets = [s[1] for s in samples]
            self.samples = samples

        else:
            self.classes = [i for i in range(200)]
            self.class_to_idx = [i for i in range(200)]
            samples = []
            for idx in self.test_sample:
                samples.append(self.samples[idx])
            self.targets = [s[1] for s in samples]
            self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int):
        image, label = super().__getitem__(index)
        image = image.convert('L')
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

    def __len__(self) -> int:
        return len(self.samples)