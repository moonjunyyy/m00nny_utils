from typing import Callable, Optional

import os
import torch
from torch.utils.data import Dataset, random_split
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder

# CUB200 dataset class
# Download code from https://github.com/JH-LEE-KR/ContinualDatasets/blob/main/continual_datasets/continual_datasets.py
# by JH-LEE-KR

class CUB200(Dataset):
    def __init__(self, 
                 root             : str, 
                 train            : bool, 
                 transform        : Optional[Callable] = None, 
                 target_transform : Optional[Callable] = None, 
                 download         : bool = False
                 ) -> None:
        super().__init__()

        self.root = os.path.expanduser(root)
        self.url = 'https://data.deepai.org/CUB200(2011).zip'
        self.filename = 'CUB200(2011).zip'

        try:
            fpath = os.path.join(self.root, self.filename)
            if not os.path.exists(os.path.join(self.root, 'CUB_200_2011')):
                if not os.path.isfile(fpath):
                    if not download:
                        raise RuntimeError('Dataset not found. You can use download=True to download it')
                    else:
                        if not download:
                            raise RuntimeError('Dataset not found. You can use download=True to download it')
                        print('Downloading from '+self.url)
                        download_url(self.url, self.root, filename=self.filename)
                if not os.path.exists(os.path.join(root, 'CUB_200_2011')):
                    import zipfile
                    with zipfile.ZipFile(fpath, 'r') as zf:
                        for member in zf.infolist():
                            try:
                                zf.extract(member, root)
                            except zipfile.error as e:
                                pass
                    import tarfile
                    with tarfile.open(os.path.join(root, 'CUB_200_2011.tgz'), 'r') as tf:
                        for member in tf.getmembers():
                            try:
                                tf.extract(member, root)
                            except tarfile.error as e:
                                pass
                for root, dirs, files in os.walk(os.path.join(self.root, 'CUB_200_2011'), topdown=False):
                    for name in files:
                        os.chmod(os.path.join(root, name), 0o777)
                    for name in dirs:
                        os.chmod(os.path.join(root, name), 0o777)
            self.dataset = ImageFolder(self.root + '/CUB_200_2011/images', None, None)
        except:
            print('Dataset failed')
            os.remove(os.path.join(self.root, self.filename))
            for root, dirs, files in os.walk(os.path.join(self.root, 'CUB_200_2011'), topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            fpath = os.path.join(self.root, self.filename)
            if not os.path.isfile(fpath):
                if not download:
                    raise RuntimeError('Dataset not found. You can use download=True to download it')
                else:
                    if not download:
                        raise RuntimeError('Dataset not found. You can use download=True to download it')
                    print('Downloading from '+self.url)
                    download_url(self.url, self.root, filename=self.filename)
            if not os.path.exists(os.path.join(self.root, 'CUB_200_2011')):
                import zipfile
                with zipfile.ZipFile(fpath, 'r') as zf:
                        for member in zf.infolist():
                            try:
                                zf.extract(member, root)
                            except zipfile.error as e:
                                pass
                import tarfile
                with tarfile.open(os.path.join(root, 'CUB_200_2011.tgz'), 'r') as tf:
                    for member in tf.getmembers():
                        try:
                            tf.extract(member, root)
                        except tarfile.error as e:
                            pass
            for root, dirs, files in os.walk(os.path.join(self.root, 'CUB_200_2011'), topdown=False):
                for name in files:
                    os.chmod(os.path.join(root, name), 0o777)
                for name in dirs:
                    os.chmod(os.path.join(root, name), 0o777)
            self.dataset = ImageFolder(self.root + '/CUB_200_2011/images', None, None)
        len_train    = int(len(self.dataset) * 0.8)
        len_val      = len(self.dataset) - len_train
        train, test  = random_split(self.dataset, [len_train, len_val], generator=torch.Generator().manual_seed(42))
        self.dataset = train if train else test
        self.classes = self.dataset.dataset.classes
        self.transform = transform
        self.target_transform = target_transform
        self.targets = []
        for i in self.dataset.indices:
            self.targets.append(self.dataset.dataset.targets[i])
        pass
    
    def __getitem__(self, index):
        image, label = self.dataset.__getitem__(index)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.dataset)
