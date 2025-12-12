from typing import Callable, Optional

import os
import torch
from torch.utils.data import Dataset, random_split
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
import PIL.Image as Image
import numpy as np

# NotMNIST dataset class
# Download code from https://github.com/JH-LEE-KR/ContinualDatasets/blob/main/continual_datasets/continual_datasets.py
# by JH-LEE-KR

class NotMNIST(Dataset):
    def __init__(self, 
                 root             : str, 
                 train            : bool, 
                 transform        : Optional[Callable] = None, 
                 target_transform : Optional[Callable] = None, 
                 download         : bool = False
                 ) -> None:
        super().__init__()

        self.url = 'https://github.com/facebookresearch/Adversarial-Continual-Learning/raw/main/data/notMNIST.zip'
        self.filename = 'notMNIST.zip'

        try:
            fpath = os.path.join(root, self.filename)
            if not os.path.isfile(fpath):
                if not download:
                    raise RuntimeError('Dataset not found. You can use download=True to download it')
                else:
                    print('Downloading from '+self.url)
                    download_url(self.url, root, filename=self.filename)

            if not os.path.exists(os.path.join(root, 'notMNIST')):
                import zipfile
                with zipfile.ZipFile(fpath, 'r') as zf:
                    for member in zf.infolist():
                        try:
                            zf.extract(member, root)
                        except zipfile.error as e:
                            pass
                os.chmod(os.path.join(root, 'notMNIST'), 0o777)
                    
        except:
            print("Dataset failed")
            os.remove(fpath)
            fpath = os.path.join(root, self.filename)
            if not os.path.isfile(fpath):
                if not download:
                    raise RuntimeError('Dataset not found. You can use download=True to download it')
                else:
                    print('Downloading from '+self.url)
                    download_url(self.url, root, filename=self.filename)

            if not os.path.exists(os.path.join(root, 'notMNIST')):
                import zipfile
                with zipfile.ZipFile(fpath, 'r') as zf:
                    for member in zf.infolist():
                        try:
                            zf.extract(member, root)
                        except zipfile.error as e:
                            pass
                os.chmod(os.path.join(root, 'notMNIST'), 0o777)
                
        if self.train:
            fpath = os.path.join(root, 'notMNIST', 'Train')

        else:
            fpath = os.path.join(root, 'notMNIST', 'Test')

        folders = os.listdir(fpath)
        for folder in folders:
            folder_path = os.path.join(fpath, folder)
            for ims in os.listdir(folder_path):
                try:
                    img_path = os.path.join(folder_path, ims)
                    _ = Image.open(img_path)
                except:
                    print("File {}/{} is broken, removing".format(folder, ims))
                    os.remove(img_path)

        self.dataset = ImageFolder(root + '/notMNIST_large/', None, None)
        len_train    = int(len(self.dataset) * 0.8)
        len_val      = len(self.dataset) - len_train
        train, test  = random_split(self.dataset, [len_train, len_val], generator=torch.Generator().manual_seed(42))
        self.dataset = train if train else test
        self.classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        self.targets = []
        self.transform = transform
        self.target_transform = target_transform
        for i in self.dataset.indices:
            self.targets.append(self.dataset.dataset.targets[i])
        pass
    
    def __getitem__(self, index):
        image, label = self.dataset.__getitem__(index)
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.dataset)
