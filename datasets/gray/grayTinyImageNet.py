from typing import Callable, Optional
import os

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url
import torchvision.transforms as transforms

# TinyImageNet dataset class
# Download code from https://github.com/JH-LEE-KR/ContinualDatasets/blob/main/continual_datasets/continual_datasets.py
# by JH-LEE-KR

class TinyImageNet(ImageFolder):
    def __init__(self, 
                 root             : str, 
                 train            : bool, 
                 transform        : Optional[Callable] = None, 
                 target_transform : Optional[Callable] = None, 
                 download         : bool = False
                 ) -> None:
        
        self.root = os.path.expanduser(root)
        self.url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        self.filename = 'tiny-imagenet-200.zip'

        fpath = os.path.join(self.root, self.filename)
        try:
            if not os.path.isfile(fpath):
                if not download:
                    raise RuntimeError('Dataset not found. You can use download=True to download it')
                else:
                    print('Downloading from '+ self.url)
                    download_url(self.url, self.root, filename=self.filename)
                if not os.path.exists(os.path.join(self.root, 'tiny-imagenet-200')):
                    import zipfile
                    with zipfile.ZipFile(fpath, 'r') as zf:
                            for member in zf.infolist():
                                try:
                                    zf.extract(member, root)
                                except zipfile.error as e:
                                    pass
            self.path = self.root + '/tiny-imagenet-200/'
            if train:
                self.dataset = ImageFolder(self.path + "train", None, None)
            else:
                self.dataset = ImageFolder(self.path + "val", None, None)
        except:
            print("Dataset Failed")
            os.remove(fpath)
            os.rmdir(self.root + '/tiny-imagenet-200')
            if not os.path.isfile(fpath):
                if not download:
                    raise RuntimeError('Dataset not found. You can use download=True to download it')
                else:
                    print('Downloading from '+ self.url)
                    download_url(self.url, self.root, filename=self.filename)
                if not os.path.exists(os.path.join(self.root, 'tiny-imagenet-200')):
                    import zipfile
                    with zipfile.ZipFile(fpath, 'r') as zf:
                            for member in zf.infolist():
                                try:
                                    zf.extract(member, root)
                                except zipfile.error as e:
                                    pass
            self.path = self.root + '/tiny-imagenet-200/'
            if train:
                self.dataset = ImageFolder(self.path + "train", None, None)
            else:
                self.dataset = ImageFolder(self.path + "val", None, None)

        self.transform = transform
        self.target_transform = target_transform

        if train:
            self.classes = []
            with open(self.path + "wnids.txt", 'r') as f:
                for id in f.readlines():
                    self.classes.append(id.split("\n")[0])
            self.class_to_idx = {clss: idx for idx, clss in enumerate(self.classes)}
            self.targets = []
            for idx, (path, label) in enumerate(self.dataset.samples):
                self.targets.append(label)
        else:
            self.classes = []
            with open(self.path + "wnids.txt", 'r') as f:
                for id in f.readlines():
                    self.classes.append(id.split("\n")[0])
            self.class_to_idx = {clss: idx for idx, clss in enumerate(self.classes)}
            self.targets = []
            for idx, (path, label) in enumerate(self.dataset.samples):
                self.targets.append(label)

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