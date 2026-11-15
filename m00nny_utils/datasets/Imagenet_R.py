import os
from .dataset import _MetaDataset, _MetaImageDataset
from ..util.imagefile import Image

# Imagenet-R dataset class
# Download code from https://github.com/JH-LEE-KR/ContinualDatasets/blob/main/continual_datasets/continual_datasets.py
# by JH-LEE-KR

class Imagenet_R(_MetaDataset):
    resources = (
        ('https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar', 'imagenet-r.tar', 'imagenet-r'),
    )

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, transform, target_transform)
        self.train = train
        self._fetch(download)
        self._scan(_MetaImageDataset.extensions, os.path.join(self.root, 'imagenet-r'))
        self._classify()
        self._split(train, ratio=0.8, seed=0, sort=True)
        self._open(Image)
