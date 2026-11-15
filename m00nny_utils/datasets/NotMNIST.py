import os
from .dataset import _MetaDataset, _MetaImageDataset
from ..util.imagefile import Image


class NotMNIST(_MetaDataset):
    # No download: root/notMNIST_large/<letter>/*.png is expected to be there.

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, transform, target_transform)
        self.train = train
        self._scan(_MetaImageDataset.extensions, os.path.join(self.root, 'notMNIST_large'))
        self._classify()
        self._split(train, ratio=0.8, seed=42)
        self._open(Image)
