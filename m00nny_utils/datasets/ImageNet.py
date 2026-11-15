import os
from .dataset import _MetaDataset, _MetaImageDataset
from ..util.imagefile import Image


class ImageNet(_MetaDataset):
    # No download (the archives are not public): root/train/<wnid>/*.JPEG and root/val/<wnid>/*.JPEG are expected.

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, transform, target_transform)
        self.train = train
        self._scan(_MetaImageDataset.extensions, os.path.join(self.root, 'train' if train else 'val'),
                   class_of=lambda path: path.split(os.sep)[0])
        self._classify()
        self._open(Image)
