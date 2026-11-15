import os
from .dataset import _MetaDataset, _MetaImageDataset
from ..util.imagefile import Image

# CUB200 dataset class
# Download code from https://github.com/JH-LEE-KR/ContinualDatasets/blob/main/continual_datasets/continual_datasets.py
# by JH-LEE-KR

class CUB200(_MetaDataset):
    # The zip holds the images as a second archive.
    resources = (
        ('https://data.deepai.org/CUB200(2011).zip', 'CUB200(2011).zip', 'CUB_200_2011.tgz'),
        (None, 'CUB_200_2011.tgz', 'CUB_200_2011'),
    )

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, transform, target_transform)
        self.train = train
        self._fetch(download)
        self._scan(_MetaImageDataset.extensions, os.path.join(self.root, 'CUB_200_2011', 'images'))
        self._classify()
        self._split(train, ratio=0.8, seed=42)
        self._open(Image)
