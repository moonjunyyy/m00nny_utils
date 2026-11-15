import os
from .dataset import _MetaImageDataset
from .DomainDataset import DomainDataset
from ..util.labels import UNLABELED
from ..util.imagefile import Image

# TinyImageNet dataset class
# Download code from https://github.com/JH-LEE-KR/ContinualDatasets/blob/main/continual_datasets/continual_datasets.py
# by JH-LEE-KR

class TinyImageNet(DomainDataset):
    resources = (
        ('http://cs231n.stanford.edu/tiny-imagenet-200.zip', 'tiny-imagenet-200.zip', 'tiny-imagenet-200'),
    )

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, transform, target_transform)
        self.train = train
        self._fetch(download)
        self.path = os.path.join(self.root, 'tiny-imagenet-200')
        # The class order is the one of wnids.txt.
        with open(os.path.join(self.path, 'wnids.txt'), 'r') as f:
            classes = [line.strip() for line in f if line.strip()]
        if train:
            # train/<wnid>/images/xxx.JPEG
            self._scan(_MetaImageDataset.extensions, os.path.join(self.path, 'train'),
                       class_of=lambda path: path.split(os.sep)[0], domain=0)
        else:
            # val/images/xxx.JPEG, the class of each file in val_annotations.txt
            with open(os.path.join(self.path, 'val', 'val_annotations.txt'), 'r') as f:
                file_to_class = {line.split('\t')[0]: line.split('\t')[1] for line in f if line.strip()}
            # Domain of the test dataset is ignored
            self._scan(_MetaImageDataset.extensions, os.path.join(self.path, 'val'),
                       class_of=lambda path: file_to_class[os.path.basename(path)], domain=UNLABELED)
        self._classify(classes)
        self.domain_names = ['domain_0']
        self._open(Image)
