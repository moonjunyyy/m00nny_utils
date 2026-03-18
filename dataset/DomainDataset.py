import torch
import os
from typing import List
from .dataset import DatasetBase as DatasetBase


class _SubDomainDataset(DatasetBase):
    pass


class DomainDataset(DatasetBase):
    """
    A base class for domain datasets.
    """

    def __init__(
        self,
        root: str,
        transform=None,
        target_transform=None,
    ) -> None:
        super().__init__(root, transform, target_transform)
        self.domains = []
        self.targets = []

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
