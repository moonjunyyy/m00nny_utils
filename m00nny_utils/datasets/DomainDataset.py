import torch
import os
from typing import List
from .dataset import _MetaDataset as _MetaDataset
from ..util.labels import UNLABELED

class _SubDomainDataset(_MetaDataset):pass

class DomainDataset(_MetaDataset):
    """
    A base class for domain datasets.
    """
    def __init__(
        self,
        root: str,
        transform = None,
        target_transform = None,
    ) -> None:
        super().__init__(root, transform, target_transform)
        self.domains = []       # The domain index of each sample (UNLABELED: none, e.g. a test sample)
        self.domain_names = []  # The domain names; a domain is an index into it

    @property
    def class_names(self):
        return self.classes

    @property
    def domains_targets(self):
        """(domain, target) of each sample."""
        return list(zip(self.domains, self.targets))

    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        return sample, target, self.domains[index]

    def _scan(self, extensions, directory=None, class_of=None, keep=None, domain=UNLABELED):
        # The samples of one scan are one domain.
        count = super()._scan(extensions, directory, class_of, keep)
        self.domains += [domain] * count
        return count

    def _keep(self, indices):
        super()._keep(indices)
        self.domains = [self.domains[i] for i in indices]

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str
