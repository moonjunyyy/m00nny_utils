import os
from typing import Dict
from ..system.log import Log
from .dataset_base import DatasetBase


logger = Log(name="dataset:concat_dataset")


class ConcatDataset(DatasetBase):
    def __init__(
        self,
        *datasets,
        **kwargs,
    ):
        self._lengths = []
        self._datasets = datasets
        self.metadata = None
        for dataset in datasets:
            self._lengths.append(len(dataset))
            assert self.metadata is None or self.metadata == dataset.metadata, \
                "All datasets must have the same metadata"
            self.metadata = dataset.metadata
        self._total_length = sum(self._lengths)

    def __len__(self):
        return self._total_length

    def __getitem__(self, index):
        if index < 0:
            index += self._total_length
        if index < 0 or index >= self._total_length:
            raise IndexError("Index out of range")
        for i, length in enumerate(self._lengths):
            if index < length:
                return self._datasets[i][index]
            index -= length
