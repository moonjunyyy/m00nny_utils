import torch
import torch.distributed as dist
from typing import Optional, Sized, Iterable, Tuple, Union
from ..dataset.dataset_base import DatasetBase

from ..system.log import Log
Logger = Log(name="sampler:sampler_base")


class SamplerBase:
    def __init__(
        self,
        dataset: DatasetBase,
        *args,
        shuffle: bool = True,
        generator: Optional[torch.Generator] = None,
        num_replicas: Optional[int] = 1,
        rank: Optional[int] = 0,
        remainders: str = "drop",
        **kwargs
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.generator = generator if generator is not None else torch.Generator()

        self.num_replicas = num_replicas
        self.rank = rank
        self.remainders = remainders
        assert remainders in ("drop", "extend", "ignore"), \
            f"Invalid remainders option: {remainders}"

        self._n_samples = len(self.dataset)
        self._n_per_rank = self._n_samples // self.num_replicas
        if remainders == "extend":
            self._n_per_rank += int(
                self._n_samples % self.num_replicas != 0)
        elif remainders == "ignore":
            self._n_per_rank = int(
                self._n_samples % self.num_replicas < self.rank)
        elif remainders == "drop":
            self._n_per_rank = int(self._n_samples // self.num_replicas)
        self.__epoch_counter = 0

    @torch.no_grad()
    def __gen_sequence__(self):
        _indices = list(range(len(self.dataset)))
        if self.shuffle:
            if self.generator is None:
                self.generator = torch.Generator()
                self.generator.manual_seed(0)
            _indices = torch.randperm(
                len(self.dataset),
                generator=self.generator
            )
        _indices = self.__distributed_split__(_indices)
        assert len(_indices) == self._n_per_rank, \
            f"Expected {self._n_per_rank} samples per rank, but got {len(_indices)}"
        return _indices

    @torch.no_grad()
    def __distributed_split__(self, _indices):
        if self.remainders == "drop":
            _indices = _indices[:self._n_per_rank * self.num_replicas]
        elif self.remainders == "extend":
            _total_size = len(_indices)
            _required_size = self._n_per_rank * self.num_replicas
            _required_size = _required_size - _total_size
            if _required_size > 0:
                _indices = torch.cat([
                    _indices,
                    torch.randint(
                        0, _total_size,
                        (_required_size,),
                        generator=self.generator)
                ])
            # Each process will generate same indices,
            # make sure the Generator is consistent across processes.
        _indices = _indices.view(self.num_replicas, -1)[self.rank].tolist()
        return _indices

        if self.num_replicas > 1:
            total_size = len(_indices)
            _indices += _indices[:(self.num_replicas - total_size %
                                 self.num_replicas) % self.num_replicas]
            assert len(_indices) % self.num_replicas == 0
            per_replica = len(_indices) // self.num_replicas
            return _indices[self.rank * per_replica:(self.rank + 1) * per_replica]
        else:
            return _indices

    def __len__(self):
        return self._n_per_rank

    def __iter__(self):
        self.__epoch_counter += 1
        return iter(self.__gen_sequence__())
