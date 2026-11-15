import torch
from torch.utils.data.sampler import Sampler
from typing import Iterable, List, Optional, Sized


class _ReplicaSampler(Sampler):
    """The part every sampler here shares: one replica's share of a list of indices."""

    def _init_replicas(self, num_replicas: Optional[int], rank: Optional[int]) -> None:
        self.distributed = num_replicas is not None and rank is not None
        self.num_replicas = num_replicas if num_replicas is not None else 1
        self.rank = rank if rank is not None else 0

    def _share(self, indices: List[int]) -> List[int]:
        # Every replica gets the same number of samples: the tail that does not divide is left out.
        num_samples = len(indices) // self.num_replicas
        return indices[self.rank:num_samples * self.num_replicas:self.num_replicas]


class SamplerBase(_ReplicaSampler):
    """Every sample of the dataset, in order or shuffled: one replica's share of them."""

    def __init__(
        self,
        dataset: Sized,
        *args,
        shuffle: bool = True,
        generator: Optional[torch.Generator] = None,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        **kwargs
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.generator = generator if generator is not None else torch.Generator()
        self._init_replicas(num_replicas, rank)

    def __iter__(self) -> Iterable[int]:
        # Every replica draws the same order (give them generators in the same state) and takes its share of it.
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=self.generator).tolist()
        else:
            indices = list(range(len(self.dataset)))
        return iter(self._share(indices))

    def __len__(self) -> int:
        return len(self.dataset) // self.num_replicas
