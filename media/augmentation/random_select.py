import numpy as np
from typing import List, Optional
from .augmentation import Augmentation


class RandomSelect(Augmentation):
    def __init__(
        self,
        *ops: Augmentation,
        n_choice: Optional[int] = 1,
        dist: Optional[List[float]] = None,
        seed: Optional[int] = None
    ):
        self.ops = ops
        self.n_apply = n_choice
        if dist is None:
            self.dist = [1 / len(ops)] * len(ops)
        elif isinstance(dist, list):
            if len(dist) != len(ops):
                raise ValueError("Length of dist must match length of ops")
            self.dist = dist
            _sum_dist = sum(dist)
            if _sum_dist > 1 or _sum_dist < 1:
                _scale = 1 / _sum_dist
                self.dist = [d * _scale for d in dist]
        if seed is None:
            seed = np.random.SeedSequence().entropy
        self.rng = np.random.default_rng(seed)

    def apply(self, stream, meta):
        op = self.rng.choice(self.ops)
        for _ in range(self.n_apply - 1):
            op = self.rng.choice(self.ops, p=self.dist)
            stream, meta = op(stream, meta)
        return stream, meta
