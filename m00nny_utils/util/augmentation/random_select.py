import numpy as np
from typing import Optional, Sequence
from .augmentation import Augmentation


class RandomSelect(Augmentation):
    """n_choice of the operations, each drawn with the probabilities dist
    (equal when not given)."""

    def __init__(
        self,
        *ops: Augmentation,
        n_choice: Optional[int] = 1,
        dist: Optional[Sequence[float]] = None,
        seed: Optional[int] = None
    ):
        self.ops = ops
        self.n_apply = n_choice
        if dist is None:
            self.dist = [1 / len(ops)] * len(ops)
        else:
            dist = list(dist)
            if len(dist) != len(ops):
                raise ValueError("Length of dist must match length of ops")
            _scale = 1 / sum(dist)
            self.dist = [d * _scale for d in dist]
        if seed is None:
            seed = np.random.SeedSequence().entropy
        self.rng = np.random.default_rng(seed)

    def apply(self, stream, meta):
        for _ in range(self.n_apply):
            op = self.ops[self.rng.choice(len(self.ops), p=self.dist)]
            stream, meta = op(stream, meta)
        return stream, meta
