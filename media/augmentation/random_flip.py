import numpy as np
from .augmentation import VisionAugmentation


class RandomHFlip(VisionAugmentation):
    def __init__(self, p=0.5, seed=None):
        self.p = p
        self.rng = np.random.default_rng(seed)

    def apply(self, stream, meta):
        if self.rng.random() < self.p:
            return stream.hflip(), meta
        return stream, meta


class RandomVFlip(VisionAugmentation):
    def __init__(self, p=0.5, seed=None):
        self.p = p
        self.rng = np.random.default_rng(seed)

    def apply(self, stream, meta):
        if self.rng.random() < self.p:
            return stream.vflip(), meta
        return stream, meta
