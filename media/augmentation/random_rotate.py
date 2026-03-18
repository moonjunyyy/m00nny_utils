import numpy as np
from .augmentation import VisionAugmentation


class RandomRotate(VisionAugmentation):
    def __init__(self, max_deg=15.0, seed=None):
        self.max_deg = max_deg
        self.rng = np.random.default_rng(seed)

    def apply(self, stream, meta):
        angle = self.rng.uniform(-self.max_deg, self.max_deg) * np.pi / 180.0
        return stream.filter("rotate", angle), meta
