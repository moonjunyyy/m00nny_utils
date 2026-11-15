import numpy as np
from .augmentation import VisionAugmentation


class ColorJitter(VisionAugmentation):
    def __init__(self, b=0.05, c=0.1, s=0.1, seed=None):
        self.rng = np.random.default_rng(seed)
        self.b = b
        self.c = c
        self.s = s

    def apply(self, stream, meta):
        # eq leaves a picture as it is at brightness 0 (an offset),
        # contrast 1 and saturation 1 (factors)
        bright = self.rng.uniform(-self.b, self.b)
        contrast = 1 + self.rng.uniform(-self.c, self.c)
        sat = 1 + self.rng.uniform(-self.s, self.s)

        return stream.filter(
            "eq",
            brightness=bright,
            contrast=contrast,
            saturation=sat
        ), meta
