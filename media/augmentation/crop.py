import numpy as np
from .augmentation import VisionAugmentation


class RandomCrop(VisionAugmentation):
    def __init__(self, crop_ratio=0.8, seed=None):
        self.rng = np.random.default_rng(seed)
        self.crop_ratio = crop_ratio

    def apply(self, stream, meta):
        w, h = meta.get("width"), meta.get("height")
        cw, ch = int(w * self.crop_ratio), int(h * self.crop_ratio)
        x = self.rng.integers(0, w - cw + 1)
        y = self.rng.integers(0, h - ch + 1)
        return stream.crop(x, y, cw, ch).filter("scale", w, h), meta


class CenterCrop(VisionAugmentation):
    def __init__(self, crop_ratio=0.8):
        self.crop_ratio = crop_ratio

    def apply(self, stream, meta):
        w, h = meta.get("width"), meta.get("height")
        cw, ch = int(w * self.crop_ratio), int(h * self.crop_ratio)
        x = (w - cw) // 2
        y = (h - ch) // 2
        return stream.crop(x, y, cw, ch).filter("scale", w, h), meta
