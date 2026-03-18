import numpy as np
import ffmpeg
from typing import List, Optional
from .augmentation import Augmentation, VisionAugmentation


class Resize(Augmentation):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def apply(self, stream, meta):
        w, h = meta.get("width"), meta.get("height")
        if not w or not h:
            return stream, meta
        meta["width"] = self.width
        meta["height"] = self.height
        return stream.filter("scale", self.width, self.height), meta
