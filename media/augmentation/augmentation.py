import numpy as np
from typing import Iterable


# Meta Class for Augmentation
class Augmentation:
    """Base class: ffmpeg filter graph builder"""

    def __init__(self, *args, **kwargs):
        pass

    def apply(self, stream, meta):
        return stream, meta

    def _ensure_in_meta(self, meta, keys):
        for key in keys:
            assert key in meta, \
                f"{key} is required in meta for {self.__class__.__name__}"

    def __call__(self, stream, meta):
        return self.apply(stream, meta)


# Meta Class for Vision Specific Augmentation
class VisionAugmentation(Augmentation):
    def __call__(self, stream, meta):
        self._ensure_in_meta(meta, ["width", "height"])
        return super().apply(stream, meta)


# Meta Class for Audio Specific Augmentation
class AudioAugmentation(Augmentation):
    def __call__(self, stream, meta):
        self._ensure_in_meta(meta, ["sample_rate"])
        return super().apply(stream, meta)


# Meta Class for Audio Specific Augmentation
class VideoAugmentation(VisionAugmentation):
    def __call__(self, stream, meta):
        self._ensure_in_meta(meta, ["width", "height", "frame_rate"])
        return super().apply(stream, meta)
