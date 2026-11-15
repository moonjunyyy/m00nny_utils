from .augmentation import VisionAugmentation


class Grayscale(VisionAugmentation):
    """Grey, in three equal channels."""

    def apply(self, stream, meta):
        return stream.filter("hue", s=0), meta
