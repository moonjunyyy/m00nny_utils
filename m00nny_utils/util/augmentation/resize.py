from .augmentation import VisionAugmentation


class Resize(VisionAugmentation):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def apply(self, stream, meta):
        meta["width"] = self.width
        meta["height"] = self.height
        return stream.filter("scale", self.width, self.height), meta
