from typing import Iterable
from .augmentation import Augmentation


class Compose(Augmentation):
    def __init__(self, ops: Iterable[Augmentation]):
        self.ops = ops

    def apply(self, stream, meta):
        for op in self.ops:
            stream, meta = op(stream, meta)
        return stream, meta
