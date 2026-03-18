import os
from typing import Dict
from torch.utils.data import Dataset
from typing import Callable, Iterable, Optional
from ..util.imagefile import Image
from ..system.log import Log
from .dataset_base import DatasetBase


logger = Log(name="dataset:imagefolder")
_ImageExtensions = set(
    ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')
)


class ImageFolder(DatasetBase):

    def __init__(
        self,
        root,
        download: bool = False,
        *args,
        metadata: Dict = {"return_items":  ["image", "class"]},
        **kwargs,
    ):
        self.download = download
        super().__init__(root, *args, metadata=metadata, **kwargs)

    def __load__(self):
        for dirpath, _, filenames in os.walk(self.root):
            for filename in filenames:
                if filename.lower().endswith(tuple(ImageFolder._ImageExtensions)):
                    self.samples.append(Image(os.path.join(dirpath, filename)))
                    self.classes.append(tuple(
                        os.path.basename(dirpath)
                        .substring(len(self.root) + 1)
                        .split(os.sep)
                    ))

    def __get_samples__(self, index):
        return self.samples[index]

    def __get_classes__(self, index):
        return self.classes[index]
