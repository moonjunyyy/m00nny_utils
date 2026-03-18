# Wraping for the SHVN dataset

import os
import numpy as np
from typing import Callable, Optional
from .dataset_base import DatasetBase


class CIFAR10(DatasetBase):
    _url_cifar10 = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
    _cifar10_md5 = "c32a1d4ab5d03f1284b67883e8d87530"
    _cifar10_classes = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        *args,
        metadata: dict = {"return_items": ["image", "class"]},
        **kwargs,
    ) -> None:
        super.__init__(root, *args, metadata=metadata, **kwargs)

    def __download__(self):
        from ..util.download_url import DownloadURL
        from ..util.signdir import get_file_hash
        import tarfile
        _filepath = os.path.join(self.root, "cifar-10-binary.tar.gz")
        DownloadURL(url=CIFAR10._url_cifar10, save_path=self.root,).download()
        if get_file_hash(_filepath) != CIFAR10._cifar10_md5:
            os.remove(_filepath)
            raise ValueError(
                "Downloaded file hash does not match expected hash")
        os.makedirs(self.root, exist_ok=True)
        with tarfile.open(_filepath, "r:gz") as tar:
            tar.extractall(path=os.path.join(self.root, "cifar-10-binary"))
        os.remove(_filepath)

    def __load__(self):
        if self.split == 'train':
            _files = [f"data_batch_{i}.bin" for i in range(1, 6)]
        else:
            _files = ["test_batch.bin"]
        for _file in _files:
            _filepath = os.path.join(self.root, "cifar-10-binary", _file)
            with open(_filepath, "rb") as f:
                while True:
                    _data = f.read(3073)
                    if not _data:
                        break
                    self.samples.append(_data[1:])  # Image data
                    self.classes.append(_data[0])  # Class label

    def __get_image__(self, index):
        _data = self.samples[index]
        return np.transpose(
            np.frombuffer(_data, dtype=np.uint8).reshape(32, 32, 3), (1, 2, 0)
        )

    def __get_class__(self, index):
        return int(self.classes[index])


class CIFAR100(CIFAR10):
    _url_cifar100 = "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz"
    _cifar100_md5 = "03b5dce01913d631647c71ecec9e9cb8"
    _cifar100_coarse_classes = {
        0: "aquatic mammals",
        1: "fish",
        2: "flowers",
        3: "food containers",
        4: "fruit and vegetables",
        5: "household electrical devices",
        6: "household furniture",
        7: "insects",
        8: "large carnivores",
        9: "large man-made outdoor things",
        10: "large natural outdoor scenes",
        11: "large omnivores and herbivores",
        12: "medium-sized mammals",
        13: "non-insect invertebrates",
        14: "people",
        15: "reptiles",
        16: "small mammals",
        17: "trees",
        18: "vehicles 1",
        19: "vehicles 2",
    }
    _cifar100_fine_classes = {
        0:  "apple", 1:  "aquarium_fish", 2:  "baby", 3:  "bear", 4:  "beaver",
        5:  "bed", 6:  "bee", 7:  "beetle", 8:  "bicycle", 9:  "bottle",
        10: "bowl", 11: "boy", 12: "bridge", 13: "bus", 14: "butterfly",
        15: "camel", 16: "can", 17: "castle", 18: "caterpillar", 19: "cattle",
        20: "chair", 21: "chimpanzee", 22: "clock", 23: "cloud", 24: "cockroach",
        25: "couch", 26: "crab", 27: "crocodile", 28: "cup", 29: "dinosaur",
        30: "dolphin", 31: "elephant", 32: "flatfish", 33: "forest", 34: "fox",
        35: "girl", 36: "hamster", 37: "house", 38: "kangaroo", 39: "keyboard",
        40: "lamp", 41: "lawn_mower", 42: "leopard", 43: "lion", 44: "lizard",
        45: "lobster", 46: "man", 47: "maple_tree", 48: "motorcycle", 49: "mountain",
        50: "mouse", 51: "mushroom", 52: "oak_tree", 53: "orange", 54: "orchid",
        55: "otter", 56: "palm_tree", 57: "pear", 58: "pickup_truck", 59: "pine_tree",
        60: "plain", 61: "plate", 62: "poppy", 63: "porcupine", 64: "possum",
        65: "rabbit", 66: "raccoon", 67: "ray", 68: "road", 69: "rocket",
        70: "rose", 71: "sea", 72: "seal", 73: "shark", 74: "shrew",
        75: "skunk", 76: "skyscraper", 77: "snail", 78: "snake", 79: "spider",
        80: "squirrel", 81: "streetcar", 82: "sunflower", 83: "sweet_pepper", 84: "table",
        85: "tank", 86: "telephone", 87: "television", 88: "tiger", 89: "tractor",
        90: "train", 91: "trout", 92: "tulip", 93: "turtle", 94: "wardrobe",
        95: "whale", 96: "willow_tree", 97: "wolf", 98: "woman", 99: "worm",
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        *args,
        metadata: dict = {
            "return_items": ["image", "super_class", "class"]
        },
        **kwargs,
    ) -> None:
        super.__init__(root, split, *args, metadata=metadata, **kwargs)

    def __download__(self):
        from ..util.download_url import DownloadURL
        from ..util.signdir import get_file_hash
        import tarfile
        _filepath = os.path.join(self.root, "cifar-100-binary.tar.gz")
        DownloadURL(url=CIFAR100._url_cifar100,
                    save_path=self.root,).download()
        if get_file_hash(_filepath) != CIFAR100._cifar100_md5:
            os.remove(_filepath)
            raise ValueError(
                "Downloaded file hash does not match expected hash")
        os.makedirs(self.root, exist_ok=True)
        with tarfile.open(_filepath, "r:gz") as tar:
            tar.extractall(path=os.path.join(self.root, "cifar-100-binary"))
        os.remove(_filepath)

    def __load__(self):
        if self.split == 'train':
            _files = ["train.bin"]
        else:
            _files = ["test.bin"]
        for _file in _files:
            _filepath = os.path.join(self.root, "cifar-100-binary", _file)
            with open(_filepath, "rb") as f:
                while True:
                    _data = f.read(3074)
                    if not _data:
                        break
                    self.samples.append(_data[2:])  # Image data
                    self.super_classes.append(_data[0])  # Super class label
                    self.classes.append(_data[1])  # Class label

    def __get_super_class__(self, index):
        return int(self.super_classes[index])
