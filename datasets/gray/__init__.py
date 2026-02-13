from .FashionMNIST import FashionMNIST
from .MNIST import MNIST
from .NotMNIST import NotMNIST
from .grayCUB200 import CUB200
from .grayFlowers102 import Flowers102
from .grayImagenet_R import Imagenet_R
from .graySVHN import SVHN
from .grayTinyImageNet import TinyImageNet

__all__ = [
    "FashionMNIST",
    "MNIST",
    "NotMNIST",
    "CUB200",
    "Flowers102",
    "Imagenet_R",
    "SVHN",
    "TinyImageNet",
]


def get_grayscale_dataset(
    name,
    train=True,
    transform=None,
    target_transform=None,
    root="./data",
    download=True,
):
    dataset_class = globals()[name]
    dataset = dataset_class(
        root=root,
        train=True,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )
    return dataset
