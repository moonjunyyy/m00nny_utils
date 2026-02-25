from .colorFashionMNIST import FashionMNIST
from .colorMNIST import MNIST
from .colorNotMNIST import NotMNIST
from .CUB200 import CUB200
from .CIFAR10 import CIFAR10
from .CIFAR100 import CIFAR100
from .Flowers102 import Flowers102
from .Imagenet_R import Imagenet_R
from .SVHN import SVHN
from .TinyImageNet import TinyImageNet

__all__ = [
    "FashionMNIST",
    "MNIST",
    "NotMNIST",
    "CUB200",
    "CIFAR10",
    "CIFAR100",
    "Flowers102",
    "Imagenet_R",
    "SVHN",
    "TinyImageNet",
]


def get_rgb_dataset(
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
