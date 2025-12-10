from .CUB200 import CUB200
from .multiDatasets import multiDatasets
from .Flowers102 import Flowers102
from .NotMNIST import NotMNIST
from .SVHN import SVHN
from .TinyImageNet import TinyImageNet
from .MNIST import MNIST
from .FashionMNIST import FashionMNIST
from .Imagenet_R import Imagenet_R
from .CORe50 import CORe50
from .DomainNet import DomainNet
from .CLEAR import CLEAR
from .SynDigit import SynDigit
from .MNISTM import MNISTM
from .MNIST_RGB import MNIST_RGB
from .iDigits import iDigits
from .UCF_101 import UCF101
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet

__all__ = [
    "CUB200",
    "multiDatasets",
    "Flowers102",
    "NotMNIST",
    "SVHN",
    "TinyImageNet",
    "CIFAR10",
    "CIFAR100",
    "MNIST",
    "FashionMNIST",
    "ImageNet",
    "Imagenet_R",
    "CORe50",
    "DomainNet",
    "CLEAR",
    "SynDigit",
    "MNISTM",
    "MNIST_RGB",
    "UCF_101",
]

# dictionary of tuple of dataset, mean, std
datasets = {
    "cifar10": (CIFAR10, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), 10),
    "cifar100": (CIFAR100, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761), 100),
    "svhn": (SVHN, (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970), 10),
    "fashionmnist": (FashionMNIST, (0.2860,), (0.3530,), 10),
    "mnist": (MNIST, (0.1307,), (0.3081,), 10),
    "tinyimagenet": (
        TinyImageNet,
        (0.4802, 0.4481, 0.3975),
        (0.2302, 0.2265, 0.2262),
        200,
    ),
    "notmnist": (NotMNIST, (0.1307,), (0.3081,), 10),
    "cub200": (CUB200, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 200),
    "imagenet": (ImageNet, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 1000),
    "imagenet-r": (Imagenet_R, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 200),
    "core50": (CORe50, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 50),
    "domainnet": (DomainNet, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 345),
    "clear100": (CLEAR, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 100),
    "syndigit": (SynDigit, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 10),
    "mnistm": (MNISTM, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 10),
    "mnist_rgb": (MNIST_RGB, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 10),
    "flowers102": (Flowers102, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 102),
    "idigits": (iDigits, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 10),
    "ucf101": (UCF101, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 101),
}


def get_dataset(name):
    return datasets[name.lower()]
