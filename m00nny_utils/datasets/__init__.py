from .color.colorFashionMNIST import FashionMNIST as FashionMNIST
from .color.colorMNIST import MNIST as MNIST
from .color.colorNotMNIST import NotMNIST as NotMNIST
from .color.CUB200 import CUB200 as CUB200
from .color.Flowers102 import Flowers102 as Flowers102 
from .color.Imagenet_R import Imagenet_R as Imagenet_R
from .color.SVHN import SVHN as SVHN
from .color.TinyImageNet import TinyImageNet as TinyImageNet

from .gray.FashionMNIST import FashionMNIST as grayFashionMNIST
from .gray.MNIST import MNIST as grayMNIST
from .gray.NotMNIST import NotMNIST as grayNotMNIST
from .gray.grayCUB200 import CUB200 as grayCUB200
from .gray.grayFlowers102 import Flowers102 as grayFlowers102
from .gray.grayImagenet_R import Imagenet_R as grayImagenet_R
from .gray.graySVHN import SVHN as graySVHN
from .gray.grayTinyImageNet import TinyImageNet as grayTinyImageNet
from .gray.grayCIFAR10 import CIFAR10 as grayCIFAR10
from .gray.grayCIFAR100 import CIFAR100 as grayCIFAR100

__all__ = [
    "FashionMNIST",
    "MNIST",
    "NotMNIST",
    "CUB200",
    "Flowers102",
    "Imagenet_R",
    "SVHN",
    "TinyImageNet",
    "grayFashionMNIST",
    "grayMNIST",
    "grayNotMNIST",
    "grayCUB200",
    "grayFlowers102",
    "grayImagenet_R",
    "graySVHN",
    "grayTinyImageNet",
    "grayCIFAR10",
    "grayCIFAR100",
]