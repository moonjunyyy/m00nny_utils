from .DomainDataset import DomainDataset
from .MNIST import MNIST
from .SVHN import SVHN
from .MNISTM import MNISTM
from .SynDigit import SynDigit


class iDigits(DomainDataset):
    # Each digit dataset is a domain.
    datasets = (MNIST, SVHN, MNISTM, SynDigit)

    def __init__(self, root, train=True, transform=None, target_transform=None, download=True):
        super().__init__(root, transform, target_transform)
        self.train = train
        for d, dataset in enumerate(self.datasets):
            part = dataset(root, train, None, None, download)
            self.samples += part.samples
            self.targets += part.targets
            self.domains += [d] * len(part)
        self.classes = [f"{i}" for i in range(10)]
        self.domain_names = [dataset.__name__ for dataset in self.datasets]
