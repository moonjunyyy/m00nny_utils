import os
import os.path
import numpy as np
from PIL import Image
from shutil import move, rmtree
import psutil

import torch
from torchvision import datasets
from torchvision.datasets.utils import (
    download_url,
    check_integrity,
    download_and_extract_archive,
)
from typing import Any, Callable, Optional, Tuple, TypeVar

PathLike = TypeVar("PathLike", str, bytes, os.PathLike)
ImageLike = TypeVar("ImageLike", Image.Image, torch.Tensor, np.ndarray)

import tqdm
import zipfile
from .DomainDataset import DomainDataset
from .dataset_utils import read_image_file, read_label_file

# from .dataset_utils import read_image_file, read_label_file


class _sub_MNIST(datasets.MNIST, DomainDataset):
    """
    MNIST RGB
    """

    def __init__(
        self,
        root: PathLike,
        train: bool = True,
        session: int = 0,
        transform: Callable[[ImageLike], torch.Tensor] | None = None,
        target_transform: Callable[[ImageLike], torch.Tensor] | None = None,
        download: bool = False,
    ) -> None:
        self.root = os.path.expanduser(root)
        root = os.path.join(root, "MNIST-RGB")
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if self._check_legacy_exist():
            self.data, self.targets = self._load_legacy_data()
            return
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )
        self.data, self.targets = self._load_data()
        # super(_sub_MNIST, self).__init__(root, transform=transform, target_transform=target_transform, download=download)
        self.domains = [session] * len(self.data)

    def _check_legacy_exist(self):
        processed_folder_exists = os.path.exists(self.processed_folder)
        if not processed_folder_exists:
            return False
        return all(
            check_integrity(os.path.join(self.processed_folder, file))
            for file in (self.training_file, self.test_file)
        )

    def _load_legacy_data(self):
        # This is for BC only. We no longer cache the data in a custom binary, but simply read from the raw data
        # directly.
        data_file = self.training_file if self.train else self.test_file
        return torch.load(os.path.join(self.processed_folder, data_file))

    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        data = read_image_file(os.path.join(self.raw_folder, image_file))

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        targets = read_label_file(os.path.join(self.raw_folder, label_file))

        return data, targets

    def __getitem__(self, index):
        sample, target = self.data[index], int(self.targets[index])
        try:
            sample = Image.fromarray(sample.numpy(), mode="L").convert("RGB")
        except:
            pass
        domain = self.domains[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, domain


class _sub_SVHN(datasets.SVHN, DomainDataset):
    def __init__(
        self,
        root: PathLike,
        train: bool = True,
        session: int = 0,
        transform: Callable[[ImageLike], torch.Tensor] | None = None,
        target_transform: Callable[[ImageLike], torch.Tensor] | None = None,
        download: bool = False,
    ) -> None:
        self.root = os.path.expanduser(root)
        root = os.path.join(root, "SVHN")
        self.train = train

        super(_sub_SVHN, self).__init__(
            root, "train" if self.train else "test", download=download
        )
        self.transform = transform
        self.target_transform = target_transform

        self.domains = [session] * len(self.data)
        self.targets = self.labels
        # del self.labels

    def __getitem__(self, index):
        sample, target = self.data[index], int(self.labels[index])
        sample = Image.fromarray(np.transpose(sample, (1, 2, 0)))
        domain = self.domains[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, domain


class _sub_MNISTM(datasets.MNIST, DomainDataset):
    def __init__(
        self,
        root: PathLike,
        train: bool = True,
        session: int = 0,
        transform: Callable[[ImageLike], torch.Tensor] | None = None,
        target_transform: Callable[[ImageLike], torch.Tensor] | None = None,
        download: bool = False,
    ) -> None:
        self.root = os.path.expanduser(root)
        root = os.path.join(root, "MNIST-M")
        self.train = train
        self.url = [
            (
                "https://github.com/liyxi/mnist-m/releases/download/data/mnist_m_train.pt.tar.gz",
                "191ed53db9933bd85cc9700558847391",
            ),
            (
                "https://github.com/liyxi/mnist-m/releases/download/data/mnist_m_test.pt.tar.gz",
                "e11cb4d7fff76d7ec588b1134907db59",
            ),
        ]

        self.training_file = "mnist_m_train.pt"
        self.test_file = "mnist_m_test.pt"

        # super(_sub_MNISTM, self).__init__(root=root)
        self.transform = transform
        self.target_transform = target_transform
        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found." + " You can use download=True to download it"
            )

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.classes = [i for i in range(10)]
        self.data, self.targets = torch.load(
            os.path.join(self.processed_folder, data_file)
        )
        # self.targets = self.targets.long().tolist()
        self.domains = [session] * len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, "raw")

    @property
    def processed_folder(self):
        return os.path.join(self.root, "processed")

    def _check_exists(self):
        return os.path.exists(
            os.path.join(self.processed_folder, self.training_file)
        ) and os.path.exists(os.path.join(self.processed_folder, self.test_file))

    def download(self):
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.url:
            filename = url.rpartition("/")[2]
            download_and_extract_archive(
                url,
                download_root=self.raw_folder,
                extract_root=self.processed_folder,
                filename=filename,
                md5=md5,
            )

    def __getitem__(self, index):
        sample, target = self.data[index], int(self.targets[index])
        sample = Image.fromarray(sample.squeeze().numpy(), mode="RGB")
        domain = self.domains[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, domain

    def __len__(self):
        return len(self.data)


class _sub_SynDigit(datasets.MNIST, DomainDataset):
    def __init__(
        self,
        root: PathLike,
        train: bool = True,
        session: int = 0,
        transform: Callable[[ImageLike], torch.Tensor] | None = None,
        target_transform: Callable[[ImageLike], torch.Tensor] | None = None,
        download: bool = False,
    ) -> None:
        self.root = os.path.expanduser(root)
        root = os.path.join(root, "SynDigit")
        self.train = train
        self.url = [
            (
                "https://github.com/liyxi/synthetic-digits/releases/download/data/synth_train.pt.gz",
                "d0e99daf379597e57448a89fc37ae5cf",
            ),
            (
                "https://github.com/liyxi/synthetic-digits/releases/download/data/synth_test.pt.gz",
                "669d94c04d1c91552103e9aded0ee625",
            ),
        ]

        self.training_file = "synth_train.pt"
        self.test_file = "synth_test.pt"

        # super(_sub_SynDigit, self).__init__(root=root)
        if download:
            self.download()
        self.transform = transform
        self.target_transform = target_transform
        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found." + " You can use download=True to download it"
            )

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.classes = [i for i in range(10)]
        self.data, self.targets = torch.load(
            os.path.join(self.processed_folder, data_file)
        )
        self.domains = [session] * len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, "raw")

    @property
    def processed_folder(self):
        return os.path.join(self.root, "processed")

    def _check_exists(self):
        return os.path.exists(
            os.path.join(self.processed_folder, self.training_file)
        ) and os.path.exists(os.path.join(self.processed_folder, self.test_file))

    def download(self):
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.url:
            filename = url.rpartition("/")[2]
            download_and_extract_archive(
                url,
                download_root=self.raw_folder,
                extract_root=self.processed_folder,
                filename=filename,
                md5=md5,
            )

    def __getitem__(self, index):
        sample, target = self.data[index], int(self.targets[index])
        sample = Image.fromarray(sample.squeeze().numpy(), mode="RGB")
        domain = self.domains[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, domain

    def __len__(self):
        return len(self.data)


class iDigits(torch.utils.data.ConcatDataset):
    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=True
    ):
        self.datasets = []
        for d, session in enumerate(
            [_sub_MNIST, _sub_SVHN, _sub_MNISTM, _sub_SynDigit]
        ):
            self.datasets.append(
                session(
                    root=root,
                    session=d,
                    train=train,
                    transform=transform,
                    target_transform=target_transform,
                    download=download,
                )
            )
            print(f"Session: {session.__name__} is loaded")
            print(f"Number of samples: {len(self.datasets[-1])}")
        super(iDigits, self).__init__(self.datasets)
        print(f"Total number of {'Train' if train else 'Test'} samples: {len(self)}")
        self.classes = [f"{i}" for i in range(10)]  # It is no meaning :)
        self.domain_names = [
            str(i) for i in range(len(["MNIST", "SVHN", "MNISTM", "SynDigit"]))
        ]  # fixed
        self.targets = []
        self.domains = []
        for dataset in self.datasets:
            self.targets += [target for target in dataset.targets.tolist()]
        for dataset in self.datasets:
            self.domains += [domain for domain in dataset.domains]

    def __getitem__(self, index):
        _sample, _target, _domain = super(iDigits, self).__getitem__(index)
        # _domain = self.domain_to_idx[_domain]
        return _sample, _target, _domain
