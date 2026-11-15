import os
import os.path
import numpy as np
from PIL import Image
from shutil import move, rmtree

import torch
from torchvision import datasets
from torchvision.datasets.utils import download_url
from typing import Any, Callable, Optional, Tuple, TypeVar

PathLike = TypeVar("PathLike", str, bytes, os.PathLike)
ImageLike = TypeVar("ImageLike", Image.Image, torch.Tensor, np.ndarray)


import tqdm
import zipfile
from .DomainDataset import DomainDataset

# from .dataset_utils import read_image_file, read_label_file


class _sub_CLEAR(datasets.ImageFolder, DomainDataset):
    def __init__(
        self,
        root: PathLike,
        train: bool = True,
        session: str = "0",
        transform: Callable[[ImageLike], torch.Tensor] | None = None,
        target_transform: Callable[[ImageLike], torch.Tensor] | None = None,
        download: bool = False,
    ) -> None:
        self.root = os.path.expanduser(root)
        root = os.path.join(root, "CLEAR")
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.session = session
        self.url = "https://huggingface.co/datasets/elvishelvis6/CLEAR-Continual_Learning_Benchmark/resolve/main/"
        self.filename = {
            "train": "clear100-train-image-only.zip",
            "test": "clear100-test.zip",
        }

        filename = self.filename["train"] if train else self.filename["test"]

        if not os.path.exists(root):
            os.makedirs(root)
        if not os.path.isfile(os.path.join(root, filename)):
            if not download:
                raise RuntimeError(
                    "Dataset not found. You can use download=True to download it"
                )
            else:
                url = self.url + filename
                print("Downloading from " + url)
                download_url(url, root)

        if not os.path.exists(
            os.path.join(root, "train_image_only" if train else "test")
        ):
            with zipfile.ZipFile(os.path.join(root, filename), "r") as zf:
                for member in tqdm.tqdm(zf.infolist(), desc=f"Extracting {filename}"):
                    try:
                        zf.extract(member, root)
                    except zipfile.error as e:
                        pass
        self.train_path = os.path.join(
            root, "train_image_only", "labeled_images", str(session)
        )
        self.test_path = os.path.join(root, "test", "labeled_images", str(session))

        super(_sub_CLEAR, self).__init__(
            self.train_path if train else self.test_path,
            transform=transform,
            target_transform=target_transform,
        )
        self.domains = [session] * len(self.samples)

    def __getitem__(self, index):
        sample, target = super(_sub_CLEAR, self).__getitem__(index)
        domain = self.domains[index]
        return sample, torch.tensor(target), torch.tensor(domain)


class CLEAR(torch.utils.data.ConcatDataset):
    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):
        self.datasets = []
        for session in range(11):
            self.datasets.append(
                _sub_CLEAR(root, train, session, transform, target_transform, download)
            )
            print(f"Session: {session} is loaded")
            print(f"Number of samples: {len(self.datasets[-1])}")
        super(CLEAR, self).__init__(self.datasets)
        print(f"Total number of {'Train' if train else 'Test'} samples: {len(self)}")
        self.classes = [f"{i}" for i in range(100)]  # It is no meaning :)
        self.domain_names = [str(i) for i in range(11)]
        self.targets = []
        self.domains = []
        for dataset in self.datasets:
            self.targets += [target for target in dataset.targets]
        for dataset in self.datasets:
            self.domains += [domain for domain in dataset.domains]
