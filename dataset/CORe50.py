from .DomainNet import DomainDataset
import zipfile
import tqdm
from torchvision.datasets.utils import download_url
from torchvision import datasets
import os
import os.path
import glob
import shutil
import torch
import numpy as np
from PIL import Image
import torch.utils
from torchvision.datasets import ImageFolder
from typing import Any, Callable, Optional, Tuple, TypeVar

PathLike = TypeVar("PathLike", str, bytes, os.PathLike)
ImageLike = TypeVar("ImageLike", Image.Image, torch.Tensor, np.ndarray)


class _sub_CORe50(ImageFolder):
    def __init__(
        self,
        root: PathLike,
        session: str = "s1",
        domain: int = 0,
        transform: Callable[[ImageLike], torch.Tensor] | None = None,
        target_transform: Callable[[ImageLike], torch.Tensor] | None = None,
        download: bool = False,
    ) -> None:
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.session = session

        self.url = "http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip"
        self.filename = "core50_128x128.zip"

        self.train_session_list = ["s1", "s2", "s4", "s5", "s6", "s8", "s9", "s11"]
        self.test_session_list = ["s3", "s7", "s10"]
        self.train = session in self.train_session_list

        self.fpath = os.path.join(root, "core50_128x128")
        if not os.path.exists(os.path.join(root, "core50_128x128")):
            if not os.path.isfile(os.path.join(root, self.filename)):
                if not download:
                    raise RuntimeError(
                        "Dataset not found." "You can use download=True to download it"
                    )
                else:
                    print("Downloading from " + self.url)
                    download_url(self.url, root, filename=self.filename)
            with zipfile.ZipFile(os.path.join(self.root, self.filename), "r") as zf:
                for member in tqdm.tqdm(
                    zf.infolist(), desc=f"Extracting {self.filename}"
                ):
                    try:
                        zf.extract(member, root)
                    except zipfile.error as _:
                        pass

        if not os.path.exists(os.path.join(root, "core50_128x128") + "/train"):
            os.makedirs(os.path.join(root, "core50_128x128") + "/train", exist_ok=True)
            for session in self.train_session_list:
                shutil.move(
                    os.path.join(root, "core50_128x128") + "/" + session,
                    os.path.join(root, "core50_128x128") + "/train/",
                )
        if not os.path.exists(os.path.join(root, "core50_128x128") + "/test"):
            os.makedirs(os.path.join(root, "core50_128x128") + "/test", exist_ok=True)
            for session in self.test_session_list:
                shutil.move(
                    os.path.join(root, "core50_128x128") + "/" + session,
                    os.path.join(root, "core50_128x128") + "/test/",
                )

        self.fpath = self.fpath + ("/train/" if self.train else "/test/") + self.session
        super(_sub_CORe50, self).__init__(self.fpath, transform=transform)
        self.targets = [target for _, target in self.samples]
        # Domain of the test dataset is ignored
        self.domains = [domain if self.train else -1] * len(self.samples)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img, target = super(_sub_CORe50, self).__getitem__(index)
        return img, target, self.domains[index]


class CORe50(torch.utils.data.ConcatDataset, DomainDataset):
    def __init__(
        self,
        root: PathLike,
        train: bool = True,
        transform: Callable[[ImageLike], torch.Tensor] | None = None,
        target_transform: Callable[[ImageLike], torch.Tensor] | None = None,
        download: bool = False,
    ) -> None:
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        self.train_session_list = ["s1", "s2", "s4", "s5", "s6", "s8", "s9", "s11"]
        self.test_session_list = ["s3", "s7", "s10"]
        if self.train:
            self.session_list = self.train_session_list
        else:
            self.session_list = self.test_session_list

        self.datasets = []
        for d, session in enumerate(self.session_list):
            self.datasets.append(
                _sub_CORe50(root, session, d, transform, target_transform, download)
            )
            print(f"Session: {session} is loaded")
            print(f"Number of samples: {len(self.datasets[-1])}")
        super(CORe50, self).__init__(self.datasets)
        print(
            "Total number of ", "Train" if train else "Test", f" samples: {len(self)}"
        )
        self.classes = [f"{i}" for i in range(50)]
        self.domain_names = [str(i) for i in range(len(self.session_list))]
        self.targets = []
        self.domains = []
        for dataset in self.datasets:
            self.targets += [target for target in dataset.targets]
        for dataset in self.datasets:
            self.domains += [domain for domain in dataset.domains]

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img, target, domain = super(CORe50, self).__getitem__(index)
        return img, target, domain
