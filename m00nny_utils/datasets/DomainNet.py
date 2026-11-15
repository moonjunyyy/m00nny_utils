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

domain_urls = {
    "clipart": {
        "url": "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip",
        "filename": "clipart.zip",
        "train_url": "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/txt/clipart_train.txt",
        "test_url": "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/txt/clipart_test.txt",
    },
    "infograph": {
        "url": "http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip",
        "filename": "infograph.zip",
        "train_url": "http://csr.bu.edu/ftp/visda/2019/multi-source/txt/infograph_train.txt",
        "test_url": "http://csr.bu.edu/ftp/visda/2019/multi-source/txt/infograph_test.txt",
    },
    "painting": {
        "url": "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip",
        "filename": "painting.zip",
        "train_url": "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/txt/painting_train.txt",
        "test_url": "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/txt/painting_test.txt",
    },
    "quickdraw": {
        "url": "http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip",
        "filename": "quickdraw.zip",
        "train_url": "http://csr.bu.edu/ftp/visda/2019/multi-source/txt/quickdraw_train.txt",
        "test_url": "http://csr.bu.edu/ftp/visda/2019/multi-source/txt/quickdraw_test.txt",
    },
    "real": {
        "url": "http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip",
        "filename": "real.zip",
        "train_url": "http://csr.bu.edu/ftp/visda/2019/multi-source/txt/real_train.txt",
        "test_url": "http://csr.bu.edu/ftp/visda/2019/multi-source/txt/real_test.txt",
    },
    "sketch": {
        "url": "http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip",
        "filename": "sketch.zip",
        "train_url": "http://csr.bu.edu/ftp/visda/2019/multi-source/txt/sketch_train.txt",
        "test_url": "http://csr.bu.edu/ftp/visda/2019/multi-source/txt/sketch_test.txt",
    },
}


class _sub_DomainNet(datasets.ImageFolder, DomainDataset):
    def __init__(
        self,
        root: PathLike,
        train: bool = True,
        session: str = "clipart",
        transform = None,
        target_transform = None,
        download: bool = False,
    ) -> None:

        self.root = os.path.expanduser(root)
        root = os.path.join(root, "DomainNet")
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.session = session
        self.url = domain_urls[session]["url"]
        self.filename = domain_urls[session]["filename"]
        self.train_url = domain_urls[session]["train_url"]
        self.test_url = domain_urls[session]["test_url"]

        if not os.path.exists(root):
            os.makedirs(root)
        if not os.path.isfile(os.path.join(root, self.filename)):
            if not download:
                raise RuntimeError(
                    "Dataset not found. You can use download=True to download it"
                )
            else:
                print("Downloading from " + self.url)
                download_url(self.url, root, filename=self.filename)
        self.fpath = os.path.join(root, session)
        if not os.path.exists(self.fpath):
            os.mkdir(self.fpath)
            with zipfile.ZipFile(os.path.join(root, self.filename), "r") as zf:
                for member in tqdm.tqdm(
                    zf.infolist(), desc=f"Extracting {self.filename}"
                ):
                    try:
                        zf.extract(member, root)
                    except zipfile.error as e:
                        print(f"Error: {e}")
                        exit(1)
        self.train_path = os.path.join(
            self.fpath, "train_data"
        )  # Train is in the class.
        if not os.path.exists(self.train_path):
            os.makedirs(self.train_path)
            download_url(
                self.train_url, self.train_path, filename=self.train_url.split("/")[-1]
            )
            with open(
                os.path.join(self.train_path, self.train_url.split("/")[-1]), "r"
            ) as f:
                for line in f.readlines():
                    line = line.replace("\n", "")
                    path, _ = line.split(" ")
                    dst = os.path.join(self.train_path, path.split("/")[1])
                    if not os.path.exists(dst):
                        os.makedirs(dst)
                    src = os.path.join(self.fpath, "/".join(path.split("/")[1:]))
                    move(src, dst)
        self.test_path = os.path.join(self.fpath, "test_data")
        if not os.path.exists(self.test_path):
            os.makedirs(self.test_path)
            download_url(
                self.test_url, self.test_path, filename=self.test_url.split("/")[-1]
            )
            with open(
                os.path.join(self.test_path, self.test_url.split("/")[-1]), "r"
            ) as f:
                for line in f.readlines():
                    line = line.replace("\n", "")
                    path, _ = line.split(" ")
                    dst = os.path.join(self.test_path, path.split("/")[1])
                    if not os.path.exists(dst):
                        os.makedirs(dst)
                    src = os.path.join(self.fpath, "/".join(path.split("/")[1:]))
                    move(src, dst)
        super(_sub_DomainNet, self).__init__(
            self.train_path if train else self.test_path,
            transform=transform,
            target_transform=target_transform,
        )
        self.domains = [list(domain_urls.keys()).index(session)] * len(self.samples)

    def __getitem__(self, index):
        sample, target = super(_sub_DomainNet, self).__getitem__(index)
        domain = self.domains[index]
        return sample, torch.tensor(target), torch.tensor(domain)


class DomainNet(torch.utils.data.ConcatDataset):
    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):
        self.datasets = []
        for session in domain_urls.keys():
            self.datasets.append(
                _sub_DomainNet(
                    root, train, session, transform, target_transform, download
                )
            )
            print(f"Session: {session} is loaded")
            print(f"Number of samples: {len(self.datasets[-1])}")
        super(DomainNet, self).__init__(self.datasets)
        print(f"Total number of {'Train' if train else 'Test'} samples: {len(self)}")
        self.class_names = [f"{i}" for i in range(345)]  # It is no meaning :)
        self.domain_names = [str(i) for i in range(len(domain_urls.keys()))]
        self.targets = []
        self.domains = []
        for dataset in self.datasets:
            self.targets += [target for target in dataset.targets]
        for dataset in self.datasets:
            self.domains += [domain for domain in dataset.domains]
