import os
from torch.utils.data import Dataset
from ..util.augmentation import Augmentation
from ..util.imagefile import Image


class _MetaDataset(Dataset):
    def __init__(
        self,
        root,
        transform: Augmentation = None,
        target_transform=None,
        *args, **kwargs
    ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.samples = []  # This should be populated by subclasses

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample, target = self.samples[index]
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            target = self.target_transform(target)
        return sample, target


class _MetaImageDataset(_MetaDataset):
    def __init__(self, root, transform: Augmentation = None, target_transform=None):
        super().__init__(root, transform, target_transform)
        # Parse the root directory to populate, the structure is assumed to be:
        # root/class_x/xxx.ext
        # or
        # root/train/class_x/xxx.ext
        self.targets = []
        for subdir, dirs, files in os.walk(root):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    class_name = os.path.basename(subdir)
                    file_path = os.path.join(subdir, file)
                    self.samples.append(Image(file_path))
                    self.targets.append(class_name)
        self.classes = list(set(self.targets))
        self.targets = [self.classes.index(t) for t in self.targets]
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        sample = self.samples[index]
        sample = sample[self.augment]
        target = self.targets[index]
        if self.target_transform:
            target = self.target_transform(target)
        return sample, target


class _MetaTextDataset(_MetaDataset):
    pass


class _MetaAudioDataset(_MetaDataset):
    pass


class _SubDomainDataset(_MetaDataset):
    pass


class DomainDataset(_MetaDataset):
    """
    A base class for domain datasets.
    """

    def __init__(
        self,
        root: str,
        transform=None,
        target_transform=None,
    ) -> None:
        super().__init__(root, transform, target_transform)
        self.domains = []
        self.targets = []

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )


class multiDatasets(Dataset):
    def __init__(
        self,
        datasets: Iterable[Dataset],
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__()
        self.datasets = []
        self.dataset_lengths = []
        self.classes = []

        for dataset in datasets:
            if not isinstance(dataset, Dataset):
                raise TypeError("dataset should be a Dataset object")
            self.datasets.append(
                dataset(root, train, transform, target_transform, download))
            self.dataset_lengths.append(len(self.datasets[-1]))
            self.classes += len(self.classes)

        self.classes = [str(i) for i in range(self.classes)]
        self.targets = []

        for i, dataset in enumerate(self.datasets):
            for cls in dataset.targets:
                self.targets.append(int(cls) + sum(self.classes[:i]))

    def __getitem__(self, index):
        target = self.targets[index]
        for i, dataset in enumerate(self.datasets):
            if index < self.dataset_lengths[i]:
                return dataset[index], target
            index -= self.dataset_lengths[i]

    def __len__(self):
        return len(self.targets)
