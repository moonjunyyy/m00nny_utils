import os
import gzip
import shutil
import torch
from torch.utils.data import Dataset
from ..util.augmentation import Augmentation
from ..util.arrayfile import ArrayImage, read_arrays
from ..util.download_url import DownloadURL
from ..util.signdir import get_file_hash
from ..util.imagefile import Image
from ..util.videofile import Video
from ..util.audiofile import Audio


class _MetaDataset(Dataset):
    # What the dataset is made of: (url, file name, what unpacking it creates[, md5]), fetched into root by _fetch.
    #   url:           the address of the file, or several (mirrors), tried in order
    #   url None:      the file comes out of an earlier archive of the list
    #   creates None:  the file is not an archive
    #   md5:           checked when the file is downloaded
    resources = ()

    def __init__(self, root, transform:Augmentation=None, target_transform=None):
        self.root = os.path.expanduser(root)
        # The augmentation policy. It is not applied here: it goes to the media with each sample, and the media
        # applies it while decoding (sample[augment]).
        self.augment = transform
        self.target_transform = target_transform
        self.samples = []  # The media of each sample (Image, Video, Audio). This should be populated by subclasses
        self.targets = []  # The class index of each sample
        self.classes = []  # The class names; a target is an index into it

    @property
    def transform(self):
        return self.augment

    @transform.setter
    def transform(self, transform:Augmentation):
        self.augment = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        sample = sample[self.augment]
        target = self.targets[index]
        if self.target_transform:
            target = self.target_transform(target)
        return sample, target

    # A dataset is built in these steps, each used as the dataset needs:
    #   _fetch     the files of `resources` present and unpacked
    #   _scan      the files of a directory as samples, with the class name of each
    #   _classify  class names -> class indices
    #   _split     the train or the test part of a seeded random split
    #   _open      file paths -> media

    def _fetch(self, download, directory=None, resources=None):
        directory = self.root if directory is None else directory
        os.makedirs(directory, exist_ok=True)
        for url, filename, creates, *md5 in (self.resources if resources is None else resources):
            path = os.path.join(directory, filename)
            created = None if creates is None else os.path.join(directory, creates)
            if created is not None and os.path.exists(created):
                continue  # Already unpacked: the archive itself is not needed any more
            if not os.path.isfile(path):
                if url is None or not download:
                    raise RuntimeError(
                        f"Dataset not found: {path}. You can use download=True to download it"
                    )
                self._download(url, path, md5[0] if md5 else None)
            if created is not None:
                try:
                    shutil.unpack_archive(path, directory)
                except shutil.ReadError:
                    # Not an archive of several files: one gzip-compressed file.
                    with gzip.open(path, "rb") as src, open(created + ".part", "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    os.replace(created + ".part", created)

    def _download(self, urls, path, md5=None):
        failures = []
        for url in ((urls,) if isinstance(urls, str) else urls):
            try:
                # Into a partial file first: an interrupted download is not taken for the file.
                DownloadURL(url=url, save_path=path + ".part", prefix=os.path.basename(path)).download()
                if md5 is not None and get_file_hash(path + ".part", "md5") != md5:
                    raise RuntimeError("the downloaded file does not have the expected md5")
                os.replace(path + ".part", path)
                return
            except Exception as e:
                failures.append(f"  {url}: {e}")
                if os.path.exists(path + ".part"):
                    os.remove(path + ".part")
        raise RuntimeError(f"{os.path.basename(path)} could not be downloaded:\n" + "\n".join(failures))

    def _scan(self, extensions, directory=None, class_of=None, keep=None):
        # The structure is assumed to be:
        # directory/class_x/xxx.ext
        # or
        # directory/train/class_x/xxx.ext
        # class_of(path relative to directory) gives the class name where it is not the folder the file is in;
        # keep(path relative to directory) says which files are samples.
        # Directories and files are visited in sorted order, so that the sample indices are the same on every run
        # and machine. Returns the number of samples added.
        directory = self.root if directory is None else directory
        if not os.path.isdir(directory):
            raise RuntimeError(f"Dataset not found: {directory}")
        if class_of is None:
            class_of = lambda path: os.path.basename(os.path.dirname(path))
        before = len(self.samples)
        for subdir, dirs, files in os.walk(directory):
            dirs.sort()
            for file in sorted(files):
                if file.lower().endswith(extensions):
                    file_path = os.path.join(subdir, file)
                    relative = os.path.relpath(file_path, directory)
                    if keep is not None and not keep(relative):
                        continue
                    self.samples.append(file_path)
                    self.targets.append(class_of(relative))
        return len(self.samples) - before

    def _classify(self, classes=None):
        # classes: the class names in index order; by default the names found, sorted.
        self.classes = sorted(set(self.targets)) if classes is None else list(classes)
        index = {name: i for i, name in enumerate(self.classes)}
        self.targets = [index[t] for t in self.targets]

    def _keep(self, indices):
        self.samples = [self.samples[i] for i in indices]
        self.targets = [self.targets[i] for i in indices]

    def _split(self, train, ratio=0.8, seed=0, sort=False):
        # The first `ratio` of a seeded permutation is the train part, the rest the test part.
        order = torch.randperm(len(self.samples), generator=torch.Generator().manual_seed(seed))
        n_train = int(len(self.samples) * ratio)
        order = order[:n_train] if train else order[n_train:]
        if sort:
            order = order.sort().values
        self._keep(order.tolist())

    def _open(self, media):
        # A sample is a file path, or (container path, index, ...) for a record of a container.
        self.samples = [media(*sample) if isinstance(sample, tuple) else media(sample) for sample in self.samples]


class _MetaImageDataset(_MetaDataset):
    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

    def __init__(self, root, transform:Augmentation=None, target_transform=None):
        super().__init__(root, transform, target_transform)
        self._scan(self.extensions)
        self._classify()
        self._open(Image)


class _MetaVideoDataset(_MetaDataset):
    extensions = ('.mp4', '.avi', '.mkv', '.mov', '.webm')

    def __init__(self, root, transform:Augmentation=None, target_transform=None):
        super().__init__(root, transform, target_transform)
        self._scan(self.extensions)
        self._classify()
        self._open(Video)


class _MetaAudioDataset(_MetaDataset):
    extensions = ('.wav', '.flac', '.mp3', '.ogg', '.m4a')

    def __init__(self, root, transform:Augmentation=None, target_transform=None):
        super().__init__(root, transform, target_transform)
        self._scan(self.extensions)
        self._classify()
        self._open(Audio)


class _MetaArrayDataset(_MetaDataset):
    # Images kept as arrays in container files (idx, MAT-file, torch, safetensors, plain records: util.arrayfile).
    # A dataset declares, for its train (True) and test (False) part:
    folder = ''            # where the files are, under root
    archive_folder = None  # where the resources are fetched, under root (None: the same folder)
    images = {}            # part -> (file or files, key of the array of images; None: the only array)
    labels = {}            # part -> (file or files, key of the array of labels)
    layout = None          # the axes of the array of images (see ArrayImage); None: found from its shape
    records = None         # the fields of one record, for files of plain records
    class_names = ()       # the class names, or the file that lists them (one per line)
    axes = {}              # further label axes: attribute -> key of its array, in the files of the labels
    axis_names = {}        # attribute -> the file that lists the names of an axis

    @classmethod
    def resources_of(cls, train):
        return ()

    def __init__(self, root, train=True, transform:Augmentation=None, target_transform=None, download=False):
        super().__init__(root, transform, target_transform)
        self.train = train
        directory = os.path.join(self.root, self.folder)
        self._fetch(
            download,
            directory if self.archive_folder is None else os.path.join(self.root, self.archive_folder),
            self.resources_of(train),
        )
        image_files, image_key = self.images[train]
        label_files, label_key = self.labels[train]
        if isinstance(image_files, str):
            image_files, label_files = (image_files,), (label_files,)
        for axis in self.axes:
            setattr(self, axis, [])
        wanted = None if label_key is None else (label_key, *self.axes.values())
        for image_file, label_file in zip(image_files, label_files):
            arrays = read_arrays(os.path.join(directory, label_file), wanted, self.records)
            targets = self._targets(arrays[next(iter(arrays)) if label_key is None else label_key])
            path = os.path.join(directory, image_file)
            self.samples += [(path, index, image_key, self.layout, self.records) for index in range(len(targets))]
            self.targets += targets
            for axis, key in self.axes.items():
                getattr(self, axis).extend(arrays[key].reshape(-1).astype(int).tolist())
        self.classes = self._names(directory, self.class_names)
        for attribute, names in self.axis_names.items():
            setattr(self, attribute, self._names(directory, names))
        self._open(ArrayImage)

    def _targets(self, labels):
        return labels.reshape(-1).astype(int).tolist()

    @staticmethod
    def _names(directory, names):
        if not isinstance(names, str):
            return list(names)
        with open(os.path.join(directory, names), "r") as f:
            return [line.strip() for line in f if line.strip()]


class _MetaTextDataset(_MetaDataset):
    # There is no text media in util yet (the samples of a dataset are media that decode themselves).
    pass
