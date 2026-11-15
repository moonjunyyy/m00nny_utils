import os
from torch.utils.data import Dataset
from ..util.augmentation import Augmentation
from ..util.imagefile import Image


class _MetaDataset(Dataset):
    def __init__(self, root, transform:Augmentation=None, target_transform=None):
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
    def __init__(self, root, transform:Augmentation=None, target_transform=None):
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
