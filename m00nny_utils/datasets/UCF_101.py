import os
from .dataset import _MetaVideoDataset
from .DomainDataset import DomainDataset
from ..util.augmentation import Compose, RandomTemporalCrop
from ..util.videofile import Video


class UCF101(DomainDataset):
    # No download: root/ucf101/videos/<class>/*.avi is expected to be there.
    frames_per_clip = 16

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, transform, target_transform)
        self.train = train
        # List all video paths and labels
        self._scan(_MetaVideoDataset.extensions, os.path.join(self.root, 'ucf101', 'videos'), domain=0)
        self._classify()
        self.domain_names = ['domain_0']
        self._open(Video)
        # A sample is a clip of frames_per_clip frames from a random start, then the given policy.
        clip = RandomTemporalCrop(self.frames_per_clip)
        self.augment = clip if self.augment is None else Compose([clip, self.augment])
