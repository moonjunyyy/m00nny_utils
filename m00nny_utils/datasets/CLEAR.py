import os
from .dataset import _MetaImageDataset
from .DomainDataset import DomainDataset
from ..util.imagefile import Image


class CLEAR(DomainDataset):
    url = 'https://huggingface.co/datasets/elvishelvis6/CLEAR-Continual_Learning_Benchmark/resolve/main/'
    train_resources = ((url + 'clear100-train-image-only.zip', 'clear100-train-image-only.zip', 'train_image_only'),)
    test_resources = ((url + 'clear100-test.zip', 'clear100-test.zip', 'test'),)

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, transform, target_transform)
        self.train = train
        root = os.path.join(self.root, 'CLEAR')
        self._fetch(download, root, self.train_resources if train else self.test_resources)
        # labeled_images/<session>/<class>/xxx.jpg; a session (a period of time) is a domain, in numeric order.
        base = os.path.join(root, 'train_image_only' if train else 'test', 'labeled_images')
        if not os.path.isdir(base):
            raise RuntimeError(f'Dataset not found: {base}')
        sessions = [s for s in os.listdir(base) if os.path.isdir(os.path.join(base, s))]
        sessions.sort(key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else 0, s))
        for d, session in enumerate(sessions):
            self._scan(_MetaImageDataset.extensions, os.path.join(base, session), domain=d)
        self._classify()
        self.domain_names = sessions
        self._open(Image)
