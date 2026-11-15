import os
from .dataset import _MetaImageDataset
from .DomainDataset import DomainDataset
from ..util.labels import UNLABELED
from ..util.imagefile import Image


class CORe50(DomainDataset):
    resources = (
        ('http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip', 'core50_128x128.zip', 'core50_128x128'),
    )
    # A session is a domain.
    train_session_list = ('s1', 's2', 's4', 's5', 's6', 's8', 's9', 's11')
    test_session_list = ('s3', 's7', 's10')

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, transform, target_transform)
        self.train = train
        self._fetch(download)
        self.session_list = self.train_session_list if train else self.test_session_list
        base = os.path.join(self.root, 'core50_128x128')
        for d, session in enumerate(self.session_list):
            # <session>/<object>/xxx.png as unpacked; earlier versions of this class moved the sessions under
            # train/ and test/, and such a tree is read too.
            places = (os.path.join(base, session), os.path.join(base, 'train' if train else 'test', session))
            directory = next((place for place in places if os.path.isdir(place)), None)
            if directory is None:
                raise RuntimeError(f'Dataset not found: CORe50 session {session} under {base}')
            # Domain of the test dataset is ignored
            self._scan(_MetaImageDataset.extensions, directory, domain=d if train else UNLABELED)
        self._classify()
        self.domain_names = list(self.session_list)
        self._open(Image)
