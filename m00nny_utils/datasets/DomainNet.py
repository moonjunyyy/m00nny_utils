import os
from .dataset import _MetaImageDataset
from .DomainDataset import DomainDataset
from ..util.imagefile import Image

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


class DomainNet(DomainDataset):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, transform, target_transform)
        self.train = train
        root = os.path.join(self.root, "DomainNet")
        for d, (session, source) in enumerate(domain_urls.items()):
            # <session>.zip unpacks to <session>/<class>/xxx.jpg
            self._fetch(download, root, ((source["url"], source["filename"], session),))
            fpath = os.path.join(root, session)
            moved = os.path.join(fpath, "train_data" if train else "test_data")
            if os.path.isdir(moved):
                # Earlier versions of this class moved the files of each list into train_data/ and test_data/,
                # and such a tree is read as it is.
                self._scan(_MetaImageDataset.extensions, moved, domain=d)
                continue
            # The files of the train or the test part are the ones its list names: "<session>/<class>/<file> <label>"
            list_url = source["train_url" if train else "test_url"]
            list_name = list_url.split("/")[-1]
            self._fetch(download, root, ((list_url, list_name, None),))
            with open(os.path.join(root, list_name), "r") as f:
                listed = set(os.path.normpath(line.split(" ")[0]) for line in f if line.strip())
            self._scan(_MetaImageDataset.extensions, fpath, domain=d,
                       keep=lambda path: os.path.join(session, path) in listed)
        self._classify()
        self.domain_names = list(domain_urls.keys())
        self._open(Image)
