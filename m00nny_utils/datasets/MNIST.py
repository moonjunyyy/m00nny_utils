from .dataset import _MetaArrayDataset


class MNIST(_MetaArrayDataset):
    mirrors = (
        'https://ossci-datasets.s3.amazonaws.com/mnist/',
        'https://storage.googleapis.com/cvdf-datasets/mnist/',
    )
    folder = 'MNIST/raw'
    images = {True: ('train-images-idx3-ubyte.gz', None), False: ('t10k-images-idx3-ubyte.gz', None)}
    labels = {True: ('train-labels-idx1-ubyte.gz', None), False: ('t10k-labels-idx1-ubyte.gz', None)}
    md5 = {
        'train-images-idx3-ubyte.gz': 'f68b3c2dcbeaaa9fbdd348bbdeb94873',
        'train-labels-idx1-ubyte.gz': 'd53e105ee54ea40749a09fcbcd1e9432',
        't10k-images-idx3-ubyte.gz': '9fb629c4189551a2d022fa330f9573f3',
        't10k-labels-idx1-ubyte.gz': 'ec29112dd5afa0611ce80d1b7f02629c',
    }
    class_names = tuple(str(i) for i in range(10))

    @classmethod
    def resources_of(cls, train):
        # The idx files are read as they are (gzip-compressed): nothing to unpack.
        return tuple(
            (tuple(mirror + name for mirror in cls.mirrors), name, None, cls.md5[name])
            for name in (cls.images[train][0], cls.labels[train][0])
        )
