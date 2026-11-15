from .dataset import _MetaArrayDataset


class SVHN(_MetaArrayDataset):
    # The original distribution first; further mirrors of the same files go after it.
    mirrors = ('http://ufldl.stanford.edu/housenumbers/',)
    folder = 'SVHN'
    # One MAT-file per part: X (32, 32, 3, N) the images, y (N, 1) the labels.
    images = {True: ('train_32x32.mat', 'X'), False: ('test_32x32.mat', 'X')}
    labels = {True: ('train_32x32.mat', 'y'), False: ('test_32x32.mat', 'y')}
    md5 = {
        'train_32x32.mat': 'e26dedcc434d2e4c54c9b2d4a06d8373',
        'test_32x32.mat': 'eb5a983be6a315427106f1b164d9cef3',
    }
    layout = 'HWCN'
    class_names = tuple(str(i) for i in range(10))

    @classmethod
    def resources_of(cls, train):
        name = cls.images[train][0]
        return ((tuple(mirror + name for mirror in cls.mirrors), name, None, cls.md5[name]),)

    def _targets(self, labels):
        # The digit 0 is labelled 10 in the files.
        return [label % 10 for label in super()._targets(labels)]
