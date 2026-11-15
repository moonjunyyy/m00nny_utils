from .dataset import _MetaArrayDataset


class MNISTM(_MetaArrayDataset):
    url = 'https://github.com/liyxi/mnist-m/releases/download/data/'
    folder = 'MNIST-M/processed'
    # One torch file per part: (data, targets).
    images = {True: ('mnist_m_train.pt', 0), False: ('mnist_m_test.pt', 0)}
    labels = {True: ('mnist_m_train.pt', 1), False: ('mnist_m_test.pt', 1)}
    archives = {
        True: ('mnist_m_train.pt.tar.gz', '191ed53db9933bd85cc9700558847391'),
        False: ('mnist_m_test.pt.tar.gz', 'e11cb4d7fff76d7ec588b1134907db59'),
    }
    class_names = tuple(str(i) for i in range(10))

    @classmethod
    def resources_of(cls, train):
        archive, md5 = cls.archives[train]
        return ((cls.url + archive, archive, cls.images[train][0], md5),)
