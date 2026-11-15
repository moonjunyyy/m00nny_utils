from .dataset import _MetaArrayDataset

# CIFAR in its binary version: plain records, no pickle.
# https://www.cs.toronto.edu/~kriz/cifar.html


class CIFAR10(_MetaArrayDataset):
    url = 'https://www.cs.toronto.edu/~kriz/'
    archive = ('cifar-10-binary.tar.gz', 'c32a1d4ab5d03f1284b67883e8d87530')
    archive_folder = ''   # the archive is kept in root and unpacks to root/<folder>
    folder = 'cifar-10-batches-bin'
    # A record: the label, then the image as three 32x32 planes (red, green, blue).
    records = (('label', 'u1'), ('image', 'u1', (3, 32, 32)))
    train_files = tuple(f'data_batch_{i}.bin' for i in range(1, 6))
    images = {True: (train_files, 'image'), False: ('test_batch.bin', 'image')}
    labels = {True: (train_files, 'label'), False: ('test_batch.bin', 'label')}
    layout = 'NCHW'
    class_names = 'batches.meta.txt'

    @classmethod
    def resources_of(cls, train):
        archive, md5 = cls.archive
        return ((cls.url + archive, archive, cls.folder, md5),)


class CIFAR100(CIFAR10):
    archive = ('cifar-100-binary.tar.gz', '03b5dce01913d631647c71ecec9e9cb8')
    folder = 'cifar-100-binary'
    # A record: the coarse label (the super class), the fine label (the class), the image.
    records = (('coarse', 'u1'), ('fine', 'u1'), ('image', 'u1', (3, 32, 32)))
    images = {True: ('train.bin', 'image'), False: ('test.bin', 'image')}
    labels = {True: ('train.bin', 'fine'), False: ('test.bin', 'fine')}
    class_names = 'fine_label_names.txt'
    axes = {'super_classes': 'coarse'}
    axis_names = {'super_class_names': 'coarse_label_names.txt'}
