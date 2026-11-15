import os
from .dataset import _MetaDataset
from ..util.arrayfile import read_arrays
from ..util.imagefile import Image


class Flowers102(_MetaDataset):
    url = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/'
    resources = (
        (url + '102flowers.tgz', '102flowers.tgz', 'jpg'),
        (url + 'imagelabels.mat', 'imagelabels.mat', None),
        (url + 'setid.mat', 'setid.mat', None),
    )
    # The ids of each part in setid.mat.
    parts = {True: 'trnid', False: 'tstid'}

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, transform, target_transform)
        self.train = train
        directory = os.path.join(self.root, 'flowers-102')
        self._fetch(download, directory)
        labels = read_arrays(os.path.join(directory, 'imagelabels.mat'))['labels'].reshape(-1)
        ids = read_arrays(os.path.join(directory, 'setid.mat'))[self.parts[train]].reshape(-1)
        # jpg/image_<id>.jpg; the ids and the labels start at 1.
        for image_id in ids.astype(int).tolist():
            self.samples.append(os.path.join(directory, 'jpg', f'image_{image_id:05d}.jpg'))
            self.targets.append(int(labels[image_id - 1]) - 1)
        self.classes = [str(i) for i in range(102)]
        self._open(Image)
