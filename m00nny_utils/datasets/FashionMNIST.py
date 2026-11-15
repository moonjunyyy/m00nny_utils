from .MNIST import MNIST


class FashionMNIST(MNIST):
    # The same files as MNIST, from its own source.
    mirrors = ('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/',)
    folder = 'FashionMNIST/raw'
    md5 = {
        'train-images-idx3-ubyte.gz': '8d4fb7e6c68d591d4c3dfef9ec88bf0d',
        'train-labels-idx1-ubyte.gz': '25c81989df183df01b3e8a0aad5dffbe',
        't10k-images-idx3-ubyte.gz': 'bef4ecab320f06d8554ea6380940ec79',
        't10k-labels-idx1-ubyte.gz': 'bb300cfdad3c16e7a12a480ee83cd310',
    }
