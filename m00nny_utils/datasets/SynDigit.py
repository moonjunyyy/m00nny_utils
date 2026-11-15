from .MNISTM import MNISTM


class SynDigit(MNISTM):
    # The same form as MNIST-M: one torch file (data, targets) per part.
    url = 'https://github.com/liyxi/synthetic-digits/releases/download/data/'
    folder = 'SynDigit/processed'
    images = {True: ('synth_train.pt', 0), False: ('synth_test.pt', 0)}
    labels = {True: ('synth_train.pt', 1), False: ('synth_test.pt', 1)}
    archives = {
        True: ('synth_train.pt.gz', 'd0e99daf379597e57448a89fc37ae5cf'),
        False: ('synth_test.pt.gz', '669d94c04d1c91552103e9aded0ee625'),
    }
